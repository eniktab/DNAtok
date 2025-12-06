from __future__ import annotations

import inspect
import logging
from collections import deque
from typing import Dict, Iterator, List, Optional, Tuple, Protocol, Any, Union

import numpy as np
import torch

_VALID_DNA_BYTES_UPPER = np.frombuffer(b"ACGTN", dtype=np.uint8)
_VALID_DNA_BYTES_BOTH = np.frombuffer(b"ACGTNacgtn", dtype=np.uint8)

def _as_torch_device(device: object) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device("cuda", device)
    if isinstance(device, str):
        return torch.device(device)
    # Fallback: assume default CUDA
    return torch.device("cuda")


def _is_cuda_device(device: object) -> bool:
    try:
        return torch.device(device).type == "cuda"
    except Exception:
        return False

class TokenizerProtocol(Protocol):
    def encode(self, text: str, **kwargs: Any) -> Any: ...
    def token_to_id(self, token: str) -> Optional[int]: ...
    def convert_tokens_to_ids(self, token: str) -> Optional[int]: ...
    @property
    def vocab(self) -> Dict[str, int]: ...
    def get_vocab(self) -> Dict[str, int]: ...

class EmbedderProtocol(Protocol):
    def embed_tokens(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...
    tokenizer: Any


class DNATok:
    """
    KevTok IDs-path helper.

    Encapsulates the "IDs path" for KevTok:
      - Discovering ASCII→token ID lookup table (LUT) from an embedder/tokenizer
      - Efficient vectorized encoding of equal-length DNA strings to IDs
      - Optional left-padding to a fixed token length (if model requires it)
      - Safe micro-batching of embed_tokens() to avoid 32-bit index math / OOM
      - Persistent pinned CPU staging buffers and host→device (H2D) / compute
        overlap with ping–pong device buffers, with optional int32 H2D to halve
        bandwidth.

    Requirements for the embedder:
      - An `embed_tokens(LongTensor[B,T]) -> Tensor[B,D]` method. If absent,
        `use_ids_path` will be False and callers should use a string-based path.
      - (Recommended) `embedder.tokenizer` compatible with Hugging Face
        Tokenizers / Transformers / Vortex tokenizers. When missing, a
        conservative fixed DNA mapping {A:1,C:2,G:3,T:4,N:0} is used.
    """

    # Default raised from 256k → 1M tokens per call now that 32-bit index
    # math failures are handled with on-the-fly shrinking.
    DEFAULT_IDS_MAX_TOKENS_PER_CALL = 1_048_576

    def __init__(
        self,
        embedder: object,
        ids_max_tokens_per_call: int = DEFAULT_IDS_MAX_TOKENS_PER_CALL,
        logger: Optional[logging.Logger] = None,
        *,
        prefer_int32_h2d: bool = True,
        overlap_h2d_compute: bool = True,
        force_fp32_outputs: bool = True,  # keep legacy behavior by default
        strict_lut_check: bool = True,  # raise on LUT/tokenizer mismatch during discover
        normalize_case: bool = False, # force uppercase inputs
        handle_invalid_chars: bool = False, # map invalid chars to N instead of error
    ) -> None:
        self.embedder = embedder
        self.log = logger or logging.getLogger("DNATok")
        if not self.log.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.log.addHandler(ch)

        # discovered at runtime by discover()
        self.use_ids_path: bool = False
        self.ascii_lut: Optional[np.ndarray] = None  # shape [256] int64
        self.ascii_start_lut: Optional[np.ndarray] = None  # shape [256] int64, for first token if different
        self.id_pad: int = 0
        self.id_N: int = 0
        self.token_len: Optional[int] = None

        # runtime safety cap
        self.ids_max_tokens_per_call: int = int(ids_max_tokens_per_call)

        # performance knobs
        self.prefer_int32_h2d: bool = bool(prefer_int32_h2d)
        self.overlap_h2d_compute: bool = bool(overlap_h2d_compute)
        self.force_fp32_outputs: bool = bool(force_fp32_outputs)
        self.strict_lut_check: bool = bool(strict_lut_check)
        self.normalize_case: bool = bool(normalize_case)
        self.handle_invalid_chars: bool = bool(handle_invalid_chars)

        # persistent staging (CPU) and ping–pong (CUDA)
        self._staging_ids_cpu: Optional[torch.Tensor] = None  # int32 or int64
        self._staging_bytes_cpu: Optional[torch.Tensor] = None  # uint8
        self._dev_ping_i: Optional[torch.Tensor] = None  # int32 or int64
        self._dev_pong_i: Optional[torch.Tensor] = None
        self._dev_ping_l: Optional[torch.Tensor] = None  # int64
        self._dev_pong_l: Optional[torch.Tensor] = None
        self._lut_cuda: Optional[torch.Tensor] = None  # cached 256 LUT on device
        self._lut_start_cuda: Optional[torch.Tensor] = None  # cached start-LUT (if needed)

        # K-mer fast path
        self.kmer_k: Optional[int] = None
        self.kmer_lut: Optional[np.ndarray] = None  # shape [5**k] int64
        self.base5_lut: Optional[np.ndarray] = None  # shape [256] int8 (0-4, -1 for invalid)

    # ---------------------------------------------------------------------
    # Tokenizer utilities (guarded; never assume .vocab is safe)
    # ---------------------------------------------------------------------

    def _maybe_unwrap_tokenizer(self, tok: object) -> object:
        inner = getattr(tok, "tokenizer", None)
        return inner if inner is not None else tok

    def _maybe_single_int(self, obj: object) -> Optional[int]:
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                val = obj.reshape(-1)[0].item()
                if isinstance(val, (int, np.integer)):
                    return int(val)
            return None
        if isinstance(obj, np.ndarray):
            if obj.size == 1:
                val = obj.reshape(-1)[0]
                if isinstance(val, (int, np.integer)):
                    return int(val)
            return None
        if isinstance(obj, (list, tuple)):
            if len(obj) == 1 and isinstance(obj[0], (int, np.integer)):
                return int(obj[0])
        return None

    def _maybe_int_seq(self, obj: object) -> Optional[List[int]]:
        """Best-effort: extract a small flat list of ints from common containers."""
        if obj is None:
            return None
        if isinstance(obj, (int, np.integer)):
            return [int(obj)]
        if isinstance(obj, torch.Tensor):
            # Flatten; safe because discovery uses tiny probes.
            flat = obj.reshape(-1)
            out: List[int] = []
            for x in flat.tolist():
                if isinstance(x, (int, np.integer)):
                    out.append(int(x))
                else:
                    return None
            return out
        if isinstance(obj, np.ndarray):
            flat = obj.reshape(-1)
            out: List[int] = []
            for x in flat:
                if isinstance(x, (int, np.integer)):
                    out.append(int(x))
                else:
                    return None
            return out
        if isinstance(obj, (list, tuple)):
            out: List[int] = []
            for x in obj:
                if isinstance(x, (int, np.integer)):
                    out.append(int(x))
                elif isinstance(x, torch.Tensor) and x.numel() == 1:
                    val = x.reshape(-1)[0].item()
                    if isinstance(val, (int, np.integer)):
                        out.append(int(val))
                    else:
                        return None
                else:
                    return None
            return out
        return None

    def _safe_get_vocab_dict(self, tok: object) -> Optional[Dict[str, int]]:
        get_vocab = getattr(tok, "get_vocab", None)
        if callable(get_vocab):
            try:
                v = get_vocab()
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
        try:
            v = getattr(tok, "vocab")
            if isinstance(v, dict):
                return v
        except Exception:
            pass
        return None

    def _safe_token_to_id(self, tok: object, token_str: str) -> Optional[int]:
        fn = getattr(tok, "token_to_id", None)  # HF tokenizers
        if callable(fn):
            try:
                out = fn(token_str)
                if isinstance(out, int) and out >= 0:
                    return out
            except Exception:
                pass
        cti = getattr(tok, "convert_tokens_to_ids", None)  # Transformers
        if callable(cti):
            try:
                out = cti(token_str)
                if isinstance(out, int) and out >= 0:
                    return out
            except Exception:
                pass
        vocab = self._safe_get_vocab_dict(tok)
        if vocab and token_str in vocab and isinstance(vocab[token_str], int):
            return int(vocab[token_str])
        return None

    def _encode_char_to_single_id(self, tok: object, ch: str) -> Optional[int]:
        enc = getattr(tok, "encode", None)
        if not callable(enc):
            return None
        try:
            try:
                out = enc(ch, add_special_tokens=False)
            except TypeError:
                out = enc(ch)
            ids = out.ids if hasattr(out, "ids") else out
            if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
                return ids[0]
        except Exception:
            pass
        return None

    def _discover_char_id(self, tok: object, ch: str) -> Optional[int]:
        # Prefer the actual encode() path (reflects tokenizer behavior) over raw vocab lookup.
        tid = self._encode_char_to_single_id(tok, ch)
        if isinstance(tid, int):
            return tid

        tid = self._safe_token_to_id(tok, ch)
        if isinstance(tid, int):
            return tid
        code = ord(ch)
        for key in (f"<0x{code:02X}>", f"<0x{code:02x}>"):
            tid = self._safe_token_to_id(tok, key)
            if isinstance(tid, int):
                return tid

        # Evo2-style tokenizer fallback: .tokenize(str) -> List[int] ---
        tok_func = getattr(tok, "tokenize", None)
        if callable(tok_func):
            try:
                token_ids = tok_func(ch)
                tid = self._maybe_single_int(token_ids)
                if tid is not None:
                    return tid
            except Exception:
                pass
        return None

    def _discover_pad_id(self, tok_or_embedder: object) -> Optional[int]:
        # Fixed: actually use the provided object, not self.embedder unconditionally.
        src = tok_or_embedder
        for name in ("pad_id", "pad_token_id"):
            v = getattr(src, name, None)
            if isinstance(v, int):
                return v
        tok = getattr(src, "tokenizer", None)
        if tok is not None:
            tok = self._maybe_unwrap_tokenizer(tok)
            for token_str in ("<pad>", "[PAD]", "PAD", "pad"):
                tid = self._safe_token_to_id(tok, token_str)
                if isinstance(tid, int):
                    return tid
        return None

    def _resolve_base_ids_for_acgtn_safe(self, tok: object, pad_id: int) -> Dict[str, int]:
        char_to_id: Dict[str, int] = {}
        # Try batch encode of single chars (preferred; reflects tokenizer behavior exactly).
        try:
            enc = tok(
                ["A", "C", "G", "T", "N"],
                add_special_tokens=False,
                padding="max_length",
                truncation=False,
                max_length=1,
                return_tensors="pt",
            )
            ids = enc["input_ids"]
            if ids.ndim == 2 and ids.shape[1] >= 1:
                for ch, row in zip("ACGTN", ids):
                    # pick first non-pad token if any
                    non_pad = row[row != pad_id]
                    if non_pad.numel() == 0:
                        continue
                    if non_pad.numel() > 1:
                        # multi-token per base → ids path not safe
                        raise ValueError("Tokenizer emits >1 token per base; ids path unsupported.")
                    tid = int(non_pad[0].item())
                    char_to_id[ch] = tid
        except Exception:
            pass

        for ch in ("A", "C", "G", "T", "N", "a", "c", "g", "t", "n"):
            tid = self._discover_char_id(tok, ch)
            if isinstance(tid, int):
                char_to_id[ch] = tid
        missing = [ch for ch in ("A", "C", "G", "T", "N") if ch not in char_to_id]
        if missing:
            tok_func = getattr(tok, "tokenize", None)
            if callable(tok_func):
                for probe in ("ACGTN", "acgtn"):
                    try:
                        ids_seq = self._maybe_int_seq(tok_func(probe))
                    except Exception:
                        continue
                    if ids_seq is None or len(ids_seq) != len(probe):
                        continue
                    for ch, tid in zip(probe, ids_seq):
                        if ch not in char_to_id:
                            char_to_id[ch] = tid
                    missing = [c for c in ("A", "C", "G", "T", "N") if c not in char_to_id]
                    if not missing:
                        break
        if missing:
            # Fallback: encode repeated single-base strings and require uniform ids.
            for ch in list(missing):
                seq = ch * 8
                try:
                    enc = tok(
                        [seq],
                        add_special_tokens=False,
                        padding="max_length",
                        truncation=False,
                        max_length=len(seq),
                        return_tensors="pt",
                    )
                    ids_t = enc["input_ids"][0]
                    ids_list = [int(x) for x in ids_t.tolist()]
                    if pad_id is not None:
                        ids_list = [v for v in ids_list if v != pad_id]
                    uniq = set(ids_list)
                    if len(uniq) == 1:
                        char_to_id[ch] = uniq.pop()
                except Exception:
                    continue
        for ch in ("A", "C", "G", "T", "N"):
            if ch in char_to_id and ch.lower() not in char_to_id:
                char_to_id[ch.lower()] = char_to_id[ch]
        if "N" not in char_to_id:
            char_to_id["N"] = char_to_id.get("n", pad_id)
        if "n" not in char_to_id:
            char_to_id["n"] = char_to_id["N"]
        return char_to_id

    def _resolve_cont_ids(self, tok: object, start_ids: Dict[str, int]) -> Dict[str, int]:
        """Detect if tokens have different IDs in continuation positions (e.g. BPE merges or sentencepiece)."""
        cont_ids = start_ids.copy()
        # We only care about ACGTN for now
        for ch in ("A", "C", "G", "T", "N", "a", "c", "g", "t", "n"):
            if ch not in start_ids:
                continue
            start_id = start_ids[ch]
            # Probe 2 chars: if the second one differs from start_id, it's a cont id.
            seq = ch * 2
            try:
                enc = tok(
                    [seq],
                    add_special_tokens=False,
                    padding="max_length",
                    truncation=False,
                    max_length=2,
                    return_tensors="pt",
                )
                ids = enc["input_ids"][0]
                # If we got 2 tokens
                if ids.numel() == 2:
                    id0 = int(ids[0].item())
                    id1 = int(ids[1].item())
                    # If id1 is different, record it.
                    # Note: id0 might also be different from start_id if "AA" -> "A" "A" (different context?)
                    # But usually id0 matches start_id.
                    if id1 != start_id:
                        cont_ids[ch] = id1
            except Exception:
                pass
        return cont_ids

    def _rebuild_ascii_luts_from_tokenizer(
        self, tok: object, pad_id: int, fallback_id: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Best-effort LUT reconstruction that probes the tokenizer directly for every byte,
        including continuation positions. Used as a recovery path when lightweight probing
        produced a mismatch.
        """
        tok = self._maybe_unwrap_tokenizer(tok)
        start_lut = np.full(256, int(fallback_id), dtype=np.int64)
        cont_lut = np.full(256, int(fallback_id), dtype=np.int64)

        def _probe_batch(texts: List[str], max_length: int) -> Optional[torch.Tensor]:
            if not callable(tok):
                return None
            try:
                enc = tok(
                    texts,
                    add_special_tokens=False,
                    padding="max_length",
                    truncation=False,
                    max_length=max_length,
                    return_tensors="pt",
                )
                ids = enc.get("input_ids", None) if isinstance(enc, dict) else None
                if ids is None and hasattr(enc, "input_ids"):
                    ids = enc.input_ids
                if ids is None:
                    return None
                ids = torch.as_tensor(ids)
                if ids.ndim != 2:
                    return None
                return ids
            except Exception:
                return None

        # Probe all 256 bytes at once for start positions.
        texts = [chr(b) for b in range(256)]
        ids1 = _probe_batch(texts, max_length=1)
        if ids1 is not None and ids1.shape[0] == 256:
            for b in range(256):
                row = ids1[b]
                non_pad = row[row != pad_id]
                if non_pad.numel() >= 1:
                    start_lut[b] = int(non_pad[0].item())

        # Continuation positions: repeat the character twice and look at the second token.
        texts2 = [chr(b) * 2 for b in range(256)]
        ids2 = _probe_batch(texts2, max_length=2)
        if ids2 is not None and ids2.shape[0] == 256 and ids2.shape[1] >= 2:
            for b in range(256):
                row = ids2[b]
                non_pad = row[row != pad_id]
                if non_pad.numel() >= 2:
                    cont_lut[b] = int(non_pad[1].item())

        # Fill any remaining holes via existing discovery helpers.
        for b in range(256):
            ch = chr(b)
            if start_lut[b] == fallback_id:
                tid = self._discover_char_id(tok, ch)
                if tid is not None:
                    start_lut[b] = int(tid)
            if cont_lut[b] == fallback_id:
                # fall back to start id if present, else the same discovery.
                cont_lut[b] = start_lut[b]

        start_lut = np.asarray(start_lut, dtype=np.int64)
        cont_lut = np.asarray(cont_lut, dtype=np.int64)
        if np.array_equal(start_lut, cont_lut):
            return cont_lut, None
        return cont_lut, start_lut

    def _build_ascii_lut_probe_all(
        self, tok: object, default_id: int, dna_overrides: Dict[str, int]
    ) -> np.ndarray:
        lut = np.full(256, int(default_id), dtype=np.int64)
        for b in range(256):
            ch = chr(b)
            tid = None
            if b in (9, 10, 13) or 32 <= b <= 126:
                tid = self._encode_char_to_single_id(tok, ch)
            if tid is None:
                tid = self._safe_token_to_id(tok, f"<0x{b:02X}>")
            if tid is None:
                tid = self._safe_token_to_id(tok, f"<0x{b:02x}>")
            if tid is None and (b in (9, 10, 13) or 32 <= b <= 126):
                tid = self._safe_token_to_id(tok, ch)
            if isinstance(tid, int):
                lut[b] = int(tid)
        for ch, tid in dna_overrides.items():
            if len(ch) == 1:
                lut[ord(ch)] = int(tid)
        return lut

    def _discover_kmer_structure(self, tok: object, pad_id: int) -> Optional[int]:
        """
        Detect if the tokenizer supports a dense k-mer encoding (k=3..6).
        Returns k if detected, else None.
        """
        def _tokenize_ids(text: str) -> Optional[List[int]]:
            # Prefer callable tokenizer with return_tensors support
            if callable(tok):
                try:
                    enc = tok(
                        [text],
                        add_special_tokens=False,
                        padding="max_length",
                        truncation=False,
                        max_length=len(text),
                        return_tensors="pt",
                    )
                    ids_obj = enc.get("input_ids", None) if isinstance(enc, dict) else None
                    if ids_obj is None and hasattr(enc, "input_ids"):
                        ids_obj = enc.input_ids
                    if ids_obj is not None:
                        ids = torch.as_tensor(ids_obj)[0]
                        ids = ids[ids != pad_id]
                        return [int(x) for x in ids.tolist()]
                except Exception:
                    pass
            # encode
            enc_fn = getattr(tok, "encode", None)
            if callable(enc_fn):
                try:
                    out = enc_fn(text, add_special_tokens=False)
                    ids = out.ids if hasattr(out, "ids") else out
                    ids_list = self._maybe_int_seq(ids)
                    if ids_list is not None:
                        return [x for x in ids_list if x != pad_id]
                except Exception:
                    pass
            # tokenize then map
            tok_fn = getattr(tok, "tokenize", None)
            if callable(tok_fn):
                try:
                    toks = tok_fn(text)
                    ids_list = self._maybe_int_seq(toks)
                    if ids_list is None and isinstance(toks, list):
                        ids_list = []
                        for t in toks:
                            if isinstance(t, (int, np.integer)):
                                ids_list.append(int(t))
                            elif isinstance(t, str):
                                tid = self._safe_token_to_id(tok, t)
                                if tid is None:
                                    ids_list = None
                                    break
                                ids_list.append(int(tid))
                            else:
                                ids_list = None
                                break
                    if ids_list is not None:
                        return [x for x in ids_list if x != pad_id]
                except Exception:
                    pass
            return None

        probe_len = 24  # divisible by 2, 3, 4, 6
        seq = "A" * probe_len
        ids_a = _tokenize_ids(seq)
        if not ids_a:
            return None
        num_tokens = len(ids_a)
        if num_tokens == 0:
            return None
        k = probe_len // num_tokens
        if k * num_tokens != probe_len or k < 2 or k > 6:
            return None

        for b in "CGT":
            ids_b = _tokenize_ids(b * probe_len)
            if ids_b is None or len(ids_b) != num_tokens:
                return None

        mixed = ("ACGT" * 10)[:probe_len]
        ids_m = _tokenize_ids(mixed)
        if ids_m is None or len(ids_m) != num_tokens:
            return None

        return int(k)

    def _build_kmer_lut(self, tok: object, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds:
        1. base5_lut: 256 -> 0..4 (A=0, C=1, G=2, T=3, N=4, others=-1)
        2. kmer_lut: 5^k -> token_id (or -1 if not a single token)
        """
        # 1. Base5 LUT
        base5 = np.full(256, -1, dtype=np.int8)
        for ch, val in zip("ACGTNacgtn", [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]):
            base5[ord(ch)] = val
            
        # 2. K-mer LUT
        # Iterate all 5^k combinations
        size = 5 ** k
        kmer_lut = np.full(size, -1, dtype=np.int64)
        
        import itertools
        bases = "ACGTN"
        
        all_kmers = ["".join(p) for p in itertools.product(bases, repeat=k)]
        
        # We process in batches to avoid OOM or huge lists
        batch_size = 4096
        for i in range(0, len(all_kmers), batch_size):
            batch = all_kmers[i : i + batch_size]
            
            try:
                enc_list = tok(
                    batch,
                    add_special_tokens=False,
                    padding=False,
                    return_tensors=None # Return lists
                )
            except Exception:
                 # Fallback for tokenizers that don't support batching well or return_tensors=None
                 enc_list = [tok.encode(s, add_special_tokens=False) for s in batch]

            # enc_list might be dict with 'input_ids' -> list of lists
            # Handle BatchEncoding or dict
            if hasattr(enc_list, "keys") and "input_ids" in enc_list:
                ids_list = enc_list["input_ids"]
            elif isinstance(enc_list, dict) and "input_ids" in enc_list:
                ids_list = enc_list["input_ids"]
            else:
                ids_list = enc_list 
            
            for idx, row in enumerate(ids_list):
                # Handle potential tensor/numpy array in row
                if hasattr(row, "tolist"):
                    row = row.tolist()
                if len(row) == 1:
                    kmer_lut[i + idx] = row[0]
        
        # Validate LUT
        invalid_count = (kmer_lut == -1).sum()
        if invalid_count > 0:
            self.log.warning(
                f"K-mer LUT incomplete: {invalid_count}/{len(kmer_lut)} k-mers are not single tokens (e.g. containing 'N'). "
                "Fast path will fail for these sequences."
            )
            
        return base5, kmer_lut

    # --------------------------- Discovery ---------------------------------
    def discover(self) -> None:
        # Always clear derived state before probing.
        self.use_ids_path = False
        self.ascii_lut = None
        self.ascii_start_lut = None
        self._lut_cuda = None
        self._lut_start_cuda = None
        self.kmer_k = None
        self.kmer_lut = None
        self.base5_lut = None
        embed_tokens = getattr(self.embedder, "embed_tokens", None)
        if not callable(embed_tokens):
            return
        # Reset k-mer helpers on each discovery
        pad_id = self._discover_pad_id(self.embedder)
        if pad_id is None:
            pad_id = 0
        tok = getattr(self.embedder, "tokenizer", None)
        if tok is None:
            char_to_id = {
                "A": 1,
                "C": 2,
                "G": 3,
                "T": 4,
                "N": 0,
                "a": 1,
                "c": 2,
                "g": 3,
                "t": 4,
                "n": 0,
            }
            n_id = char_to_id["N"]
            self.ascii_lut = np.full(256, n_id, dtype=np.int64)
            for ch, tid in char_to_id.items():
                self.ascii_lut[ord(ch)] = tid
            self.id_pad = int(pad_id)
            self.id_N = int(n_id)
            self.use_ids_path = True
            self.log.warning(
                "IDs path: tokenizer missing; using fixed DNA vocab {A:1,C:2,G:3,T:4,N:0}."
            )
            return
        tok = self._maybe_unwrap_tokenizer(tok)
        
        # 0. Check for K-mer structure first (fastest path if applicable)
        k = self._discover_kmer_structure(tok, pad_id=int(pad_id))
        if k is not None:
            self.log.info(f"K-mer structure detected (k={k}). Building K-mer LUT...")
            try:
                base5, kmer_lut = self._build_kmer_lut(tok, k)
                self.kmer_k = k
                self.base5_lut = base5
                self.kmer_lut = kmer_lut
                self.id_pad = int(pad_id)
                # We don't strictly need id_N for k-mer path if we handle Ns in LUT
                # But let's keep it.
                self.id_N = 0 # Placeholder
                self.use_ids_path = True
                
                # Optional fixed token length hint
                for name in ("model_max_length", "max_position_embeddings", "max_seq_len"):
                    v = getattr(self.embedder, name, None)
                    if isinstance(v, int) and v > 0:
                        self.token_len = v
                        break
                
                self.log.info(f"K-mer fast path enabled (k={k}).")
                return
            except Exception as e:
                self.log.warning(f"Failed to build K-mer LUT: {e}. Falling back to char path.")
                self.kmer_k = None
                self.base5_lut = None
                self.kmer_lut = None

        dna_ids = self._resolve_base_ids_for_acgtn_safe(tok, pad_id)
        dna_cont_ids = self._resolve_cont_ids(tok, dna_ids)
        
        n_id = int(dna_ids.get("N", pad_id))
        n_cont_id = int(dna_cont_ids.get("N", n_id))
        
        # Main LUT uses continuation IDs (for the bulk of the sequence)
        self.ascii_lut = self._build_ascii_lut_probe_all(
            tok, default_id=n_cont_id, dna_overrides=dna_cont_ids
        )
        
        # If any Cont ID differs from Start ID, build start LUT
        if dna_cont_ids != dna_ids:
            self.ascii_start_lut = self._build_ascii_lut_probe_all(
                tok, default_id=n_id, dna_overrides=dna_ids
            )
            self.log.info("Context-dependent tokens detected; using separate start/cont LUTs.")
        else:
            self.ascii_start_lut = None



        self.id_pad = int(pad_id)
        self.id_N = int(n_cont_id)
        # Optional fixed token length hint
        for name in ("model_max_length", "max_position_embeddings", "max_seq_len"):
            v = getattr(self.embedder, name, None)
            if isinstance(v, int) and v > 0:
                self.token_len = v
                break
        self.use_ids_path = True
        self.log.info(
            "KevTok IDs path enabled (embed_tokens): PAD=%d, N=%d",
            self.id_pad,
            self.id_N,
        )
        # Sanity: verify ACGTN mapping is self-consistent using both LUT and tokenizer.
        try:
            self._sanity_verify_mapping(tok)
        except Exception as e:
            self.log.warning("IDs path mapping verification raised: %s", e)
        # Sanity: ensure tokenizer batch path matches LUT encoding exactly.
        try:
            self._sanity_compare_tokenizer_batch(tok)
        except Exception as e:
            # Attempt a heavier recovery by rebuilding LUTs directly from the tokenizer.
            try:
                rebuilt_cont, rebuilt_start = self._rebuild_ascii_luts_from_tokenizer(
                    tok, pad_id=int(pad_id), fallback_id=n_cont_id
                )
                self.ascii_lut = rebuilt_cont
                self.ascii_start_lut = rebuilt_start
                self._sanity_compare_tokenizer_batch(tok)  # re-validate
                self.log.info("Recovered IDs path after LUT rebuild using tokenizer probes.")
            except Exception as e_rebuild:
                if self.strict_lut_check:
                    self.log.warning(
                        "DNATok tokenizer vs LUT mismatch persists (%s). Disabling IDs path.",
                        e_rebuild,
                    )
                    self.use_ids_path = False
                    self.ascii_lut = None
                    self.ascii_start_lut = None
                else:
                    self.log.warning(
                        "DNATok tokenizer vs LUT mismatch: %s. Continuing with IDs path (strict_lut_check=False).",
                        e_rebuild,
                    )

    def _sanity_verify_mapping(self, tok: object) -> None:
        """Ensure LUT-encoded ACGTN matches tokenizer-derived single-char ids.
        Raises on egregious inconsistencies; logs otherwise.
        """
        if self.ascii_lut is None:
            return
        test = self._normalize_and_clean_seqs(["ACGTNacgtn"])[0]
        # Tokenizer single-char discovery
        tok_ids = []
        for ch in test:
            tid = self._discover_char_id(tok, ch)
            tok_ids.append(self.id_N if tid is None else int(tid))
        tok_ids = np.asarray(tok_ids, dtype=np.int64).reshape(1, -1)
        # LUT encoding
        lut_ids = self._encode_batch_numpy([test])
        if not np.array_equal(tok_ids, lut_ids):
            diff = (tok_ids != lut_ids).sum()
            self.log.warning(
                "IDs path: %d/%d LUT bytes differ from tokenizer single-char ids.",
                int(diff),
                tok_ids.size,
            )

    def _sanity_compare_tokenizer_batch(self, tok: object) -> None:
        """
        Compare full-sequence tokenization between the tokenizer string path and
        the fast ASCII LUT path to catch padding/order mistakes.
        """
        tok = self._maybe_unwrap_tokenizer(tok)
        if self.ascii_lut is None:
            return
        if tok is None or not callable(getattr(tok, "__call__", None)):
            return

        # Simple, fixed-length probes to avoid padding ambiguity.
        probes = [
            "ACGTNACG",
            "NNNNNNNN",
            "TGCATGCA",
            "AcGtNaCg",
        ]
        probes = self._normalize_and_clean_seqs(probes)
        T = len(probes[0])
        try:
            enc = tok(
                probes,
                add_special_tokens=False,
                padding="max_length",
                truncation=False,
                max_length=T,
                return_tensors="pt",
            )
            tok_ids = enc["input_ids"].to(dtype=torch.int64, copy=False)
        except Exception as e:
            self.log.warning("Tokenizer batch path unavailable for sanity check: %s", e)
            return

        lut_ids = self._encode_batch_numpy(probes)
        if tok_ids.shape != lut_ids.shape:
            raise ValueError(
                f"Tokenizer vs LUT shape mismatch: {tuple(tok_ids.shape)} vs {tuple(lut_ids.shape)}"
            )
        lut_ids_t = torch.as_tensor(lut_ids, dtype=tok_ids.dtype, device=tok_ids.device)
        if not torch.equal(tok_ids, lut_ids_t):
            diff_mask = (tok_ids != lut_ids_t)
            diff_count = diff_mask.sum().item()
            
            if diff_count > 0:
                # Use logger if available, else print
                log_fn = self.log.error if self.log else print
                log_fn(f"DEBUG: Mismatch count: {diff_count}")
                log_fn(f"DEBUG: tok_ids shape: {tok_ids.shape}")
                log_fn(f"DEBUG: lut_ids shape: {lut_ids.shape}")
                
                # Print first few mismatches
                mismatch_indices = diff_mask.nonzero(as_tuple=False)
                for idx in mismatch_indices[:10]:
                    b, t = idx.tolist()
                    log_fn(f"DEBUG: Mismatch at ({b}, {t}): tok={tok_ids[b, t].item()}, lut={lut_ids[b, t]}")
                    if b < len(probes) and t < len(probes[b]):
                        log_fn(f"DEBUG: Char was '{probes[b][t]}'")
                
                log_fn(f"DEBUG: Probes: {probes}")
                log_fn(f"DEBUG: Tok IDs: \n{tok_ids}")
                log_fn(f"DEBUG: LUT IDs: \n{lut_ids}")

            raise ValueError(f"Tokenizer vs LUT mismatch on {diff_count} tokens.")

    def _encode_batch_kmer_numpy(self, seqs: List[str]) -> np.ndarray:
        """
        Fast path for K-mer tokenization.
        1. Map chars to 0..4 (base 5).
        2. Reshape to (B, T/k, k).
        3. Pack to integers.
        4. Lookup in kmer_lut.
        """
        if self.kmer_k is None or self.kmer_lut is None or self.base5_lut is None:
            raise RuntimeError("K-mer path not initialized.")
            
        k = self.kmer_k
        T = len(seqs[0])
        for s in seqs:
            if len(s) != T:
                raise ValueError("All sequences in a batch must have equal length.")
        if T % k != 0:
            # Fallback to slow path if length not divisible by k
            # (Or we could handle partials, but for now fallback)
            raise ValueError(f"Sequence length {T} not divisible by k={k}.")
             
        # 1. Map to base 5
        # We reuse the ascii_lut logic but with base5_lut
        
        # Flatten and map
        if self.normalize_case:
            # Upper-case everything first
            # Note: this might be slow for huge batches, but robustness is requested.
            # We can do it per-string or joined. Joined is likely faster for numpy.
            buf = ("".join(seqs)).upper().encode("ascii", errors="replace")
        else:
            buf = ("".join(seqs)).encode("ascii", errors="replace")
        arr = np.frombuffer(buf, dtype=np.uint8)
        
        # Check size
        if arr.size != len(seqs) * T:
             # Handle unicode/weird chars by re-encoding individually
             out = np.empty((len(seqs), T), dtype=np.uint8)
             for i, s in enumerate(seqs):
                 out[i, :] = np.frombuffer(s.encode("ascii", errors="replace"), dtype=np.uint8)[:T]
             arr = out.reshape(-1)
        
        # Map to 0..4
        # base5_lut is int8.
        mapped = self.base5_lut[arr] # [Total_Chars]
        
        # Check for invalid chars (-1)
        if np.any(mapped < 0):
            if self.handle_invalid_chars:
                # Map invalid chars to N (4)
                mapped[mapped < 0] = 4
            else:
                raise ValueError("Invalid characters found for K-mer path.")
            
        # Reshape to (B, T/k, k)
        B = len(seqs)
        mapped = mapped.reshape(B, T // k, k)
        
        # Pack
        # index = d0 * 5^(k-1) + ... + dk * 5^0
        # Vectorized packing
        mapped = mapped.astype(np.int64)
        
        # Precompute powers of 5
        powers = np.power(5, np.arange(k - 1, -1, -1), dtype=np.int64)
        
        # Dot product over the last dimension
        packed = np.dot(mapped, powers)
            
        # Lookup
        ids = self.kmer_lut[packed]
        
        # Check for -1 (invalid k-mers, e.g. "AN" if tokenizer splits it)
        if np.any(ids < 0):
            raise ValueError("Sequence contains K-mers not supported by single tokens (e.g. split tokens).")
            
        return ids

    # ------------------------------ Encoding -------------------------------

    def _normalize_and_clean_seqs(self, seqs: List[str]) -> List[str]:
        """
        Apply optional case normalization and invalid-char mapping before encoding.
        Keeps length unchanged while ensuring downstream paths see consistent text.
        """
        if not seqs:
            return seqs
        need_norm = self.normalize_case
        need_clean = self.handle_invalid_chars
        if not (need_norm or need_clean):
            return seqs

        cleaned: List[str] = []
        allowed_upper = {"A", "C", "G", "T", "N"}
        allowed_all = allowed_upper | {"a", "c", "g", "t", "n"}
        for s in seqs:
            text = s.upper() if need_norm else s
            if need_clean:
                allowed = allowed_upper if self.normalize_case else allowed_all
                if not all(ch in allowed for ch in text):
                    text = "".join(ch if ch in allowed else "N" for ch in text)
            cleaned.append(text)
        return cleaned


    def _encode_batch_numpy(self, seqs: List[str]) -> np.ndarray:
        ids = self._encode_batch_ascii_lut_numpy(seqs, self.ascii_lut)
        if self.ascii_start_lut is not None and ids.shape[1] > 0:
            start_ids = self._encode_batch_ascii_lut_numpy(
                [s[:1] for s in seqs], self.ascii_start_lut
            )
            ids[:, 0] = start_ids[:, 0]
        return ids

    def _tokenize_batch_cpu(
        self, seqs: List[str], *, dtype: torch.dtype = torch.long, pin: bool = True
    ) -> torch.Tensor:
        """
        Slow fallback: use the embedder tokenizer to produce CPU IDs when the LUT path
        is unavailable. Keeps padding semantics aligned with embed_from_strings().
        """
        if not seqs:
            raise ValueError("No sequences provided for tokenization.")
        seqs = self._normalize_and_clean_seqs(list(seqs))
        tok = getattr(self.embedder, "tokenizer", None)
        if tok is None:
            raise RuntimeError("IDs path disabled and no tokenizer found on embedder.")
        tok = self._maybe_unwrap_tokenizer(tok)

        def _extract_ids(enc_obj: object) -> Optional[List[int]]:
            for attr in ("ids", "input_ids"):
                ids_val = getattr(enc_obj, attr, None)
                ids_seq = self._maybe_int_seq(ids_val)
                if ids_seq is not None:
                    return ids_seq
            return self._maybe_int_seq(enc_obj)

        def _tokenize_via_encode_api() -> Optional[List[List[int]]]:
            enc_batch = getattr(tok, "encode_batch", None)
            if callable(enc_batch):
                try:
                    try:
                        encs = enc_batch(seqs, add_special_tokens=False)
                    except TypeError:
                        encs = enc_batch(seqs)
                    out: List[List[int]] = []
                    for enc in encs:
                        ids_seq = _extract_ids(enc)
                        if ids_seq is None:
                            out = None
                            break
                        out.append(ids_seq)
                    if out is not None:
                        return out
                except Exception:
                    pass

            enc_single = getattr(tok, "encode", None)
            if callable(enc_single):
                out: List[List[int]] = []
                for s in seqs:
                    try:
                        try:
                            enc = enc_single(s, add_special_tokens=False)
                        except TypeError:
                            enc = enc_single(s)
                    except Exception:
                        out = None
                        break
                    ids_seq = _extract_ids(enc)
                    if ids_seq is None:
                        out = None
                        break
                    out.append(ids_seq)
                if out is not None:
                    return out

            tok_func = getattr(tok, "tokenize", None)
            if callable(tok_func):
                out: List[List[int]] = []
                for s in seqs:
                    try:
                        toks = tok_func(s)
                    except Exception:
                        return None
                    ids_seq = self._maybe_int_seq(toks)
                    if ids_seq is None:
                        ids_seq = []
                        if not isinstance(toks, (list, tuple, torch.Tensor, np.ndarray)):
                            return None
                        for t in toks:
                            if isinstance(t, (int, np.integer)):
                                ids_seq.append(int(t))
                                continue
                            if isinstance(t, str):
                                tid = self._safe_token_to_id(tok, t)
                                if tid is None:
                                    ids_seq = None
                                    break
                                ids_seq.append(int(tid))
                                continue
                            ids_seq = None
                            break
                    if ids_seq is None:
                        return None
                    out.append(ids_seq)
                return out
            return None

        # Enforce left-padding if token_len is set, to match DNATok's contract.
        # This is critical because some tokenizers (e.g. Nucleotide Transformer) default to right-padding,
        # but DNATok's optimized paths (IDs/Bytes) always left-pad.
        old_side = getattr(tok, "padding_side", None)
        if self.token_len and old_side != "left":
            try:
                tok.padding_side = "left"
            except Exception:
                pass
                
        try:
            enc = None
            if callable(tok):
                try:
                    enc = tok(
                        seqs,
                        add_special_tokens=False,
                        padding="max_length" if self.token_len else True,
                        truncation=False,
                        max_length=self.token_len if self.token_len else None,
                        return_tensors="pt",
                    )
                except TypeError:
                    enc = None

            if enc is None:
                ids_list = _tokenize_via_encode_api()
                if ids_list is None or len(ids_list) == 0:
                    raise RuntimeError(
                        "Tokenizer is not callable and no encode/encode_batch/tokenize path produced IDs."
                    )
                target_len = max(len(x) for x in ids_list)
                padding_side = getattr(tok, "padding_side", "right")
                if padding_side not in ("left", "right"):
                    padding_side = "right"
                if self.token_len:
                    if any(len(x) > int(self.token_len) for x in ids_list):
                        raise ValueError("Tokenizer output longer than configured token_len.")
                    target_len = max(target_len, int(self.token_len))
                    padding_side = "left"
                pad_val = int(getattr(self, "id_pad", 0))
                ids = torch.full((len(ids_list), target_len), pad_val, dtype=dtype)
                for i, seq_ids in enumerate(ids_list):
                    cur = torch.as_tensor(seq_ids, dtype=dtype)
                    cur_len = int(cur.numel())
                    if cur_len > target_len:
                        raise ValueError("Tokenizer output length exceeds padded target.")
                    if padding_side == "left" and cur_len < target_len:
                        ids[i, target_len - cur_len :] = cur
                    else:
                        ids[i, :cur_len] = cur
                if pin:
                    try:
                        ids = ids.pin_memory()
                    except Exception:
                        pass
                return ids
        finally:
            # Restore padding side
            if self.token_len and old_side != "left":
                try:
                    tok.padding_side = old_side
                except Exception:
                    pass
        ids = enc["input_ids"]
        if ids.ndim != 2:
            raise RuntimeError(f"Tokenizer returned unexpected ids shape: {tuple(ids.shape)}")
        if ids.device.type != "cpu":
            ids = ids.cpu()
        ids = ids.to(dtype=dtype, copy=False)
        if pin:
            try:
                ids = ids.pin_memory()
            except Exception:
                pass
        return ids

    def _encode_batch_ascii_lut_numpy(self, seqs: List[str], lut: np.ndarray) -> np.ndarray:
        if lut is None:
            raise RuntimeError("DNATok.discover() must be called before encoding.")
        if not seqs:
            raise ValueError("No sequences provided for encoding.")
        T = len(seqs[0])
        for s in seqs:
            if len(s) != T:
                raise ValueError("All sequences in a batch must have equal length.")

        # Normalize case if requested before building the ASCII buffer.
        joined = "".join(seqs)
        if self.normalize_case:
            joined = joined.upper()
        buf = joined.encode("ascii", errors="replace")

        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size != len(seqs) * T:
            out = np.empty((len(seqs), T), dtype=np.uint8)
            for i, s in enumerate(seqs):
                encoded = s.upper() if self.normalize_case else s
                out[i, :] = np.frombuffer(
                    encoded.encode("ascii", errors="replace"), dtype=np.uint8
                )[:T]
            arr = out
        else:
            arr = arr.reshape(len(seqs), T)
        if self.handle_invalid_chars:
            valid_bytes = _VALID_DNA_BYTES_UPPER if self.normalize_case else _VALID_DNA_BYTES_BOTH
            mask = ~np.isin(arr, valid_bytes)
            if mask.any():
                arr = np.array(arr, copy=True)
                arr[mask] = ord("N")
        return lut[arr]  # int64 [B,T]

    def _maybe_left_pad(self, ids_np: np.ndarray) -> np.ndarray:
        if self.token_len is not None and self.token_len > ids_np.shape[1]:
            pad = self.token_len - ids_np.shape[1]
            ids_np = np.pad(ids_np, ((0, 0), (pad, 0)), constant_values=self.id_pad)
        return ids_np

    def encode_batch_to_ids(self, seqs: List[str]) -> torch.Tensor:
        """Backward-compatible path: returns CPU pinned int64 [B,T]."""
        if not seqs:
            raise ValueError("No sequences provided.")
        seqs = self._normalize_and_clean_seqs(list(seqs))
        if not self.use_ids_path:
            return self._tokenize_batch_cpu(seqs, dtype=torch.long, pin=True)
             
        if self.kmer_k is not None:
            try:
                ids_np = self._maybe_left_pad(self._encode_batch_kmer_numpy(seqs))
                ids_cpu = torch.as_tensor(ids_np, dtype=torch.long)
                try:
                    ids_cpu = ids_cpu.pin_memory()
                except Exception:
                    pass
                return ids_cpu
            except ValueError:
                # Prefer ASCII LUT path if available before falling back to tokenizer.
                if self.ascii_lut is not None:
                    ids_np = self._maybe_left_pad(self._encode_batch_numpy(seqs))
                    ids_cpu = torch.as_tensor(ids_np, dtype=torch.long)
                    try:
                        ids_cpu = ids_cpu.pin_memory()
                    except Exception:
                        pass
                    return ids_cpu
                # Fallback to slow path if k-mer encoding fails (e.g. invalid chars or length)
                return self._tokenize_batch_cpu(seqs, dtype=torch.long, pin=True)

        if self.ascii_lut is None:
            return self._tokenize_batch_cpu(seqs, dtype=torch.long, pin=True)
        ids_np = self._maybe_left_pad(self._encode_batch_numpy(seqs))
        ids_cpu = torch.as_tensor(ids_np, dtype=torch.long)
        try:
            ids_cpu = ids_cpu.pin_memory()
        except Exception:
            pass
        return ids_cpu

    # New: persistent pinned staging (int32 or int64)
    def encode_batch_to_ids_staging(
        self, seqs: List[str], *, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Return persistent pinned CPU tensor (int32 by default for H2D), reused across calls."""
        if dtype is None:
            dtype = torch.int32 if self.prefer_int32_h2d else torch.long

        if not seqs:
            raise ValueError("No sequences provided.")
        seqs = self._normalize_and_clean_seqs(list(seqs))

        ids_np: Optional[np.ndarray] = None
        
        # Try fast paths if enabled
        if self.use_ids_path:
            # 1. K-mer path
            if self.kmer_k is not None:
                try:
                    ids_np = self._encode_batch_kmer_numpy(seqs)
                    ids_np = self._maybe_left_pad(ids_np)
                except ValueError:
                    # Fallback to other paths
                    pass
            
            # 2. ASCII LUT path
            if ids_np is None and self.ascii_lut is not None:
                ids_np = self._encode_batch_numpy(seqs)
                ids_np = self._maybe_left_pad(ids_np)

        # 3. Fallback to tokenizer
        if ids_np is None:
            ids_cpu = self._tokenize_batch_cpu(seqs, dtype=dtype, pin=True)
            # Copy to staging
            want_shape = tuple(ids_cpu.shape)
            if (
                self._staging_ids_cpu is None
                or self._staging_ids_cpu.shape != want_shape
                or self._staging_ids_cpu.dtype != dtype
            ):
                self._staging_ids_cpu = torch.empty(want_shape, dtype=dtype)
                try:
                    self._staging_ids_cpu = self._staging_ids_cpu.pin_memory()
                except Exception:
                    pass
            self._staging_ids_cpu.copy_(ids_cpu, non_blocking=False)
            return self._staging_ids_cpu

        # If we got ids_np, copy to staging
        want_shape = (ids_np.shape[0], ids_np.shape[1])
        if (
            self._staging_ids_cpu is None
            or self._staging_ids_cpu.shape != want_shape
            or self._staging_ids_cpu.dtype != dtype
        ):
            # (Re)allocate pinned staging
            self._staging_ids_cpu = torch.empty(want_shape, dtype=dtype)
            try:
                self._staging_ids_cpu = self._staging_ids_cpu.pin_memory()
            except Exception:
                pass
        
        if dtype == torch.int32:
            # Avoid double copy if possible, but ids_np is int64 usually.
            # torch.as_tensor will copy if dtype mismatch.
            self._staging_ids_cpu.copy_(
                torch.as_tensor(ids_np, dtype=torch.int32), non_blocking=False
            )
        else:
            self._staging_ids_cpu.copy_(
                torch.as_tensor(ids_np, dtype=torch.long), non_blocking=False
            )
        return self._staging_ids_cpu

    # -------------------------- H2D/Compute overlap ------------------------
    def _ensure_device_pingpong(self, micro_bs: int, T: int, device: object, use_i32: bool) -> None:
        dev = _as_torch_device(device)
        # If a fixed token length is known and larger than T, size to token_len once.
        alloc_T = max(int(T), int(self.token_len) if self.token_len else int(T))
        dtype_i = torch.int32 if use_i32 else torch.long
        # Upload buffers for copy stream
        for name in ("_dev_ping_i", "_dev_pong_i"):
            buf = getattr(self, name)
            if (
                buf is None
                or buf.shape != (micro_bs, alloc_T)
                or buf.dtype != dtype_i
                or buf.device != dev
            ):
                new_buf = torch.empty((micro_bs, alloc_T), dtype=dtype_i, device=dev)
                # Initialize left pad region to pad_id to avoid stale values when token_len > T.
                new_buf.fill_(int(self.id_pad))
                setattr(self, name, new_buf)
        # Long views for embedder call
        for name in ("_dev_ping_l", "_dev_pong_l"):
            buf = getattr(self, name)
            if (
                buf is None
                or buf.shape != (micro_bs, alloc_T)
                or buf.dtype != torch.long
                or buf.device != dev
            ):
                new_buf = torch.empty((micro_bs, alloc_T), dtype=torch.long, device=dev)
                new_buf.fill_(int(self.id_pad))
                setattr(self, name, new_buf)

    def _ids_micro_bs_for_T(self, T: int, emb_batch: int) -> int:
        tokens_budget = max(1, int(self.ids_max_tokens_per_call))
        per_call = max(1, tokens_budget // max(1, T))
        return max(1, min(int(emb_batch), per_call))

    def iter_embed_tokens_in_slices(
        self,
        ids_cpu: torch.Tensor,
        emb_batch: int,
        device: object = "cuda",
    ) -> Iterator[torch.Tensor]:
        """
        Baseline safe streaming without overlap (kept for backwards-compatibility).
        Yields CUDA activations [cur_bs, D]. Shrinks on index-math or OOM.
        """
        if not self.use_ids_path:
            # Fallback to normal tokenizer
            # Note: ids_cpu argument is expected to be tensor, but if use_ids_path is False,
            # the caller shouldn't be calling this with tensor ids derived from DNATok?
            # Actually, if use_ids_path is False, encode_batch_to_ids fails.
            # So this method is only called if use_ids_path is True OR if we change the contract.
            # But wait, embed_from_strings calls this.
            raise RuntimeError("IDs path not available.")
        if _is_cuda_device(device) and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
        B, T = int(ids_cpu.shape[0]), int(ids_cpu.shape[1])
        start_idx = 0
        micro_bs = self._ids_micro_bs_for_T(T, emb_batch)
        kwargs_template = {}
        try:
            sig = inspect.signature(self.embedder.embed_tokens)
            if "rc_invariant" in sig.parameters:
                kwargs_template["rc_invariant"] = False
        except Exception:
            pass
        dev = _as_torch_device(device)
        while start_idx < B:
            end_idx = min(B, start_idx + micro_bs)
            sub_cpu = ids_cpu[start_idx:end_idx]
            cur_bs = int(sub_cpu.shape[0])
            while True:
                try:
                    if (
                        self.prefer_int32_h2d
                        and _is_cuda_device(dev)
                        and sub_cpu.dtype == torch.int32
                    ):
                        # Preserve int32 H2D bandwidth, then cast once on device.
                        sub_dev = sub_cpu.to(device=dev, dtype=torch.int32, non_blocking=True)
                        sub_dev = sub_dev.to(dtype=torch.long)
                    else:
                        sub_dev = sub_cpu.to(device=dev, dtype=torch.long, non_blocking=True)
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=_is_cuda_device(dev),
                    ):
                        out = self.embedder.embed_tokens(sub_dev, **kwargs_template)
                    if out.device != dev:
                        out = out.to(device=dev, non_blocking=True)
                    if self.force_fp32_outputs and out.dtype != torch.float32:
                        out = out.float()
                    yield out
                    break
                except RuntimeError as e:
                    msg = str(e).lower()
                    triggers = (
                        "canuse32bitindexmath" in msg
                        or "32-bit index" in msg
                        or ("conv1d" in msg and "index" in msg)
                        or "out of memory" in msg
                    )
                    if triggers and cur_bs > 1:
                        new_bs = max(1, cur_bs // 2)
                        if new_bs == cur_bs:
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        cur_bs = new_bs
                        sub_cpu = ids_cpu[start_idx : start_idx + cur_bs]
                        continue
                    else:
                        raise
            start_idx += cur_bs

    def iter_embed_tokens_pipelined(
        self,
        ids_cpu: torch.Tensor,
        emb_batch: int,
        device: object = "cuda",
        *,
        use_int32_h2d: Optional[bool] = None,
    ) -> Iterator[torch.Tensor]:
        """
        Overlapping streamer: copies the next micro-batch on a separate CUDA
        stream while the current micro-batch computes on the default stream. If
        `use_int32_h2d` is True, host→device uses int32 then casts to int64 on
        device just before `embed_tokens`.

        Falls back to iter_embed_tokens_in_slices for non-CUDA devices or on
        failure mid-stream (without duplicating already-emitted outputs).
        """
        if not self.use_ids_path:
            raise RuntimeError("IDs path not available.")
        if not _is_cuda_device(device) or not torch.cuda.is_available():
            # Nothing to overlap; just stream baseline.
            yield from self.iter_embed_tokens_in_slices(ids_cpu, emb_batch, device=device)
            return

        dev = _as_torch_device(device)
        if use_int32_h2d is None:
            use_int32_h2d = self.prefer_int32_h2d

        B, T = int(ids_cpu.shape[0]), int(ids_cpu.shape[1])
        if B == 0:
            return

        micro_bs = self._ids_micro_bs_for_T(T, emb_batch)
        self._ensure_device_pingpong(micro_bs, T, dev, use_i32=use_int32_h2d)

        copy_stream = torch.cuda.Stream(device=dev)
        ready_ping = torch.cuda.Event()
        ready_pong = torch.cuda.Event()

        kwargs_template = {}
        try:
            sig = inspect.signature(self.embedder.embed_tokens)
            if "rc_invariant" in sig.parameters:
                kwargs_template["rc_invariant"] = False
        except Exception:
            pass

        # queue of (lo, hi, bs, use_ping) — now a deque for O(1) pops
        scheduled: deque[Tuple[int, int, int, bool]] = deque()

        def schedule_h2d(lo: int, hi: int, into_ping: bool) -> None:
            cur_bs = max(0, hi - lo)
            if cur_bs <= 0:
                return
            cur = ids_cpu[lo:hi]
            # write targets (consider token_len)
            alloc_T = max(T, int(self.token_len) if self.token_len else T)
            with torch.cuda.stream(copy_stream):
                if use_int32_h2d:
                    if cur.dtype != torch.int32:
                        cur = cur.to(torch.int32, copy=True)
                    dev_i = self._dev_ping_i if into_ping else self._dev_pong_i
                    dev_l = self._dev_ping_l if into_ping else self._dev_pong_l
                    # right-align if we have larger alloc_T due to token_len
                    dev_i[:cur_bs, -T:].copy_(cur, non_blocking=True)
                    # cast to long on device (no extra H2D)
                    dev_l[:cur_bs, -T:].copy_(dev_i[:cur_bs, -T:].to(torch.long))
                    (ready_ping if into_ping else ready_pong).record(copy_stream)
                else:
                    if cur.dtype != torch.long:
                        cur = cur.to(torch.long, copy=True)
                    dev_l = self._dev_ping_l if into_ping else self._dev_pong_l
                    dev_l[:cur_bs, -T:].copy_(cur, non_blocking=True)
                    (ready_ping if into_ping else ready_pong).record(copy_stream)
            scheduled.append((lo, hi, cur_bs, into_ping))

        # Prime up to two micro-batches (ping then pong)
        next_lo = 0
        use_ping = True
        while next_lo < B and len(scheduled) < 2:
            next_hi = min(B, next_lo + micro_bs)
            schedule_h2d(next_lo, next_hi, into_ping=use_ping)
            use_ping = not use_ping
            next_lo = next_hi

        # Main pipeline loop
        while scheduled:
            lo, hi, cur_bs, use_ping = scheduled.popleft()
            if cur_bs <= 0:
                continue
            ready_ev = ready_ping if use_ping else ready_pong
            dev_l_full = self._dev_ping_l if use_ping else self._dev_pong_l
            # effective slice accounts for potential left-padding area
            dev_slice = dev_l_full[:cur_bs]

            # Ensure H2D copy for this batch is visible to default stream
            torch.cuda.current_stream(device=dev).wait_event(ready_ev)

            # While we compute on this buffer, schedule another batch (if any)
            if next_lo < B:
                next_hi = min(B, next_lo + micro_bs)
                # Reuse the buffer we are *not* currently using.
                schedule_h2d(next_lo, next_hi, into_ping=not use_ping)
                next_lo = next_hi

            # Compute on default stream
            try:
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    out = self.embedder.embed_tokens(dev_slice, **kwargs_template)
                if out.device != dev:
                    out = out.to(device=dev, non_blocking=True)
                if self.force_fp32_outputs and out.dtype != torch.float32:
                    out = out.float()
                yield out
            except RuntimeError as e:
                # Fallback: if we trip index-math / OOM, stream the remainder
                # (from `lo` onwards) using the baseline helper. Already-emitted
                # batches are not recomputed.
                self.log.warning(
                    "KevTok pipelined path failed (%s); falling back to baseline from index %d.",
                    str(e),
                    lo,
                )
                remaining = ids_cpu[lo:]
                if remaining.numel() > 0:
                    yield from self.iter_embed_tokens_in_slices(
                        remaining, emb_batch, device=dev
                    )
                return

        torch.cuda.synchronize(dev)

    # -------------------- ASCII bytes → ids (device) -----------------------
    def encode_batch_to_ascii_bytes(self, seqs: List[str]) -> torch.Tensor:
        if not seqs:
            raise ValueError("No sequences provided.")
        seqs = self._normalize_and_clean_seqs(list(seqs))
        T = len(seqs[0])
        for s in seqs:
            if len(s) != T:
                raise ValueError("All sequences must have equal length.")

        # Materialize a writable view to avoid torch's warning about non-writable NumPy buffers.
        # The copy is cheap relative to the embedding work and keeps the staging tensor quiet.
        buf = ("".join(seqs)).encode("ascii", errors="replace")
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(len(seqs), T).copy()
        if self.handle_invalid_chars:
            valid_bytes = _VALID_DNA_BYTES_UPPER if self.normalize_case else _VALID_DNA_BYTES_BOTH
            mask = ~np.isin(arr, valid_bytes)
            if mask.any():
                arr[mask] = ord("N")

        if self._staging_bytes_cpu is None or self._staging_bytes_cpu.shape != arr.shape:
            self._staging_bytes_cpu = torch.empty(arr.shape, dtype=torch.uint8)
            try:
                self._staging_bytes_cpu = self._staging_bytes_cpu.pin_memory()
            except Exception:
                pass

        # Source is now writable → torch.from_numpy(arr) is safe and quiet.
        self._staging_bytes_cpu.copy_(torch.from_numpy(arr))
        return self._staging_bytes_cpu

    def _map_ascii_bytes_to_ids_cuda(
        self, ascii_bytes_cpu: torch.Tensor, device: object
    ) -> torch.Tensor:
        if self.ascii_lut is None:
            raise RuntimeError("discover() must be called before mapping bytes→ids.")
        if ascii_bytes_cpu.dtype != torch.uint8 or ascii_bytes_cpu.device.type != "cpu":
            raise TypeError("ascii_bytes_cpu must be CPU uint8 tensor.")
        if ascii_bytes_cpu.ndim != 2:
            raise ValueError(f"ascii_bytes_cpu must be 2D [B,T], got shape {tuple(ascii_bytes_cpu.shape)}.")

        # Normalize/correct bytes on CPU to ensure consistent mapping.
        was_pinned = ascii_bytes_cpu.is_pinned()
        need_clone = False
        if self.normalize_case:
            lower_mask = (ascii_bytes_cpu >= ord("a")) & (ascii_bytes_cpu <= ord("z"))
            if lower_mask.any():
                need_clone = True
        if self.handle_invalid_chars:
            valid_bytes = torch.as_tensor(
                _VALID_DNA_BYTES_UPPER if self.normalize_case else _VALID_DNA_BYTES_BOTH,
                device=ascii_bytes_cpu.device,
                dtype=ascii_bytes_cpu.dtype,
            )
            invalid_mask = ~torch.isin(ascii_bytes_cpu, valid_bytes)
            if invalid_mask.any():
                need_clone = True

        if need_clone:
            ascii_bytes_cpu = ascii_bytes_cpu.clone()
            if was_pinned:
                ascii_bytes_cpu = ascii_bytes_cpu.pin_memory()

        if self.normalize_case:
            lower_mask = (ascii_bytes_cpu >= ord("a")) & (ascii_bytes_cpu <= ord("z"))
            if lower_mask.any():
                ascii_bytes_cpu[lower_mask] = ascii_bytes_cpu[lower_mask] - 32
        if self.handle_invalid_chars:
            valid_bytes = torch.as_tensor(
                _VALID_DNA_BYTES_UPPER if self.normalize_case else _VALID_DNA_BYTES_BOTH,
                device=ascii_bytes_cpu.device,
                dtype=ascii_bytes_cpu.dtype,
            )
            invalid_mask = ~torch.isin(ascii_bytes_cpu, valid_bytes)
            if invalid_mask.any():
                ascii_bytes_cpu[invalid_mask] = ord("N")

        dev = _as_torch_device(device)

        if self._lut_cuda is None or self._lut_cuda.device != dev:
            self._lut_cuda = torch.as_tensor(self.ascii_lut, dtype=torch.long, device=dev)

        ascii_dev = ascii_bytes_cpu.to(dev, non_blocking=True)
        ids_dev = self._lut_cuda[ascii_dev.long()]

        if self.ascii_start_lut is not None and ids_dev.shape[1] > 0:
            if self._lut_start_cuda is None or self._lut_start_cuda.device != dev:
                self._lut_start_cuda = torch.as_tensor(
                    self.ascii_start_lut, dtype=torch.long, device=dev
                )
            ids_dev[:, 0] = self._lut_start_cuda[ascii_dev[:, 0].long()]

        return ids_dev

    def _map_kmer_bytes_to_ids_cuda(
        self, ascii_bytes_cpu: torch.Tensor, device: object = "cuda:0"
    ) -> torch.Tensor:
        """
        GPU-accelerated K-mer tokenization.
        ascii_bytes_cpu: [B, T] uint8 tensor
        """
        dev = _as_torch_device(device)
        B, T = ascii_bytes_cpu.shape
        k = self.kmer_k
        
        if T % k != 0:
            raise ValueError(f"Sequence length {T} not divisible by k={k}.")
             
        # Ensure LUTs are on device
        if getattr(self, "_base5_lut_cuda", None) is None or self._base5_lut_cuda.device != dev:
            self._base5_lut_cuda = torch.as_tensor(self.base5_lut, dtype=torch.long, device=dev)
            self._kmer_lut_cuda = torch.as_tensor(self.kmer_lut, dtype=torch.long, device=dev)

        # Move bytes to device
        ascii_dev = ascii_bytes_cpu.to(dev, non_blocking=True).long() # [B, T]
        
        # Map to base5: 0..4
        base5_dev = self._base5_lut_cuda[ascii_dev] # [B, T]
        if torch.any(base5_dev < 0):
            if self.handle_invalid_chars:
                # Map invalid to N (4)
                base5_dev[base5_dev < 0] = 4
            else:
                raise ValueError("Invalid characters found for K-mer path.")
        
        # Reshape to [B, T/k, k]
        base5_dev = base5_dev.view(B, T // k, k)
        
        # Pack into integers
        if not hasattr(self, "_kmer_weights_cuda") or self._kmer_weights_cuda.device != dev:
            weights = torch.tensor([5**(k-1-i) for i in range(k)], dtype=torch.long, device=dev)
            self._kmer_weights_cuda = weights
             
        # [B, T/k, k] * [k] -> [B, T/k, k] -> sum(-1) -> [B, T/k]
        # Avoid matmul for Long on CUDA
        packed = torch.sum(base5_dev * self._kmer_weights_cuda, dim=-1)
        
        # Lookup IDs
        ids_dev = self._kmer_lut_cuda[packed]
        if torch.any(ids_dev < 0):
            # Indicates a k-mer (e.g., containing 'N') that does not map to a single token.
            raise ValueError("Sequence contains K-mers not supported by single tokens.")
        
        return ids_dev

    def ids_from_ascii_bytes_cuda(
        self, ascii_bytes_cpu: torch.Tensor, device: object = "cuda:0"
    ) -> torch.Tensor:
        return self._map_ascii_bytes_to_ids_cuda(ascii_bytes_cpu, device)

    # -------------------- Device-side left padding -------------------------
    def _left_pad_device(self, ids_dev: torch.Tensor, T: int, dev: torch.device) -> torch.Tensor:
        """Left-pad on device to self.token_len if needed (right-align original)."""
        if not self.token_len or self.token_len <= T:
            return ids_dev
        B = ids_dev.shape[0]
        out = torch.empty((B, self.token_len), dtype=ids_dev.dtype, device=dev)
        out.fill_(self.id_pad)
        out[:, -T:].copy_(ids_dev)
        return out

    # Convenience: end-to-end from strings with overlap or bytes path
    def embed_from_strings(
        self,
        seqs: List[str],
        emb_batch: int,
        device: object = "cuda",
        *,
        path: str = "auto",  # "ids" (legacy default), "bytes", or "auto"
    ) -> Iterator[torch.Tensor]:
        """
        End-to-end convenience wrapper.
        - path="ids"  : legacy behavior (CPU ids staging, optional overlap)
        - path="bytes": CPU ascii bytes → device LUT map → device left-pad
        - path="auto" : choose "bytes" on CUDA else "ids"
        """
        if not seqs:
            raise ValueError("No sequences provided.")
        seqs = self._normalize_and_clean_seqs(list(seqs))
        if path not in ("ids", "bytes", "auto"):
            raise ValueError("path must be one of {'ids','bytes','auto'}")
        if path == "auto":
            # Prefer bytes path if available (fastest for char-level AND k-mer)
            if (
                _is_cuda_device(device) 
                and torch.cuda.is_available() 
                and self.use_ids_path 
                and (self.ascii_lut is not None or self.kmer_k is not None)
            ):
                path = "bytes"
            else:
                path = "ids"

        if path == "bytes" and self.use_ids_path:
            dev = _as_torch_device(device)
            ascii_cpu = self.encode_batch_to_ascii_bytes(seqs)  # pinned uint8 [B,T]
            B, T = int(ascii_cpu.shape[0]), int(ascii_cpu.shape[1])
            if B == 0:
                return
            try:
                if self.kmer_k is not None:
                    ids_dev = self._map_kmer_bytes_to_ids_cuda(ascii_cpu, dev)
                else:
                    ids_dev = self._map_ascii_bytes_to_ids_cuda(ascii_cpu, dev)
                
                # Pass actual token length to left_pad
                ids_dev = self._left_pad_device(ids_dev, int(ids_dev.shape[1]), dev)
            except Exception as e:
                self.log.warning(
                    "KevTok bytes path unavailable (%s); falling back to ids path.", e
                )
            else:
                # Micro-batch purely on device
                micro_bs = self._ids_micro_bs_for_T(int(ids_dev.shape[1]), emb_batch)
                start = 0
                kwargs_template = {}
                try:
                    sig = inspect.signature(self.embedder.embed_tokens)
                    if "rc_invariant" in sig.parameters:
                        kwargs_template["rc_invariant"] = False
                except Exception:
                    pass
                while start < B:
                    end = min(B, start + micro_bs)
                    try:
                        with torch.amp.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                            enabled=_is_cuda_device(dev),
                        ):
                            out = self.embedder.embed_tokens(
                                ids_dev[start:end], **kwargs_template
                            )
                        if out.device != dev:
                            out = out.to(device=dev, non_blocking=True)
                        if self.force_fp32_outputs and out.dtype != torch.float32:
                            out = out.float()
                        yield out
                    except RuntimeError as e:
                        self.log.warning(
                            "KevTok bytes path failed at batch starting %d (%s); "
                            "falling back to ids path for remaining sequences.",
                            start,
                            e,
                        )
                        remaining = seqs[start:]
                        if remaining:
                            yield from self.embed_from_strings(
                                remaining, emb_batch, device=device, path="ids"
                            )
                        return
                    start = end
                return

        # Legacy "ids" path (unchanged defaults)
        if not self.use_ids_path:
            # Fallback: use embedder directly (tokenizer + embed)
            # We'll batch the strings to avoid OOM on tokenization of huge lists
            B = len(seqs)
            start = 0
            # Use a reasonable batch size for tokenization + embedding
            # emb_batch is for embedding, but tokenization also needs batching if B is huge.
            # Let's use emb_batch.
            
            kwargs_template = {}
            try:
                sig = inspect.signature(self.embedder.embed_tokens)
                if "rc_invariant" in sig.parameters:
                    kwargs_template["rc_invariant"] = False
            except Exception:
                pass

            dev = _as_torch_device(device)
            tok = getattr(self.embedder, "tokenizer", None)
            if tok is None:
                raise RuntimeError("IDs path disabled and no tokenizer found on embedder.")

            while start < B:
                end = min(B, start + emb_batch)
                sub_seqs = seqs[start:end]
                
                # Tokenize
                try:
                    enc = tok(
                        sub_seqs,
                        add_special_tokens=False,
                        padding="max_length" if self.token_len else True,
                        truncation=False, # DNATok usually doesn't truncate?
                        max_length=self.token_len if self.token_len else None,
                        return_tensors="pt"
                    )
                    ids = enc["input_ids"]
                    if ids.device != dev:
                        ids = ids.to(device=dev, non_blocking=True)
                    
                    # Embed
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=_is_cuda_device(dev)):
                        out = self.embedder.embed_tokens(ids, **kwargs_template)
                    
                    if out.device != dev:
                        out = out.to(device=dev, non_blocking=True)
                    if self.force_fp32_outputs and out.dtype != torch.float32:
                        out = out.float()
                    yield out
                except Exception as e:
                    self.log.error(f"Fallback tokenization failed: {e}")
                    raise e
                
                start = end
            return

        ids_cpu_staging = self.encode_batch_to_ids_staging(seqs)
        if self.overlap_h2d_compute:
            yield from self.iter_embed_tokens_pipelined(
                ids_cpu_staging, emb_batch, device=device
            )
        else:
            yield from self.iter_embed_tokens_in_slices(
                ids_cpu_staging.to(torch.long), emb_batch, device=device
            )

    def free_buffers(self) -> None:
        """Explicitly release pinned memory buffers."""
        self._staging_ids_cpu = None
        self._staging_bytes_cpu = None
        self._dev_ping_i = None
        self._dev_pong_i = None
        self._dev_ping_l = None
        self._dev_pong_l = None
        self._lut_cuda = None
        self._lut_start_cuda = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
