from __future__ import annotations

import inspect
import logging
import os
from collections import deque
from typing import Dict, Iterator, List, Optional, Tuple, Protocol, Any, Union

import numpy as np
import torch

try:
    import dnatok_compat  # noqa: F401
except Exception as _dnatok_compat_exc:
    # Best-effort compatibility patch; safe to skip when module unavailable.
    # Surface the cause at WARNING so downstream model-load failures are not cryptic.
    logging.getLogger(__name__).warning(
        "dnatok_compat unavailable (%s); NTv3/custom-tokenizer paths may fall back.",
        _dnatok_compat_exc,
    )

_VALID_DNA_BYTES_UPPER = np.frombuffer(b"ACGTN", dtype=np.uint8).copy()
_VALID_DNA_BYTES_BOTH = np.frombuffer(b"ACGTNacgtn", dtype=np.uint8).copy()
_MAX_TOKEN_LEN = 1_000_000


# NVTX annotation helper. When CUDA is available, decorates the hot paths
# with named ranges so `nsys profile` produces a readable timeline that
# reviewers (or downstream optimisation work) can use without hunting through
# PyTorch's autograd ranges.
try:
    from torch.cuda import nvtx as _nvtx
    def _nvtx_range(name):  # noqa: D401
        class _RangeCM:
            __slots__ = ("name",)
            def __init__(self, n): self.name = n
            def __enter__(self): _nvtx.range_push(self.name)
            def __exit__(self, *exc): _nvtx.range_pop()
        return _RangeCM(name)
except Exception:  # pragma: no cover
    class _NoOpRange:
        def __enter__(self): pass
        def __exit__(self, *exc): pass
    def _nvtx_range(name):  # noqa: D401
        return _NoOpRange()


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
            normalize_case: bool = False,  # force uppercase inputs
            handle_invalid_chars: bool = False,  # map invalid chars to N instead of error
            require_valid_chars: bool = False,  # fall back when non-ACGTN chars appear
            allow_incomplete_kmer_lut: bool = False,  # map missing k-mers to fallback ID
            padding_side: str = "left",  # align padding with tokenizer when needed
            # ------------------------------------------------------------
            # New optimizations (each opt-in; default off preserves prior
            # measured numbers and avoids surprising callers).
            # ------------------------------------------------------------
            pack_2bit_h2d: bool = False,        # 2-bit nucleotide packing for H2D (4x bandwidth)
            use_uint8_ids: bool = False,        # uint8 IDs on bus when vocab <= 256
            use_cuda_graph: bool = False,       # CUDA-graph capture+replay for static shapes
            use_fused_kernel: bool = False,     # Triton fused tokenize+gather kernel
    ) -> None:
        self.embedder = embedder
        self.log = logger or logging.getLogger("DNATok")
        if not self.log.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.log.addHandler(ch)
            # Default to WARNING so the plug-and-play path stays quiet.
            # Users who want the discovery-trace output can opt in with
            # logging.getLogger("DNATok").setLevel(logging.INFO) or the
            # DNATOK_LOG_LEVEL environment variable.
            level_env = os.environ.get("DNATOK_LOG_LEVEL", "WARNING").upper()
            self.log.setLevel(getattr(logging, level_env, logging.WARNING))

        # discovered at runtime by discover()
        self.use_ids_path: bool = False
        self.ascii_lut: Optional[np.ndarray] = None  # shape [256] int64
        self.ascii_start_lut: Optional[np.ndarray] = None  # shape [256] int64, for first token if different
        self.id_pad: int = 0
        self.id_N: int = 0
        self.token_len: Optional[int] = None
        # Total vocabulary size derived at discover() time. Used to pick the
        # smallest safe integer dtype for the public encode_batch_with_padding
        # path; never reaching for an actual upper bound here causes silent
        # uint8 truncation on tokenizers with vocab > 256.
        self.vocab_size: int = 0
        # Opt-in BPE backend (set by discover() when the tokenizer is one
        # of the supported genomic BPE families: DNABERT-2, GENA-LM,
        # METAGENE-1). When non-None, encode_batch_to_ids routes BPE
        # tokenisation through this GPU kernel instead of HF native.
        self.bpe_backend: Optional[Any] = None

        # Cached LMM BPE (CPU, bit-identical to HF). Built by discover()
        # if the tokenizer is a non-SP BPE and the env var
        # DNATOK_LMM_CACHE_PATH points to a usable cache, OR if
        # DNATOK_LMM_ENABLE=1 to build lazily. Routed to BEFORE
        # bpe_backend in encode_batch_to_ids.
        self.lmm_bpe: Optional[Any] = None

        # runtime safety cap
        self.ids_max_tokens_per_call: int = int(ids_max_tokens_per_call)

        # performance knobs
        self.prefer_int32_h2d: bool = bool(prefer_int32_h2d)
        self.overlap_h2d_compute: bool = bool(overlap_h2d_compute)
        self.force_fp32_outputs: bool = bool(force_fp32_outputs)
        self.strict_lut_check: bool = bool(strict_lut_check)
        self.normalize_case: bool = bool(normalize_case)
        self.handle_invalid_chars: bool = bool(handle_invalid_chars)
        self.require_valid_chars: bool = bool(require_valid_chars)
        env_allow = os.environ.get("DNATOK_ALLOW_INCOMPLETE_KMER_LUT")
        if env_allow is not None:
            self.allow_incomplete_kmer_lut = env_allow.strip().lower() in ("1", "true", "yes", "y")
        else:
            self.allow_incomplete_kmer_lut = bool(allow_incomplete_kmer_lut)
        if padding_side not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")
        self.padding_side: str = padding_side

        # persistent staging (CPU) and ping–pong (CUDA)
        self._staging_ids_cpu: Optional[torch.Tensor] = None  # int32 or int64
        self._staging_bytes_cpu: Optional[torch.Tensor] = None  # uint8
        self._dev_ping_i: Optional[torch.Tensor] = None  # int32 or int64
        self._dev_pong_i: Optional[torch.Tensor] = None
        self._dev_ping_l: Optional[torch.Tensor] = None  # int64
        self._dev_pong_l: Optional[torch.Tensor] = None
        self._lut_cuda: Optional[torch.Tensor] = None  # cached 256 LUT on device
        self._lut_start_cuda: Optional[torch.Tensor] = None  # cached start-LUT (if needed)
        self._base5_lut_cuda: Optional[torch.Tensor] = None
        self._kmer_lut_cuda: Optional[torch.Tensor] = None
        self._kmer_offsets_cuda: Optional[torch.Tensor] = None
        self._kmer_lengths_cuda: Optional[torch.Tensor] = None
        self._kmer_flat_cuda: Optional[torch.Tensor] = None
        self._kmer_weights_cuda: Optional[torch.Tensor] = None

        # K-mer fast path
        self.kmer_k: Optional[int] = None
        self.kmer_lut: Optional[np.ndarray] = None  # shape [5**k] int64
        self.base5_lut: Optional[np.ndarray] = None  # shape [256] int8 (0-4, -1 for invalid)
        self.kmer_lut_incomplete: bool = False
        self.kmer_lut_offsets: Optional[np.ndarray] = None  # shape [5**k] int32
        self.kmer_lut_lengths: Optional[np.ndarray] = None  # shape [5**k] int16
        self.kmer_lut_flat: Optional[np.ndarray] = None  # flat list of token ids
        self.kmer_single_char_lut: Optional[np.ndarray] = None  # [256] int64; per-byte single-char ids for the partial-k-mer tail

        # invalid/unknown token mapping
        self.id_unk: Optional[int] = None
        self.invalid_char_id: Optional[int] = None
        # K-mer invalid handling: replace_with_n | map_to_unk | error
        self.kmer_invalid_policy: str = "replace_with_n"
        self._kmer_partial_warned: bool = False

        # ----- New optimization flags + caches -----
        self.pack_2bit_h2d: bool = bool(pack_2bit_h2d)
        self.use_uint8_ids: bool = bool(use_uint8_ids)
        self.use_cuda_graph: bool = bool(use_cuda_graph)
        self.use_fused_kernel: bool = bool(use_fused_kernel)

        # 2-bit packing scratch (CPU pinned + device).
        self._packed_2bit_cpu: Optional[torch.Tensor] = None  # uint8 [B, ceil(T/4)]
        self._packed_2bit_dev: Optional[torch.Tensor] = None  # uint8 [B, ceil(T/4)] device
        # uint8 LUT for narrow vocabularies (filled lazily by discover()).
        self._lut_u8_cuda: Optional[torch.Tensor] = None  # uint8 [256]
        # CUDA-graph machinery
        self._graph: Optional[Any] = None
        self._graph_static_in: Optional[torch.Tensor] = None
        self._graph_static_out: Optional[torch.Tensor] = None
        self._graph_shape: Optional[Tuple[int, int]] = None
        self._graph_dev: Optional[torch.device] = None
        # Triton fused kernel availability is detected lazily.
        self._fused_kernel_available: Optional[bool] = None

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
        # Prefer canonical HF attributes on the embedder *and* the inner
        # tokenizer. Falling back to _safe_token_to_id with synthetic names
        # ("<pad>", "PAD", ...) is dangerous because HF's
        # convert_tokens_to_ids returns the UNK id for unknown tokens, which
        # would silently misidentify UNK as the pad id (this is what made
        # GENA-LM's pad_id resolve to 0 instead of 3).
        src = tok_or_embedder
        for name in ("pad_id", "pad_token_id"):
            v = getattr(src, name, None)
            if isinstance(v, int) and v >= 0:
                return v
        tok = getattr(src, "tokenizer", None)
        if tok is not None:
            tok = self._maybe_unwrap_tokenizer(tok)
            # Canonical attribute first.
            for name in ("pad_id", "pad_token_id"):
                v = getattr(tok, name, None)
                if isinstance(v, int) and v >= 0:
                    return v
            # Discover the UNK id once so we can reject UNK-fallback hits.
            unk_candidates: set[int] = set()
            for unk_name in ("unk_id", "unk_token_id", "unknown_token_id"):
                v = getattr(tok, unk_name, None)
                if isinstance(v, int) and v >= 0:
                    unk_candidates.add(v)
            for token_str in ("[PAD]", "<pad>", "PAD", "pad"):
                tid = self._safe_token_to_id(tok, token_str)
                if isinstance(tid, int) and tid not in unk_candidates:
                    return tid
        return None

    def _discover_unk_id(self, tok_or_embedder: object) -> Optional[int]:
        src = tok_or_embedder
        for name in ("unk_id", "unk_token_id", "unknown_token_id"):
            v = getattr(src, name, None)
            if isinstance(v, int) and v >= 0:
                return v
        tok = getattr(src, "tokenizer", None)
        if tok is not None:
            tok = self._maybe_unwrap_tokenizer(tok)
            for name in ("unk_id", "unk_token_id", "unknown_token_id"):
                v = getattr(tok, name, None)
                if isinstance(v, int) and v >= 0:
                    return v
            for token_str in ("[UNK]", "<unk>", "UNK", "unk"):
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
        if self.normalize_case:
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
        for ch, val in zip("ACGTN", [0, 1, 2, 3, 4]):
            base5[ord(ch)] = val
        if self.normalize_case:
            for ch, val in zip("acgtn", [0, 1, 2, 3, 4]):
                base5[ord(ch)] = val

        # 2. K-mer LUT
        # Iterate all 5^k combinations
        size = 5 ** k
        kmer_lut = np.full(size, -1, dtype=np.int64)
        kmer_offsets = np.full(size, -1, dtype=np.int32)
        kmer_lengths = np.zeros(size, dtype=np.int16)
        kmer_flat: List[int] = []

        import itertools
        bases = "ACGTN"

        all_kmers = ["".join(p) for p in itertools.product(bases, repeat=k)]

        # We process in batches to avoid OOM or huge lists
        batch_size = 4096
        for i in range(0, len(all_kmers), batch_size):
            batch = all_kmers[i: i + batch_size]

            try:
                enc_list = tok(
                    batch,
                    add_special_tokens=False,
                    padding=False,
                    return_tensors=None  # Return lists
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
                row_ids = self._maybe_int_seq(row)
                if not row_ids:
                    continue
                if len(row_ids) == 1:
                    kmer_lut[i + idx] = int(row_ids[0])
                else:
                    off = len(kmer_flat)
                    kmer_flat.extend(int(x) for x in row_ids)
                    kmer_offsets[i + idx] = int(off)
                    kmer_lengths[i + idx] = int(len(row_ids))

        # Validate LUT
        invalid_count = (kmer_lut == -1).sum()
        self.kmer_lut_incomplete = bool(invalid_count > 0)
        self.kmer_lut_offsets = kmer_offsets
        self.kmer_lut_lengths = kmer_lengths
        self.kmer_lut_flat = np.asarray(kmer_flat, dtype=np.int64)
        if invalid_count > 0:
            multi_count = int((kmer_lengths > 0).sum())
            self.log.warning(
                f"K-mer LUT incomplete: {invalid_count}/{len(kmer_lut)} k-mers are not single tokens (e.g. containing 'N'). "
                f"{multi_count} entries have multi-token expansions; exact fallback will use these when needed. "
                "Set DNATOK_ALLOW_INCOMPLETE_KMER_LUT=1 to map missing k-mers to a fallback ID (approx)."
            )

        return base5, kmer_lut

    def _resolve_vocab_size(self, tok: object) -> int:
        """Best-effort upper bound on the tokenizer's vocab size.

        Used to pick a safe integer dtype for public-API outputs. Falling
        back to 0 here causes silent uint8 truncation downstream, so this
        method is conservative: it returns the max of every signal it can
        find, plus the largest ID present in the discovered LUTs.
        """
        candidates: List[int] = []
        for attr in ("vocab_size", "n_vocab"):
            v = getattr(tok, attr, None)
            if isinstance(v, int) and v > 0:
                candidates.append(int(v))
        try:
            length = len(tok)  # type: ignore[arg-type]
            if isinstance(length, int) and length > 0:
                candidates.append(int(length))
        except Exception:
            pass
        try:
            vocab = self._safe_get_vocab_dict(tok)
            if vocab:
                candidates.append(int(len(vocab)))
                # Some sparse tokenizers (e.g. with reserved/added ids beyond
                # vocab_size) report a smaller len than the largest id.
                max_id = max((int(v) for v in vocab.values() if isinstance(v, (int, np.integer))),
                             default=0)
                if max_id + 1 > 0:
                    candidates.append(int(max_id + 1))
        except Exception:
            pass
        if self.ascii_lut is not None:
            candidates.append(int(self.ascii_lut.max(initial=0)) + 1)
        if self.kmer_lut is not None:
            candidates.append(int(self.kmer_lut.max(initial=0)) + 1)
        if isinstance(self.id_pad, int):
            candidates.append(int(self.id_pad) + 1)
        return max(candidates) if candidates else 0

    # --------------------------- Discovery ---------------------------------
    def discover(self) -> None:
        # Always clear derived state before probing.

        self.use_ids_path = False
        self.vocab_size = 0
        self.bpe_backend = None

        self.ascii_lut = None
        self.ascii_start_lut = None
        self._lut_cuda = None
        self._lut_start_cuda = None
        self._base5_lut_cuda = None
        self._kmer_lut_cuda = None
        self._kmer_offsets_cuda = None
        self._kmer_lengths_cuda = None
        self._kmer_flat_cuda = None
        self._kmer_weights_cuda = None
        self.kmer_k = None
        self.kmer_lut = None
        self.base5_lut = None
        self.kmer_lut_incomplete = False
        self.kmer_lut_offsets = None
        self.kmer_lut_lengths = None
        self.kmer_lut_flat = None
        embed_tokens = getattr(self.embedder, "embed_tokens", None)
        if not callable(embed_tokens):
            return
        # Reset k-mer helpers on each discovery
        pad_id = self._discover_pad_id(self.embedder)
        if pad_id is None:
            pad_id = 0
        unk_id = self._discover_unk_id(self.embedder)
        self.id_unk = int(unk_id) if isinstance(unk_id, int) else None
        tok = getattr(self.embedder, "tokenizer", None)
        # Populate vocab_size unconditionally from the tokenizer so callers
        # of encode_batch_with_padding pick a safe dtype even when the fast
        # IDs path cannot be enabled (BPE tokenizers etc. fall through to
        # the tokenizer-backed fallback but still need correct dtype).
        if tok is not None:
            try:
                self.vocab_size = self._resolve_vocab_size(self._maybe_unwrap_tokenizer(tok))
            except Exception:
                self.vocab_size = 0
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
            self.invalid_char_id = self.id_unk if self.id_unk is not None else int(n_id)
            self.vocab_size = int(self.ascii_lut.max(initial=0)) + 1
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
                self.id_N = 0  # Placeholder
                self.invalid_char_id = self.id_unk if self.id_unk is not None else int(self.id_N)
                self.vocab_size = self._resolve_vocab_size(tok)
                self.use_ids_path = True

                # Build a single-character byte→id LUT for the partial-k-mer
                # tail. HF tokenises trailing T%k bases as single-char
                # tokens (e.g. NTv2: "A" → 4102, "C" → 4104). Probing
                # them here lets the fast path accept inputs of any
                # length, not just multiples of k.
                try:
                    self.kmer_single_char_lut = np.full(256, int(pad_id), dtype=np.int64)
                    for ch in "ACGTN":
                        sid = self._encode_char_to_single_id(tok, ch)
                        if sid is None:
                            self.kmer_single_char_lut = None
                            break
                        self.kmer_single_char_lut[ord(ch)] = int(sid)
                        # Lowercase variants — some tokenisers map them to
                        # the uppercase id, others have separate entries.
                        sid_lc = self._encode_char_to_single_id(tok, ch.lower())
                        if sid_lc is not None:
                            self.kmer_single_char_lut[ord(ch.lower())] = int(sid_lc)
                except Exception:
                    self.kmer_single_char_lut = None

                # Optional fixed token length hint
                for name in ("model_max_length", "max_position_embeddings", "max_seq_len"):
                    v = getattr(self.embedder, name, None)
                    if isinstance(v, int) and 0 < v <= _MAX_TOKEN_LEN:
                        self.token_len = v
                        break

                self.log.info(f"K-mer fast path enabled (k={k}).")
                try:
                    self._sanity_compare_kmer_expansion(tok)
                except Exception as e:
                    self.log.warning("K-mer expansion sanity check failed: %s", e)
                return
            except Exception as e:
                self.log.warning(f"Failed to build K-mer LUT: {e}. Falling back to char path.")
                self.kmer_k = None
                self.base5_lut = None
                self.kmer_lut = None
                self.kmer_lut_offsets = None
                self.kmer_lut_lengths = None
                self.kmer_lut_flat = None

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
        self.invalid_char_id = self.id_unk if self.id_unk is not None else int(n_cont_id)
        self.vocab_size = self._resolve_vocab_size(tok)
        # Optional fixed token length hint
        for name in ("model_max_length", "max_position_embeddings", "max_seq_len"):
            v = getattr(self.embedder, name, None)
            if isinstance(v, int) and 0 < v <= _MAX_TOKEN_LEN:
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
                    # INFO not WARNING: disabling the ASCII LUT path is the
                    # expected behaviour for multi-byte tokenizers (BPE,
                    # k-mer). The downstream encode_batch_to_ids routes
                    # through the appropriate fast path.
                    self.log.info(
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

        # ---- CachedLMM BPE opt-in (CPU, bit-identical, often faster) --
        # If the LUT/k-mer fast paths are unavailable and the tokenizer
        # is a non-SP BPE (DNABERT-2 / GENA-LM, etc.), try the cached
        # safe-margin LMM encoder. SP tokenizers (METAGENE-1) are
        # rejected by is_supported() and fall through to bpe_backend / HF.
        if not self.use_ids_path and tok is not None:
            try:
                from dnatok_lmm_bpe import CachedLMMBPE, is_supported as _lmm_is_supported
                if _lmm_is_supported(tok):
                    cache_path = os.environ.get("DNATOK_LMM_CACHE_PATH")
                    self.lmm_bpe = CachedLMMBPE(
                        tok, cache_path=cache_path, log=self.log,
                    )
                    self.log.info(
                        "CachedLMM BPE enabled (K=%d, safety=%d, "
                        "cache_size=%d)",
                        self.lmm_bpe.K, self.lmm_bpe.safety,
                        len(self.lmm_bpe.cache))
            except Exception as e:
                self.log.info(
                    "CachedLMM BPE not built (%s); will try bpe_backend / "
                    "HF fallback.", e)
                self.lmm_bpe = None

        # ---- BPE backend opt-in (GPU kernel) -------------------------
        # If the LUT/k-mer fast paths are unavailable and the tokenizer is
        # a supported genomic BPE (DNABERT-2 / GENA-LM / METAGENE-1),
        # build the GPU BPE backend. Failure here is non-fatal — we
        # leave bpe_backend=None and the standard HF-tokenizer fallback
        # in encode_batch_to_ids handles the model just like before.
        if not self.use_ids_path and tok is not None:
            try:
                from gputok_bpe_backend import GPUTokBPEBackend
                if GPUTokBPEBackend.is_supported(self.embedder.tokenizer):
                    self.bpe_backend = GPUTokBPEBackend(
                        self.embedder.tokenizer, engine="dnatok"
                    )
                    self.log.info("DNATok BPE backend enabled (engine=dnatok).")
            except Exception as e:
                self.log.info(
                    "DNATok BPE backend not built (%s); falling back to HF tokenizer.", e
                )
                self.bpe_backend = None

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
            # INFO not WARNING: a mismatch here is the *expected* path for
            # multi-byte tokenizers (BPE, k-mer). It tells us the simple
            # single-byte LUT can't be used; we then route to the correct
            # fast path (BPE / k-mer LUT). User-facing default is silent.
            self.log.info(
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
        # Probe each sequence individually so the sanity check still works on
        # BPE tokenizers that emit variable-length output (HF's
        # return_tensors='pt' with padding='max_length' would fail to pack
        # variable-length BPE results into a single tensor and previously
        # caused the sanity check to silently SKIP — which let METAGENE_1
        # falsely enable the LUT fast path and produce wrong tokens).
        try:
            per_seq_ids: List[List[int]] = []
            for seq in probes:
                enc = tok(
                    [seq],
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_tensors=None,
                )
                ids = enc["input_ids"][0] if isinstance(enc, dict) else enc.input_ids[0]
                per_seq_ids.append([int(x) for x in ids])
        except Exception as e:
            # Tokenizer call itself failed — fail closed and let the caller
            # disable the IDs path.
            raise ValueError(
                f"Sanity-check tokenizer call failed; cannot validate LUT: {e}"
            ) from e

        # Every probe must produce exactly T tokens for the ASCII-LUT path to
        # be valid (1 byte ↔ 1 token). BPE tokenizers fail this gate.
        bad_lens = [(i, len(ids)) for i, ids in enumerate(per_seq_ids) if len(ids) != T]
        if bad_lens:
            raise ValueError(
                f"Tokenizer is not 1-byte-per-token (lengths: {bad_lens}); "
                f"ASCII LUT path is not valid for this tokenizer."
            )
        tok_ids = torch.tensor(per_seq_ids, dtype=torch.int64)

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
                # Diagnostic detail at DEBUG level only; the raised exception
                # carries the count for higher-level handlers.
                self.log.debug("LUT mismatch: %d tokens", diff_count)
                self.log.debug("tok shape=%s lut shape=%s", tuple(tok_ids.shape), tuple(lut_ids.shape))
                mismatch_indices = diff_mask.nonzero(as_tuple=False)
                for idx in mismatch_indices[:10]:
                    b, t = idx.tolist()
                    self.log.debug(
                        "  mismatch (%d,%d): tok=%s lut=%s char=%r",
                        b, t, tok_ids[b, t].item(), lut_ids[b, t],
                        probes[b][t] if b < len(probes) and t < len(probes[b]) else "?",
                    )

            raise ValueError(f"Tokenizer vs LUT mismatch on {diff_count} tokens.")

    def _sanity_compare_kmer_expansion(self, tok: object) -> None:
        """
        Validate that per-k-mer expansions reproduce tokenizer outputs.
        Disables multi-token expansion LUTs on mismatch.
        """
        if (
            self.kmer_k is None
            or self.kmer_lut_offsets is None
            or self.kmer_lut_lengths is None
            or self.kmer_lut_flat is None
        ):
            return
        k = int(self.kmer_k)
        if k <= 0:
            return

        rng = np.random.default_rng(0)
        alphabet = np.array(list("ACGTN"))
        num_samples = 32
        seq_len = k * 4
        seqs = ["".join(rng.choice(alphabet, size=seq_len)) for _ in range(num_samples)]

        old_token_len = self.token_len
        self.token_len = None
        try:
            expanded = self._encode_kmer_fallback_from_lut(seqs)
            if expanded is None:
                return
            ids_list, success = expanded
            if not success.all():
                self.log.warning(
                    "K-mer LUT expansion incomplete during sanity check; disabling multi-token expansions."
                )
                self.kmer_lut_offsets = None
                self.kmer_lut_lengths = None
                self.kmer_lut_flat = None
                self._kmer_offsets_cuda = None
                self._kmer_lengths_cuda = None
                self._kmer_flat_cuda = None
                return
            ids_exp = self._pad_id_list(ids_list, dtype=torch.long)
            ids_tok = self._tokenize_batch_cpu(seqs, dtype=torch.long, pin=False)
            if ids_exp.shape != ids_tok.shape or not torch.equal(ids_exp, ids_tok):
                self.log.warning(
                    "K-mer LUT expansion mismatch vs tokenizer; disabling multi-token expansions."
                )
                self.kmer_lut_offsets = None
                self.kmer_lut_lengths = None
                self.kmer_lut_flat = None
                self._kmer_offsets_cuda = None
                self._kmer_lengths_cuda = None
                self._kmer_flat_cuda = None
        finally:
            self.token_len = old_token_len

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
            raise ValueError("Sequence length not divisible by k for K-mer path.")

        # 1. Map to base 5
        # We reuse the ascii_lut logic but with base5_lut

        # Flatten and map
        if self.normalize_case:
            buf = "".join(seqs).upper().encode("ascii", errors="replace")
        else:
            buf = "".join(seqs).encode("ascii", errors="replace")
        arr = np.frombuffer(buf, dtype=np.uint8)

        # Check size
        if arr.size != len(seqs) * T:
            # Handle unicode/weird chars by re-encoding individually
            out = np.empty((len(seqs), T), dtype=np.uint8)
            for i, s in enumerate(seqs):
                out[i, :] = np.frombuffer(s.encode("ascii", errors="replace"), dtype=np.uint8)[:T]
            arr = out.reshape(-1)

        B = len(seqs)

        # Map to 0..4
        # base5_lut is int8.
        mapped = self.base5_lut[arr]  # [Total_Chars]

        # Check for invalid chars (-1)
        invalid_mask = mapped < 0
        invalid_kmer = None
        if np.any(invalid_mask):
            if not self.handle_invalid_chars:
                raise ValueError("Invalid characters found for K-mer path.")
            if self.kmer_invalid_policy == "replace_with_n":
                mapped = mapped.copy()
                mapped[invalid_mask] = 4
            elif self.kmer_invalid_policy == "map_to_unk":
                if self.invalid_char_id is None:
                    raise ValueError("Invalid characters found but invalid_char_id is unset.")
                mapped = mapped.copy()
                mapped[invalid_mask] = 0
                invalid_kmer = invalid_mask.reshape(B, T // k, k).any(axis=2)
            else:
                raise ValueError("Invalid characters found for K-mer path.")

        # Reshape to (B, T/k, k)
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
        if invalid_kmer is not None:
            ids = np.array(ids, copy=True)
            ids[invalid_kmer] = int(self.invalid_char_id)

        # Check for -1 (invalid k-mers, e.g. "AN" if tokenizer splits it)
        if np.any(ids < 0):
            if self.allow_incomplete_kmer_lut:
                fill_id = self._resolve_kmer_missing_id()
                ids = np.array(ids, copy=True)
                ids[ids < 0] = int(fill_id)
            else:
                raise ValueError(
                    "Sequence contains K-mers not supported by single tokens (e.g. split tokens)."
                )

        return ids

    def _encode_batch_kmer_numpy_partial(self, seqs: List[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        K-mer path that reports which sequences must fall back to native tokenization.
        Returns (ids, fallback_mask).
        """
        if self.kmer_k is None or self.kmer_lut is None or self.base5_lut is None:
            raise RuntimeError("K-mer path not initialized.")

        k = self.kmer_k
        T = len(seqs[0])
        for s in seqs:
            if len(s) != T:
                raise ValueError("All sequences in a batch must have equal length.")
        # If T is not a multiple of k, we handle the trailing T%k bases as
        # single-character tokens (mirroring HF's behaviour). This needs
        # the single-char LUT we built during discover(); without it we
        # bail to the slow path. We also bail when T < k (whole input is
        # tail) because HF can compress consecutive-UNK runs into a
        # single token in that regime — our per-char path can't.
        tail_n = T % k
        if tail_n != 0 and self.kmer_single_char_lut is None:
            raise ValueError(
                "Sequence length not divisible by k and no single-char "
                "LUT available for the partial tail."
            )
        if T < k:
            raise ValueError(
                "Sequence length below k; HF may collapse to a single "
                "UNK token. Slow-path handles this."
            )
        T_full = T - tail_n
        if self.normalize_case:
            buf = ("".join(seqs)).upper().encode("ascii", errors="replace")
        else:
            buf = ("".join(seqs)).encode("ascii", errors="replace")
        arr = np.frombuffer(buf, dtype=np.uint8)

        if arr.size != len(seqs) * T:
            out = np.empty((len(seqs), T), dtype=np.uint8)
            for i, s in enumerate(seqs):
                out[i, :] = np.frombuffer(s.encode("ascii", errors="replace"), dtype=np.uint8)[:T]
            arr = out.reshape(-1)

        B = len(seqs)
        # Slice off the tail before the 5-base mapping; the tail is encoded
        # via the single-char LUT below.
        if tail_n:
            arr_2d = arr.reshape(B, T)
            arr_full = arr_2d[:, :T_full].reshape(-1)
            arr_tail = arr_2d[:, T_full:]
            arr = arr_full
        mapped = self.base5_lut[arr]

        invalid_mask = mapped < 0
        invalid_seq_mask = None
        invalid_seq_fallback = False
        invalid_kmer = None
        if np.any(invalid_mask):
            invalid_seq_mask = invalid_mask.reshape(B, T_full).any(axis=1)
            if self.handle_invalid_chars:
                if self.kmer_invalid_policy == "replace_with_n":
                    mapped = mapped.copy()
                    mapped[invalid_mask] = 4
                elif self.kmer_invalid_policy == "map_to_unk":
                    mapped = mapped.copy()
                    mapped[invalid_mask] = 0
                    invalid_kmer = invalid_mask.reshape(B, T_full // k, k).any(axis=2)
                    if self.invalid_char_id is None:
                        invalid_seq_fallback = True
                else:
                    invalid_seq_fallback = True
                    mapped = mapped.copy()
                    mapped[invalid_mask] = 4
            else:
                invalid_seq_fallback = True
                mapped = mapped.copy()
                mapped[invalid_mask] = 4

        mapped = mapped.reshape(B, T_full // k, k)
        mapped = mapped.astype(np.int64)
        powers = np.power(5, np.arange(k - 1, -1, -1), dtype=np.int64)
        packed = np.dot(mapped, powers)

        ids = self.kmer_lut[packed]
        if invalid_kmer is not None and self.invalid_char_id is not None:
            ids = np.array(ids, copy=True)
            ids[invalid_kmer] = int(self.invalid_char_id)

        # Append the trailing partial as single-char tokens (when T%k != 0).
        if tail_n:
            tail_ids = self.kmer_single_char_lut[arr_tail]  # [B, tail_n] int64
            ids = np.concatenate([ids, tail_ids], axis=1)

        missing = ids < 0
        missing_rows = np.any(missing, axis=1) if missing.any() else np.zeros(B, dtype=bool)
        if missing.any() and self.allow_incomplete_kmer_lut:
            fill_id = self._resolve_kmer_missing_id()
            ids = np.array(ids, copy=True)
            ids[missing] = int(fill_id)
            missing_rows = np.zeros(B, dtype=bool)

        fallback_mask = missing_rows
        if invalid_seq_mask is not None and invalid_seq_fallback:
            fallback_mask = fallback_mask | invalid_seq_mask

        return ids, fallback_mask

    def _encode_batch_kmer_variable_length(
        self, seqs: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """K-mer fast path for batches whose sequences differ in length.

        Groups sequences by exact length, processes each homogeneous
        group through ``_encode_batch_kmer_numpy_partial`` (which
        requires equal length within a call), then assembles the results
        back into the original-order, pad-right tensor.

        For realistic workloads (Illumina reads at ~150 ± 30 bp, etc.)
        there are far fewer unique lengths than sequences, so each
        length-group still benefits from numpy vectorisation. For very
        short sequences (T < k) the existing equal-length path bails
        with ValueError — we catch that and mark those specific
        sequences for HF fallback via the returned ``fallback_mask``.

        Returns: (ids_np[B, max_token_len] int64,
                  fallback_mask[B] bool — True for sequences that
                                          need HF fallback).
        """
        from collections import defaultdict

        by_len: Dict[int, List[int]] = defaultdict(list)
        for i, s in enumerate(seqs):
            by_len[len(s)].append(i)

        per_seq_ids: List[Optional[List[int]]] = [None] * len(seqs)
        fallback_mask = np.zeros(len(seqs), dtype=bool)

        for L, indices in by_len.items():
            sub_seqs = [seqs[i] for i in indices]
            try:
                sub_ids, sub_fb = self._encode_batch_kmer_numpy_partial(sub_seqs)
            except ValueError:
                # The equal-length path bails for T < k (HF compresses
                # all-UNK runs to one token there); route those
                # sequences through HF.
                for src_i in indices:
                    fallback_mask[src_i] = True
                continue
            for local_i, src_i in enumerate(indices):
                per_seq_ids[src_i] = sub_ids[local_i].tolist()
                if sub_fb[local_i]:
                    fallback_mask[src_i] = True

        max_len = max((len(x) for x in per_seq_ids if x is not None),
                       default=0)
        if max_len == 0:
            return np.zeros((len(seqs), 0), dtype=np.int64), fallback_mask

        out = np.full((len(seqs), max_len), int(self.id_pad), dtype=np.int64)
        for i, ids_row in enumerate(per_seq_ids):
            if ids_row:
                out[i, : len(ids_row)] = ids_row
        return out, fallback_mask

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
                replace_char = "N" if (self.invalid_char_id is None or self.invalid_char_id == self.id_N) else None
                if replace_char is not None and not all(ch in allowed for ch in text):
                    text = "".join(ch if ch in allowed else replace_char for ch in text)
            cleaned.append(text)
        return cleaned

    def _has_invalid_chars(self, seqs: List[str]) -> bool:
        if not seqs:
            return False
        joined = "".join(seqs)
        if self.normalize_case:
            joined = joined.upper()
            valid = _VALID_DNA_BYTES_UPPER
        else:
            valid = _VALID_DNA_BYTES_BOTH
        buf = joined.encode("ascii", errors="replace")
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size == 0:
            return False
        return bool(np.any(~np.isin(arr, valid)))

    def _resolve_kmer_missing_id(self) -> int:
        if self.invalid_char_id is not None:
            return int(self.invalid_char_id)
        if self.id_unk is not None:
            return int(self.id_unk)
        return int(self.id_pad)

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
        desired_side = self.padding_side if self.padding_side in ("left", "right") else old_side
        if old_side is not None and desired_side and old_side != desired_side:
            try:
                tok.padding_side = desired_side
            except Exception:
                pass

        try:
            enc = None
            if callable(tok):
                try:
                    # Always pad to the longest sequence in this batch — never
                    # to self.token_len. For BPE tokenizers (GENA-LM, DNABERT-2,
                    # METAGENE-1) the variable-length output makes
                    # padding="max_length" with max_length=model_max_length
                    # (often 32 768 or larger) inflate the H2D payload by up
                    # to 100x. If a caller needs a wider tensor it pads
                    # afterwards via _pad_ids_cpu using its own target.
                    enc = tok(
                        seqs,
                        add_special_tokens=False,
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    )
                except TypeError as e:
                    self.log.debug("tok call failed with TypeError: %s", e)
                    enc = None
                except Exception as e:
                    self.log.debug("tok call failed with Exception: %s", e)
                    raise

            if enc is None:
                ids_list = _tokenize_via_encode_api()
                if ids_list is None or len(ids_list) == 0:
                    raise RuntimeError(
                        f"Tokenizer is not callable and no encode/encode_batch/tokenize path produced IDs. Tok type: {type(tok)}"
                    )
                target_len = max(len(x) for x in ids_list)
                padding_side = desired_side or getattr(tok, "padding_side", "right")
                if padding_side not in ("left", "right"):
                    padding_side = "right"
                if self.token_len:
                    if any(len(x) > int(self.token_len) for x in ids_list):
                        raise ValueError("Tokenizer output longer than configured token_len.")
                    target_len = max(target_len, int(self.token_len))
                    padding_side = desired_side or "right"
                pad_val = int(getattr(self, "id_pad", 0))
                ids = torch.full((len(ids_list), target_len), pad_val, dtype=dtype)
                for i, seq_ids in enumerate(ids_list):
                    cur = torch.as_tensor(seq_ids, dtype=dtype)
                    cur_len = int(cur.numel())
                    if cur_len > target_len:
                        raise ValueError("Tokenizer output length exceeds padded target.")
                    if padding_side == "left" and cur_len < target_len:
                        ids[i, target_len - cur_len:] = cur
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
            if old_side is not None and desired_side and old_side != desired_side:
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
        # NOTE: self.token_len is informational (it is the model's max context),
        # not a required output shape. The old code padded every batch to
        # token_len which inflated H2D by ~100x for BPE tokenizers whose
        # output length is much smaller than model_max_length. Callers that
        # need a specific output shape should use encode_batch_with_padding.
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
                # Variable-length char-level: route to the var-length path
                # (per-sequence pad-on-the-right with id_pad to max length).
                # This is the NTv3 / Evo2 single-base nanopore_long regime
                # (1k–100k variable). Equal-length fast path is preserved
                # for the common case where the batch is already uniform.
                return self._encode_batch_ascii_lut_variable_length(seqs, lut)

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
        invalid_mask = None
        if self.handle_invalid_chars:
            valid_bytes = _VALID_DNA_BYTES_UPPER if self.normalize_case else _VALID_DNA_BYTES_BOTH
            invalid_mask = ~np.isin(arr, valid_bytes)
        ids = lut[arr]  # int64 [B,T]
        if invalid_mask is not None and invalid_mask.any():
            invalid_id = self.invalid_char_id if self.invalid_char_id is not None else self.id_N
            ids = np.array(ids, copy=True)
            ids[invalid_mask] = int(invalid_id)
        return ids

    def _encode_batch_ascii_lut_variable_length(
        self, seqs: List[str], lut: np.ndarray
    ) -> np.ndarray:
        """Char-level / single-base encoder for variable-length batches.

        Each sequence is byte-decoded, LUT-applied, then right-padded
        with ``id_pad`` to the batch's max length. No equal-length
        precondition.

        Used by the NTv3 and Evo2 paths when batch lengths vary (e.g.,
        nanopore_long with sequence lengths spanning 1 k – 100 kbp).
        """
        if lut is None:
            raise RuntimeError("DNATok.discover() must be called before encoding.")
        if not seqs:
            raise ValueError("No sequences provided for encoding.")
        T_max = max(len(s) for s in seqs)
        B = len(seqs)
        # Pre-allocate output filled with pad id.
        ids = np.full((B, T_max), int(self.id_pad), dtype=np.int64)
        for i, s in enumerate(seqs):
            encoded = s.upper() if self.normalize_case else s
            arr = np.frombuffer(encoded.encode("ascii", errors="replace"), dtype=np.uint8)
            n = arr.size
            row_ids = lut[arr].astype(np.int64, copy=False)
            if self.handle_invalid_chars:
                valid_bytes = _VALID_DNA_BYTES_UPPER if self.normalize_case else _VALID_DNA_BYTES_BOTH
                invalid_mask = ~np.isin(arr, valid_bytes)
                if invalid_mask.any():
                    invalid_id = self.invalid_char_id if self.invalid_char_id is not None else self.id_N
                    row_ids = np.array(row_ids, copy=True)
                    row_ids[invalid_mask] = int(invalid_id)
            if self.padding_side == "left":
                ids[i, T_max - n:] = row_ids
            else:
                ids[i, :n] = row_ids
        return ids

    def _maybe_pad(self, ids_np: np.ndarray) -> np.ndarray:
        if self.token_len is not None and self.token_len > ids_np.shape[1]:
            pad = self.token_len - ids_np.shape[1]
            if self.padding_side == "left":
                ids_np = np.pad(ids_np, ((0, 0), (pad, 0)), constant_values=self.id_pad)
            else:
                ids_np = np.pad(ids_np, ((0, 0), (0, pad)), constant_values=self.id_pad)
        return ids_np

    def _pad_ids_cpu(self, ids_cpu: torch.Tensor, target_len: int) -> torch.Tensor:
        if ids_cpu.shape[1] == target_len:
            return ids_cpu
        pad_len = target_len - ids_cpu.shape[1]
        pad = torch.full((ids_cpu.shape[0], pad_len), int(self.id_pad), dtype=ids_cpu.dtype)
        if self.padding_side == "left":
            return torch.cat([pad, ids_cpu], dim=1)
        return torch.cat([ids_cpu, pad], dim=1)

    def _pad_id_list(self, ids_list: List[List[int]], *, dtype: torch.dtype) -> torch.Tensor:
        if not ids_list:
            return torch.empty((0, 0), dtype=dtype)
        target_len = max(len(x) for x in ids_list)
        if self.token_len is not None and int(self.token_len) > target_len:
            target_len = int(self.token_len)
        out = torch.full((len(ids_list), target_len), int(self.id_pad), dtype=dtype)
        for i, seq_ids in enumerate(ids_list):
            if not seq_ids:
                continue
            cur = torch.as_tensor(seq_ids, dtype=dtype)
            cur_len = int(cur.numel())
            if self.padding_side == "left":
                out[i, target_len - cur_len:] = cur
            else:
                out[i, :cur_len] = cur
        return out

    def _encode_kmer_fallback_from_lut(
        self, seqs: List[str]
    ) -> Optional[tuple[List[List[int]], np.ndarray]]:
        if (
            self.kmer_k is None
            or self.kmer_lut is None
            or self.base5_lut is None
            or self.kmer_lut_offsets is None
            or self.kmer_lut_lengths is None
            or self.kmer_lut_flat is None
        ):
            return None
        if not seqs:
            return [], np.zeros(0, dtype=bool)

        k = self.kmer_k
        powers = np.power(5, np.arange(k - 1, -1, -1), dtype=np.int64)
        out: List[List[int]] = [[] for _ in seqs]
        success = np.zeros(len(seqs), dtype=bool)

        base5_lut = self.base5_lut
        kmer_lut = self.kmer_lut
        offsets = self.kmer_lut_offsets
        lengths = self.kmer_lut_lengths
        flat = self.kmer_lut_flat

        for i, seq in enumerate(seqs):
            if len(seq) % k != 0:
                continue
            text = seq.upper() if self.normalize_case else seq
            buf = text.encode("ascii", errors="replace")
            if len(buf) != len(seq):
                continue
            arr = np.frombuffer(buf, dtype=np.uint8)
            mapped = base5_lut[arr]
            if np.any(mapped < 0):
                continue
            mapped = mapped.reshape(-1, k).astype(np.int64)
            packed = np.dot(mapped, powers)
            seq_ids: List[int] = []
            ok = True
            for idx in packed:
                lut_id = kmer_lut[int(idx)]
                if lut_id >= 0:
                    seq_ids.append(int(lut_id))
                    continue
                off = int(offsets[int(idx)])
                length = int(lengths[int(idx)])
                if off < 0 or length <= 0:
                    ok = False
                    break
                seq_ids.extend(flat[off: off + length].tolist())
            if not ok:
                continue
            out[i] = seq_ids
            success[i] = True

        return out, success

    def _merge_kmer_fallback(
        self,
        seqs: List[str],
        ids_np: np.ndarray,
        fallback_mask: np.ndarray,
        *,
        dtype: torch.dtype,
        pin: bool,
    ) -> torch.Tensor:
        if not fallback_mask.any():
            ids_cpu = torch.as_tensor(ids_np, dtype=dtype)
            if pin:
                try:
                    ids_cpu = ids_cpu.pin_memory()
                except Exception:
                    pass
            return ids_cpu

        fallback_idx = np.where(fallback_mask)[0]
        fast_idx = np.where(~fallback_mask)[0]
        fallback_seqs = [seqs[i] for i in fallback_idx]

        kmer_fallback = self._encode_kmer_fallback_from_lut(fallback_seqs)
        kmer_ids_tensor = None
        kmer_idx = np.array([], dtype=int)
        tok_idx = np.array([], dtype=int)
        tok_ids_tensor = None

        if kmer_fallback is not None:
            ids_list, success = kmer_fallback
            if success.any():
                kmer_idx = np.where(success)[0]
                kmer_ids = [ids_list[i] for i in kmer_idx]
                kmer_ids_tensor = self._pad_id_list(kmer_ids, dtype=dtype)
            tok_idx = np.where(~success)[0]
        else:
            tok_idx = np.arange(len(fallback_seqs), dtype=int)

        if tok_idx.size > 0:
            tok_seqs = [fallback_seqs[i] for i in tok_idx]
            tok_ids_tensor = self._tokenize_batch_cpu(tok_seqs, dtype=dtype, pin=False)

        fast_len = ids_np.shape[1] if fast_idx.size > 0 else 0
        target_len = fast_len
        if kmer_ids_tensor is not None:
            target_len = max(target_len, kmer_ids_tensor.shape[1])
        if tok_ids_tensor is not None:
            target_len = max(target_len, tok_ids_tensor.shape[1])
        if self.token_len is not None and self.token_len > target_len:
            target_len = int(self.token_len)

        out = torch.full((len(seqs), target_len), int(self.id_pad), dtype=dtype)
        if fast_idx.size > 0:
            fast_ids = torch.as_tensor(ids_np[fast_idx], dtype=dtype)
            fast_ids = self._pad_ids_cpu(fast_ids, target_len)
            out[fast_idx] = fast_ids
        if kmer_ids_tensor is not None and kmer_idx.size > 0:
            kmer_ids_tensor = self._pad_ids_cpu(kmer_ids_tensor, target_len)
            out[fallback_idx[kmer_idx]] = kmer_ids_tensor
        if tok_ids_tensor is not None and tok_idx.size > 0:
            tok_ids_tensor = self._pad_ids_cpu(tok_ids_tensor, target_len)
            out[fallback_idx[tok_idx]] = tok_ids_tensor

        if not self._kmer_partial_warned:
            fallback_total = int(fallback_idx.size)
            if kmer_idx.size > 0 and tok_idx.size > 0:
                self.log.info(
                    "K-mer path fallback: exact LUT for %d/%d sequences; tokenizer for %d/%d.",
                    int(kmer_idx.size),
                    fallback_total,
                    int(tok_idx.size),
                    fallback_total,
                )
            elif kmer_idx.size > 0:
                self.log.info(
                    "K-mer path fallback: exact LUT used for %d/%d sequences.",
                    int(kmer_idx.size),
                    fallback_total,
                )
            else:
                self.log.info(
                    "K-mer path fallback: tokenizer used for %d/%d sequences.",
                    int(tok_idx.size),
                    fallback_total,
                )
            self._kmer_partial_warned = True

        if pin:
            try:
                out = out.pin_memory()
            except Exception:
                pass
        return out

    def encode_batch_to_ids(self, seqs: List[str]) -> torch.Tensor:
        """Backward-compatible path: returns CPU pinned int64 [B,T]."""
        if not seqs:
            raise ValueError("No sequences provided.")
        seqs = self._normalize_and_clean_seqs(list(seqs))

        if not self.use_ids_path:
            # GPU BPE backend (genomic BPE: DNABERT-2 / GENA-LM /
            # METAGENE-1). Tried first because it is bit-identical to HF
            # AND faster than CachedLMM on novel data (CachedLMM only
            # wins for streaming workloads with high cache-hit rates,
            # which the embed_from_strings / pre-tokenize paths do not
            # exhibit on diverse biological data). The backend output
            # is GPU-resident and right-padded; we D2H, then re-pad
            # according to self.padding_side to preserve this method's
            # contract (legacy callers may have set padding_side="left").
            if self.bpe_backend is not None:
                try:
                    ids_dev, mask_dev = self.bpe_backend.encode_batch(
                        seqs, device="cuda"
                    )
                    ids_cpu = ids_dev.cpu()
                    mask_cpu = mask_dev.cpu()
                    if self.padding_side != "right":
                        ids_list: List[List[int]] = []
                        for row, m in zip(ids_cpu.tolist(), mask_cpu.tolist()):
                            ids_list.append(
                                [t for t, mv in zip(row, m) if mv]
                            )
                        ids_cpu = self._pad_id_list(ids_list, dtype=torch.long)
                    try:
                        ids_cpu = ids_cpu.pin_memory()
                    except Exception:
                        pass
                    return ids_cpu
                except Exception as e:
                    self.log.warning(
                        "BPE backend failed (%s); falling back to CachedLMM / HF.", e
                    )
            # CachedLMM BPE (CPU, approximate). Acts as a streaming
            # cache: 100% bit-identical only when the safe-margin
            # condition holds for the input distribution; on diverse
            # held-out data we see ~85-99% match. Used as fallback when
            # bpe_backend isn't available.
            if self.lmm_bpe is not None:
                try:
                    out_lists = self.lmm_bpe.encode_batch(seqs)
                    return self._pad_id_list(out_lists, dtype=torch.long)
                except Exception as e:
                    self.log.warning(
                        "CachedLMM failed (%s); falling back to HF tokenizer.", e)
            return self._tokenize_batch_cpu(seqs, dtype=torch.long, pin=True)

        if self.require_valid_chars and not self.handle_invalid_chars:
            if self._has_invalid_chars(seqs):
                self.log.warning("Invalid characters detected; falling back to tokenizer.")
                return self._tokenize_batch_cpu(seqs, dtype=torch.long, pin=True)

        if self.kmer_k is not None:
            try:
                # The equal-length fast path requires all sequences in
                # the batch to be the same length. Real workloads
                # (Illumina reads at ~150 ± 30 bp, varied gene models,
                # etc.) usually fail this; route to the length-group
                # dispatcher in that case, which still uses numpy
                # vectorisation per group.
                same_len = True
                first_len = len(seqs[0])
                for s in seqs:
                    if len(s) != first_len:
                        same_len = False
                        break
                if same_len:
                    ids_np, fallback_mask = self._encode_batch_kmer_numpy_partial(seqs)
                else:
                    ids_np, fallback_mask = self._encode_batch_kmer_variable_length(seqs)
                ids_np = self._maybe_pad(ids_np)
                if fallback_mask.any():
                    return self._merge_kmer_fallback(
                        seqs, ids_np, fallback_mask, dtype=torch.long, pin=True
                    )
                ids_cpu = torch.as_tensor(ids_np, dtype=torch.long)
                try:
                    ids_cpu = ids_cpu.pin_memory()
                except Exception:
                    pass
                return ids_cpu
            except ValueError as e:
                self.log.warning("K-mer fast path failed (%s); falling back to tokenizer.", e)
                return self._tokenize_batch_cpu(seqs, dtype=torch.long, pin=True)

        if self.ascii_lut is None:
            return self._tokenize_batch_cpu(seqs, dtype=torch.long, pin=True)
        ids_np = self._maybe_pad(self._encode_batch_numpy(seqs))
        ids_cpu = torch.as_tensor(ids_np, dtype=torch.long)
        try:
            ids_cpu = ids_cpu.pin_memory()
        except Exception:
            pass
        return ids_cpu

    def encode_batch_with_padding(
        self,
        seqs: List[str],
        max_tokens: int,
        pad_to_max: bool = False,
        pad_multiple: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch with consistent truncation, padding, and attention mask.

        `max_tokens` is the maximum number of output tokens per sequence.
        For ASCII (1 byte → 1 token), one input character produces one token,
        so the character budget is `max_tokens`. For k-mer (k chars → 1 token,
        non-overlapping), the character budget is `max_tokens * k`.
        """
        if not seqs:
            raise ValueError("Empty batch")

        # 1. Convert token budget to character budget (k-mer aware) and truncate.
        kmer_k = int(self.kmer_k) if getattr(self, "kmer_k", None) else 1
        max_chars = int(max_tokens) * kmer_k
        seqs = [s[:max_chars] for s in seqs]

        # 2. Determine target length (in characters).
        char_lengths = [len(s) for s in seqs]
        max_len_chars = max(char_lengths)
        target_chars = max_len_chars
        if pad_to_max:
            target_chars = max(target_chars, max_chars)

        # K-mer requires character count divisible by k for clean alignment.
        if kmer_k > 1 and target_chars % kmer_k != 0:
            target_chars += kmer_k - (target_chars % kmer_k)

        # Now apply pad_multiple in *token* units (not character units).
        target_tokens = target_chars // kmer_k
        if pad_multiple > 1 and target_tokens % pad_multiple != 0:
            target_tokens += pad_multiple - (target_tokens % pad_multiple)
        target_chars = target_tokens * kmer_k

        old_token_len = self.token_len
        self.token_len = None
        try:
            if self.padding_side == "left":
                padded_seqs = [s.rjust(target_chars, "N") for s in seqs]
            else:
                padded_seqs = [s.ljust(target_chars, "N") for s in seqs]

            ids = self.encode_batch_to_ids(padded_seqs)  # [B, T_tokens]
            T = ids.shape[1]

            # Length in tokens for mask construction (not chars).
            token_lengths = torch.tensor(
                [(L + kmer_k - 1) // kmer_k for L in char_lengths],
                dtype=torch.long,
            )
            arange_buf = torch.arange(T, dtype=torch.long).unsqueeze(0)
            if self.padding_side == "left":
                mask_core = arange_buf >= (T - token_lengths.unsqueeze(1))
            else:
                mask_core = arange_buf < token_lengths.unsqueeze(1)

            # Pick the smallest integer dtype that holds the full vocab id range
            # without truncation. vocab_size is now populated by discover(); if
            # it is somehow missing we fall through to the (largest observed id)
            # from `ids` so a stale state still casts safely.
            pad_id_val = int(self.id_pad)
            vocab_size = int(self.vocab_size or 0)
            max_id = max(vocab_size - 1, pad_id_val)
            if vocab_size <= 0 and ids.numel() > 0:
                max_id = max(max_id, int(ids.max().item()))
            if max_id <= 255:
                target_dtype = torch.uint8
            elif max_id <= 32_767:
                target_dtype = torch.int16
            elif max_id <= 2_147_483_647:
                target_dtype = torch.int32
            else:
                target_dtype = torch.int64
            input_ids = ids.to(dtype=target_dtype)

            pad_id = pad_id_val
            input_ids[~mask_core] = pad_id

            attention_mask = mask_core.to(dtype=torch.uint8)
            return input_ids, attention_mask
        finally:
            self.token_len = old_token_len

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
        ids_cpu: Optional[torch.Tensor] = None

        kmer_failed = False

        # Try fast paths if enabled
        if self.use_ids_path:
            if self.require_valid_chars and not self.handle_invalid_chars:
                if self._has_invalid_chars(seqs):
                    self.log.warning("Invalid characters detected; falling back to tokenizer.")
                    return self._tokenize_batch_cpu(seqs, dtype=dtype, pin=True)
            # 1. K-mer path
            if self.kmer_k is not None:
                try:
                    ids_np, fallback_mask = self._encode_batch_kmer_numpy_partial(seqs)
                    ids_np = self._maybe_pad(ids_np)
                    if fallback_mask.any():
                        ids_cpu = self._merge_kmer_fallback(
                            seqs, ids_np, fallback_mask, dtype=dtype, pin=False
                        )
                except ValueError as e:
                    kmer_failed = True
                    ids_np = None

            # 2. ASCII LUT path (only if not a k-mer tokenizer)
            if ids_np is None and self.ascii_lut is not None and self.kmer_k is None:
                ids_np = self._encode_batch_numpy(seqs)
                ids_np = self._maybe_pad(ids_np)

        # 3. Fallback to tokenizer
        if ids_cpu is not None:
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

        if ids_np is None:
            if self.kmer_k is not None and kmer_failed:
                self.log.warning("K-mer fast path failed; falling back to tokenizer.")
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

        Operates on caller-supplied CPU ids (pinned or otherwise); does not
        require self.use_ids_path to be True. This enables BPE tokenizer
        fallbacks (HF Rust encode → pinned tensor) to reuse the same
        streamer.
        """
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
                        sub_cpu = ids_cpu[start_idx: start_idx + cur_bs]
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

        Operates on caller-supplied CPU ids; does not require self.use_ids_path
        to be True. BPE tokenizer fallbacks reuse this streamer.
        """
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
            offset = alloc_T - T if self.padding_side == "left" and alloc_T > T else 0
            with torch.cuda.stream(copy_stream):
                if use_int32_h2d:
                    if cur.dtype != torch.int32:
                        cur = cur.to(torch.int32, copy=True)
                    dev_i = self._dev_ping_i if into_ping else self._dev_pong_i
                    dev_l = self._dev_ping_l if into_ping else self._dev_pong_l
                    dev_i[:cur_bs, offset:offset + T].copy_(cur, non_blocking=True)
                    # cast to long on device (no extra H2D)
                    dev_l[:cur_bs, offset:offset + T].copy_(
                        dev_i[:cur_bs, offset:offset + T].to(torch.long)
                    )
                    (ready_ping if into_ping else ready_pong).record(copy_stream)
                else:
                    if cur.dtype != torch.long:
                        cur = cur.to(torch.long, copy=True)
                    dev_l = self._dev_ping_l if into_ping else self._dev_pong_l
                    dev_l[:cur_bs, offset:offset + T].copy_(cur, non_blocking=True)
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
        """Encode a same-length batch of DNA strings into a pinned uint8
        tensor [B, T] on the host. Optimised CPU path: writes each row
        directly into the pinned staging buffer, skipping the
        join → encode → frombuffer → copy pipeline of earlier revisions.

        Case normalisation and invalid-character replacement are vectorised
        as ufunc operations on the staging buffer (no Python per-character
        loops).
        """
        with _nvtx_range("DNAtok.encode_ascii_bytes"):
            return self._encode_ascii_bytes_impl(seqs)

    def _encode_ascii_bytes_impl(self, seqs: List[str]) -> torch.Tensor:
        if not seqs:
            raise ValueError("No sequences provided.")
        B = len(seqs)
        T = len(seqs[0])
        for s in seqs:
            if len(s) != T:
                raise ValueError("All sequences must have equal length.")

        # (Re)allocate persistent pinned staging buffer when shape changes.
        if (self._staging_bytes_cpu is None
                or self._staging_bytes_cpu.shape != (B, T)):
            buf = torch.empty((B, T), dtype=torch.uint8)
            try:
                buf = buf.pin_memory()
            except Exception:
                pass
            self._staging_bytes_cpu = buf

        # numpy view shares memory with the pinned tensor (CPU side).
        arr = self._staging_bytes_cpu.numpy()

        # Fill row-by-row from str.encode() — this skips the giant
        # "".join() intermediate allocation and the .reshape().copy()
        # bounce of the earlier code. The Python loop is short (B is
        # always <= 256 in practice; tokenisation throughput on this
        # box exceeds the loop overhead by 10x+).
        for i, s in enumerate(seqs):
            arr[i, :] = np.frombuffer(s.encode("ascii", errors="replace"),
                                      dtype=np.uint8)

        # Vectorised case normalisation: lowercase -> uppercase in place.
        if self.normalize_case:
            np.subtract(arr, 32, out=arr,
                        where=((arr >= ord("a")) & (arr <= ord("z"))))

        # Vectorised invalid-character replacement (to N).
        if self.handle_invalid_chars and (
                self.invalid_char_id is None or self.invalid_char_id == self.id_N):
            valid_bytes = _VALID_DNA_BYTES_UPPER if self.normalize_case else _VALID_DNA_BYTES_BOTH
            invalid = ~np.isin(arr, valid_bytes)
            if invalid.any():
                arr[invalid] = ord("N")

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
        invalid_mask = None
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
        if (
            self.handle_invalid_chars
            and invalid_mask is not None
            and invalid_mask.any()
            and (self.invalid_char_id is None or self.invalid_char_id == self.id_N)
        ):
            ascii_bytes_cpu[invalid_mask] = ord("N")

        dev = _as_torch_device(device)

        if self._lut_cuda is None or self._lut_cuda.device != dev:
            self._lut_cuda = torch.as_tensor(self.ascii_lut, dtype=torch.long, device=dev)

        ascii_dev = ascii_bytes_cpu.to(dev, non_blocking=True)
        # int32 indexing is sufficient for a 256-entry LUT and halves the
        # transient index buffer vs the legacy .long() promotion.
        ids_dev = self._lut_cuda[ascii_dev.int()]
        if (
            self.handle_invalid_chars
            and invalid_mask is not None
            and invalid_mask.any()
            and self.invalid_char_id is not None
            and self.invalid_char_id != self.id_N
        ):
            ids_dev[invalid_mask.to(device=dev, non_blocking=True)] = int(self.invalid_char_id)

        if self.ascii_start_lut is not None and ids_dev.shape[1] > 0:
            if self._lut_start_cuda is None or self._lut_start_cuda.device != dev:
                self._lut_start_cuda = torch.as_tensor(
                    self.ascii_start_lut, dtype=torch.long, device=dev
                )
            ids_dev[:, 0] = self._lut_start_cuda[ascii_dev[:, 0].int()]

        return ids_dev

    def _pad_device_to_len(self, ids_dev: torch.Tensor, target_len: int) -> torch.Tensor:
        if ids_dev.shape[1] == target_len:
            return ids_dev
        B = ids_dev.shape[0]
        out = torch.full((B, target_len), int(self.id_pad), dtype=ids_dev.dtype, device=ids_dev.device)
        if self.padding_side == "left":
            out[:, -ids_dev.shape[1]:] = ids_dev
        else:
            out[:, :ids_dev.shape[1]] = ids_dev
        return out

    def _expand_kmer_packed_cuda(
        self,
        packed: torch.Tensor,
        *,
        invalid_kmer: Optional[torch.Tensor],
        target_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._kmer_offsets_cuda is None
            or self._kmer_lengths_cuda is None
            or self._kmer_flat_cuda is None
            or self._kmer_lut_cuda is None
        ):
            raise RuntimeError("K-mer expansion LUTs are not available on device.")

        B, M = packed.shape
        packed_flat = packed.reshape(-1)

        single_flat = self._kmer_lut_cuda[packed_flat]
        lengths_flat = torch.where(
            single_flat >= 0,
            torch.ones_like(single_flat, dtype=torch.long),
            self._kmer_lengths_cuda[packed_flat].to(torch.long),
        )

        if invalid_kmer is not None:
            invalid_flat = invalid_kmer.reshape(-1)
            if invalid_flat.any().item():
                if self.invalid_char_id is None:
                    raise ValueError("Invalid characters found but invalid_char_id is unset.")
                single_flat = single_flat.clone()
                lengths_flat = lengths_flat.clone()
                single_flat[invalid_flat] = int(self.invalid_char_id)
                lengths_flat[invalid_flat] = 1

        missing_flat = (single_flat < 0) & (lengths_flat <= 0)
        if missing_flat.any().item():
            if self.allow_incomplete_kmer_lut:
                fill_id = self._resolve_kmer_missing_id()
                single_flat = single_flat.clone()
                lengths_flat = lengths_flat.clone()
                single_flat[missing_flat] = int(fill_id)
                lengths_flat[missing_flat] = 1
            else:
                raise ValueError("Sequence contains K-mers not supported by single tokens.")

        lengths_2d = lengths_flat.view(B, M)
        row_lengths = lengths_2d.sum(dim=1)
        if target_len is None:
            target_len = int(row_lengths.max().item()) if row_lengths.numel() else 0

        if target_len == 0:
            return torch.empty((B, 0), dtype=torch.long, device=packed.device), row_lengths

        row_offsets = torch.cumsum(lengths_2d, dim=1) - lengths_2d
        row_offsets_flat = row_offsets.reshape(-1)

        total = int(lengths_flat.sum().item())
        out = torch.full((B, target_len), int(self.id_pad), dtype=torch.long, device=packed.device)
        if total == 0:
            return out, row_lengths

        kmer_index = torch.repeat_interleave(
            torch.arange(packed_flat.numel(), device=packed.device),
            lengths_flat,
        )
        start_flat = torch.cumsum(lengths_flat, dim=0) - lengths_flat
        within = torch.arange(total, device=packed.device) - start_flat[kmer_index]

        row_ids_flat = torch.arange(B, device=packed.device).unsqueeze(1).expand(B, M).reshape(-1)
        out_row = row_ids_flat[kmer_index]
        if self.padding_side == "left":
            row_shift = (target_len - row_lengths).to(device=packed.device)
            out_pos = row_offsets_flat[kmer_index] + within + row_shift[out_row]
        else:
            out_pos = row_offsets_flat[kmer_index] + within

        single_for_elem = single_flat[kmer_index]
        use_single = single_for_elem >= 0
        token_ids = torch.empty(total, dtype=torch.long, device=packed.device)
        if use_single.any().item():
            token_ids[use_single] = single_for_elem[use_single]
        if (~use_single).any().item():
            offsets_flat = self._kmer_offsets_cuda[packed_flat]
            offsets_for_elem = offsets_flat[kmer_index][~use_single]
            token_ids[~use_single] = self._kmer_flat_cuda[offsets_for_elem + within[~use_single]]

        out[out_row, out_pos] = token_ids
        return out, row_lengths

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
        if self._base5_lut_cuda is None or self._base5_lut_cuda.device != dev:
            self._base5_lut_cuda = torch.as_tensor(self.base5_lut, dtype=torch.long, device=dev)
        if self._kmer_lut_cuda is None or self._kmer_lut_cuda.device != dev:
            self._kmer_lut_cuda = torch.as_tensor(self.kmer_lut, dtype=torch.long, device=dev)
        if self.kmer_lut_offsets is not None and (
            self._kmer_offsets_cuda is None or self._kmer_offsets_cuda.device != dev
        ):
            self._kmer_offsets_cuda = torch.as_tensor(
                self.kmer_lut_offsets, dtype=torch.long, device=dev
            )
        if self.kmer_lut_lengths is not None and (
            self._kmer_lengths_cuda is None or self._kmer_lengths_cuda.device != dev
        ):
            self._kmer_lengths_cuda = torch.as_tensor(
                self.kmer_lut_lengths, dtype=torch.long, device=dev
            )
        if self.kmer_lut_flat is not None and (
            self._kmer_flat_cuda is None or self._kmer_flat_cuda.device != dev
        ):
            self._kmer_flat_cuda = torch.as_tensor(self.kmer_lut_flat, dtype=torch.long, device=dev)

        # Move bytes to device, promoting to int32 for the gather
        # (uint8 is treated as a mask; int32 halves the temp vs .long()).
        ascii_dev = ascii_bytes_cpu.to(dev, non_blocking=True).int()  # [B, T]

        # Map to base5: 0..4
        base5_dev = self._base5_lut_cuda[ascii_dev]  # [B, T]
        invalid_mask = base5_dev < 0
        invalid_kmer = None
        if invalid_mask.any().item():
            if not self.handle_invalid_chars:
                raise ValueError("Invalid characters found for K-mer path.")
            if self.kmer_invalid_policy == "replace_with_n":
                base5_dev = base5_dev.clone()
                base5_dev[invalid_mask] = 4
            elif self.kmer_invalid_policy == "map_to_unk":
                if self.invalid_char_id is None:
                    raise ValueError("Invalid characters found but invalid_char_id is unset.")
                base5_dev = base5_dev.clone()
                base5_dev[invalid_mask] = 0
                invalid_kmer = invalid_mask.view(B, T // k, k).any(dim=-1)
            else:
                raise ValueError("Invalid characters found for K-mer path.")

        # Reshape to [B, T/k, k]
        base5_dev = base5_dev.view(B, T // k, k)

        # Pack into integers
        if self._kmer_weights_cuda is None or self._kmer_weights_cuda.device != dev:
            weights = torch.tensor([5 ** (k - 1 - i) for i in range(k)], dtype=torch.long, device=dev)
            self._kmer_weights_cuda = weights

        # [B, T/k, k] * [k] -> [B, T/k, k] -> sum(-1) -> [B, T/k]
        # Avoid matmul for Long on CUDA
        packed = torch.sum(base5_dev * self._kmer_weights_cuda, dim=-1)

        # Lookup IDs
        ids_dev = self._kmer_lut_cuda[packed]
        if invalid_kmer is not None:
            ids_dev = ids_dev.clone()
            ids_dev[invalid_kmer] = int(self.invalid_char_id)
        missing = ids_dev < 0
        if missing.any().item():
            # Indicates a k-mer (e.g., containing 'N') that does not map to a single token.
            if self.allow_incomplete_kmer_lut:
                fill_id = self._resolve_kmer_missing_id()
                ids_dev = ids_dev.clone()
                ids_dev[missing] = int(fill_id)
            else:
                if (
                    self._kmer_offsets_cuda is None
                    or self._kmer_lengths_cuda is None
                    or self._kmer_flat_cuda is None
                ):
                    raise ValueError("Sequence contains K-mers not supported by single tokens.")
                row_missing = torch.any(missing, dim=1)
                if torch.all(row_missing):
                    ids_dev, _ = self._expand_kmer_packed_cuda(
                        packed, invalid_kmer=invalid_kmer, target_len=None
                    )
                else:
                    packed_missing = packed[row_missing]
                    invalid_missing = invalid_kmer[row_missing] if invalid_kmer is not None else None
                    expanded_missing, _ = self._expand_kmer_packed_cuda(
                        packed_missing, invalid_kmer=invalid_missing, target_len=None
                    )
                    target_len = max(int(ids_dev.shape[1]), int(expanded_missing.shape[1]))
                    out = torch.full(
                        (B, target_len), int(self.id_pad), dtype=torch.long, device=dev
                    )
                    if (~row_missing).any().item():
                        offset = target_len - ids_dev.shape[1] if self.padding_side == "left" else 0
                        out[~row_missing, offset:offset + ids_dev.shape[1]] = ids_dev[~row_missing]
                    if row_missing.any().item():
                        if expanded_missing.shape[1] != target_len:
                            expanded_missing = self._pad_device_to_len(expanded_missing, target_len)
                        out[row_missing] = expanded_missing
                    ids_dev = out

        return ids_dev

    def ids_from_ascii_bytes_cuda(
            self, ascii_bytes_cpu: torch.Tensor, device: object = "cuda:0"
    ) -> torch.Tensor:
        return self._map_ascii_bytes_to_ids_cuda(ascii_bytes_cpu, device)

    # -------------------- Device-side left padding -------------------------
    def _pad_device(self, ids_dev: torch.Tensor, T: int, dev: torch.device) -> torch.Tensor:
        """Pad on device to self.token_len if needed, honoring padding_side."""
        if not self.token_len or self.token_len <= T:
            return ids_dev
        B = ids_dev.shape[0]
        out = torch.empty((B, self.token_len), dtype=ids_dev.dtype, device=dev)
        out.fill_(self.id_pad)
        if self.padding_side == "left":
            out[:, -T:].copy_(ids_dev)
        else:
            out[:, :T].copy_(ids_dev)
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
        if self.require_valid_chars and not self.handle_invalid_chars:
            if self._has_invalid_chars(seqs):
                path = "ids"
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

                # Pad to token_len if needed
                ids_dev = self._pad_device(ids_dev, int(ids_dev.shape[1]), dev)
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

        # "ids" path — works for both the fast LUT path (use_ids_path=True)
        # and for the BPE / tokenizer-fallback path (use_ids_path=False).
        # encode_batch_to_ids_staging routes BPE through _tokenize_batch_cpu
        # with pin=True, so the H2D step still benefits from pinned memory
        # and the iter_embed_tokens_pipelined ping-pong overlap.
        #
        # Genomic-BPE fast path: when the GPU BPE backend is built
        # (DNABERT-2 / GENA-LM / METAGENE-1), we go directly from byte
        # strings → GPU ids → embed without touching the CPU at all.
        # That skips both the HF-tokeniser CPU encode AND the staging
        # H2D copy.
        if self.bpe_backend is not None and not self.use_ids_path:
            dev = _as_torch_device(device)
            kwargs_template = {}
            try:
                sig = inspect.signature(self.embedder.embed_tokens)
                if "rc_invariant" in sig.parameters:
                    kwargs_template["rc_invariant"] = False
            except Exception:
                pass
            try:
                ids_dev, _mask = self.bpe_backend.encode_batch(seqs, device=dev)
            except Exception as e:
                self.log.warning(
                    "BPE backend failed (%s); falling back to HF tokenizer path.", e
                )
            else:
                B_total = int(ids_dev.shape[0])
                start = 0
                while start < B_total:
                    end = min(B_total, start + emb_batch)
                    sub = ids_dev[start:end]
                    with torch.amp.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                            enabled=_is_cuda_device(dev)):
                        out = self.embedder.embed_tokens(sub, **kwargs_template)
                    if out.device != dev:
                        out = out.to(device=dev, non_blocking=True)
                    if self.force_fp32_outputs and out.dtype != torch.float32:
                        out = out.float()
                    yield out
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

    # ===================================================================
    # OPTIMIZED PATHS (opt-in; standard behavior preserved when off)
    # ===================================================================

    # -- 2-bit nucleotide packing --------------------------------------
    # ACGT in two bits: A=0b00, C=0b01, G=0b10, T=0b11. N (and everything
    # else) is pre-replaced with N's mapping on the CPU side via the LUT,
    # so the packed stream stays {0,1,2,3} per base — four bases per byte.
    # Two-pass design:
    #   CPU:   string → uint8 ASCII → 2-bit packed [B, ceil(T/4)]
    #   PCIe:  half / quarter of the uncompressed payload
    #   GPU:   unpack + ascii LUT lookup in one fused kernel
    # The header byte stride is 4 bases per byte; the trailing partial byte
    # is padded with the "A" code (0) and masked out by the downstream pad
    # routine using sequence-length metadata.
    _ASCII_TO_2BIT: Optional[np.ndarray] = None

    @classmethod
    def _ascii_to_2bit_lut(cls) -> np.ndarray:
        if cls._ASCII_TO_2BIT is None:
            lut = np.zeros(256, dtype=np.uint8)  # default A=0 (safe filler)
            lut[ord("A")] = 0
            lut[ord("C")] = 1
            lut[ord("G")] = 2
            lut[ord("T")] = 3
            lut[ord("N")] = 0  # represented as A in the packed payload; masked downstream
            for c in "ACGTN":
                lut[ord(c.lower())] = lut[ord(c)]
            cls._ASCII_TO_2BIT = lut
        return cls._ASCII_TO_2BIT

    def encode_batch_to_packed_2bit(self, seqs: List[str]) -> Tuple[torch.Tensor, int]:
        """Pack DNA bases ACGT(N) into a 2-bit stream (4 bases per byte).

        Returns (packed_uint8 tensor of shape [B, ceil(T/4)], true T).
        Caller is expected to know the un-padded length T for unpacking.

        Invalid-character handling matches the standard `encode_batch_to_ascii_bytes`
        path when `handle_invalid_chars=True`: non-ACGTN characters are
        replaced with the N code (which maps to A in the packed stream
        but is corrected by start-LUT / pad metadata downstream). When
        `handle_invalid_chars=False`, invalid characters fall through and
        the caller is responsible for ensuring input validity.
        """
        if not seqs:
            raise ValueError("No sequences provided.")
        seqs = self._normalize_and_clean_seqs(list(seqs))
        T = len(seqs[0])
        for s in seqs:
            if len(s) != T:
                raise ValueError("All sequences must have equal length.")
        B = len(seqs)
        Tp = (T + 3) // 4  # output bytes per row

        # Allocate or reuse the pinned destination buffer up-front so the bit
        # pack can write directly into it (one alloc, no torch.from_numpy
        # round-trip and no intermediate `packed_np`).
        if (
            self._packed_2bit_cpu is None
            or tuple(self._packed_2bit_cpu.shape) != (B, Tp)
        ):
            buf = torch.empty((B, Tp), dtype=torch.uint8)
            try:
                buf = buf.pin_memory()
            except Exception:
                pass
            self._packed_2bit_cpu = buf
        packed_np = self._packed_2bit_cpu.numpy()  # shares pinned memory

        # Encode ascii once into a contiguous scratch view we already own
        # (an extra reuse pool would not pay off — encoding rarely changes
        # shape between calls and the buffer is small).
        arr = np.empty((B, T), dtype=np.uint8)
        for i, s in enumerate(seqs):
            arr[i, :] = np.frombuffer(s.encode("ascii", errors="replace"),
                                      dtype=np.uint8)
        if self.handle_invalid_chars and (
                self.invalid_char_id is None or self.invalid_char_id == self.id_N):
            valid_bytes = _VALID_DNA_BYTES_UPPER if self.normalize_case else _VALID_DNA_BYTES_BOTH
            invalid = ~np.isin(arr, valid_bytes)
            if invalid.any():
                arr[invalid] = ord("N")

        lut = self._ascii_to_2bit_lut()
        codes = lut[arr]  # [B, T] uint8 in {0,1,2,3}

        if Tp * 4 != T:
            pad = np.zeros((B, Tp * 4 - T), dtype=np.uint8)
            codes = np.concatenate([codes, pad], axis=1)

        # Pack 4 codes per byte, low-position-first within a byte:
        #   byte = c0 | (c1<<2) | (c2<<4) | (c3<<6)
        # Write each shift result straight into the pinned destination via
        # `out=` to avoid the extra `astype(uint8)` cycle.
        c = codes.reshape(B, Tp, 4)
        np.bitwise_or(c[..., 0], c[..., 1] << 2, out=packed_np)
        np.bitwise_or(packed_np, c[..., 2] << 4, out=packed_np)
        np.bitwise_or(packed_np, c[..., 3] << 6, out=packed_np)
        return self._packed_2bit_cpu, T

    def _unpack_2bit_to_ids_cuda(
        self,
        packed_cpu: torch.Tensor,
        T_true: int,
        device: object,
    ) -> torch.Tensor:
        """H2D the packed bytes, then unpack + LUT on the device in one pass.

        Result: int64 [B, T_true] token-id tensor.
        """
        if self.ascii_lut is None:
            raise RuntimeError("discover() must be called before 2-bit unpack.")
        dev = _as_torch_device(device)

        # Persistent LUTs on device
        if self._lut_cuda is None or self._lut_cuda.device != dev:
            self._lut_cuda = torch.as_tensor(self.ascii_lut, dtype=torch.long, device=dev)
        # 2-bit code → ASCII byte LUT on device
        if not hasattr(self, "_lut_2bit_to_ascii_cuda") or self._lut_2bit_to_ascii_cuda is None \
                or self._lut_2bit_to_ascii_cuda.device != dev:
            code_to_ascii = torch.tensor(
                [ord("A"), ord("C"), ord("G"), ord("T")], dtype=torch.long, device=dev
            )
            self._lut_2bit_to_ascii_cuda = code_to_ascii

        packed_dev = packed_cpu.to(dev, non_blocking=True)  # uint8 [B, Tp]
        B, Tp = packed_dev.shape

        # Unpack each byte into its four codes via bit-twiddling on device,
        # keeping the working tensor as uint8 throughout (8x less memory
        # bandwidth than promoting to int64 for the bit ops).
        p = packed_dev  # uint8 [B, Tp]
        c0 = (p & 0b11)
        c1 = ((p >> 2) & 0b11)
        c2 = ((p >> 4) & 0b11)
        c3 = ((p >> 6) & 0b11)
        codes = torch.stack([c0, c1, c2, c3], dim=2).reshape(B, Tp * 4)  # uint8 [B, Tp*4]
        if codes.shape[1] != T_true:
            codes = codes[:, :T_true]

        # Fused 4-entry "2-bit code -> token id" LUT, derived from the ASCII LUT.
        # This collapses the previous two-stage (code -> ASCII -> id) gather
        # into a single device-side lookup.
        if not hasattr(self, "_lut_2bit_to_id_cuda") or self._lut_2bit_to_id_cuda is None \
                or self._lut_2bit_to_id_cuda.device != dev:
            self._lut_2bit_to_id_cuda = torch.tensor(
                [int(self.ascii_lut[ord(c)]) for c in "ACGT"],
                dtype=torch.long, device=dev,
            )
        # PyTorch indexing rejects uint8 (treated as a mask) so we cast to
        # int32 — half the memory of int64 for the transient index tensor.
        ids_dev = self._lut_2bit_to_id_cuda[codes.int()]

        # Start LUT correction for the first column (if needed). We need the
        # ASCII byte of position 0 to index the start LUT, so reconstruct
        # just that column.
        if self.ascii_start_lut is not None and ids_dev.shape[1] > 0:
            if self._lut_start_cuda is None or self._lut_start_cuda.device != dev:
                self._lut_start_cuda = torch.as_tensor(
                    self.ascii_start_lut, dtype=torch.long, device=dev
                )
            code0 = codes[:, 0].int()
            ascii0 = self._lut_2bit_to_ascii_cuda[code0]
            ids_dev[:, 0] = self._lut_start_cuda[ascii0]
        return ids_dev

    # -- 2-bit packed bytes -> uint8 IDs (combined) --------------------
    def _unpack_2bit_to_ids_cuda_u8(
        self,
        packed_cpu: torch.Tensor,
        T_true: int,
        device: object,
    ) -> torch.Tensor:
        """Combined 2-bit-pack + uint8 LUT path. Sends quarter-size PCIe
        payload (2-bit pack), unpacks on device, gathers into a uint8
        token-id buffer, and promotes to int64 only at the end for
        embed_tokens compatibility.

        Compared to `_unpack_2bit_to_ids_cuda`, this collapses the
        device-side token-id tensor from int64 to uint8 — 8x less device
        memory and L1 traffic.
        """
        if not self._ascii_lut_fits_uint8():
            raise RuntimeError("uint8 IDs path requires vocab <= 256.")
        dev = _as_torch_device(device)

        # 4-entry uint8 LUT keyed by 2-bit nucleotide code (A=0,C=1,G=2,T=3).
        attr = "_lut_2bit_to_id_u8_cuda"
        if not hasattr(self, attr) or getattr(self, attr) is None \
                or getattr(self, attr).device != dev:
            lut_u8 = torch.tensor(
                [int(self.ascii_lut[ord(c)]) for c in "ACGT"],
                dtype=torch.uint8, device=dev,
            )
            setattr(self, attr, lut_u8)
        lut_u8 = getattr(self, attr)

        packed_dev = packed_cpu.to(dev, non_blocking=True)
        B, Tp = packed_dev.shape
        p = packed_dev  # uint8 [B, Tp]
        c0 = (p & 0b11)
        c1 = ((p >> 2) & 0b11)
        c2 = ((p >> 4) & 0b11)
        c3 = ((p >> 6) & 0b11)
        codes = torch.stack([c0, c1, c2, c3], dim=2).reshape(B, Tp * 4)
        if codes.shape[1] != T_true:
            codes = codes[:, :T_true]

        ids_u8 = lut_u8[codes.int()]               # uint8 [B, T_true]
        ids = ids_u8.to(torch.long)                # promote for embed_tokens

        if self.ascii_start_lut is not None and ids.shape[1] > 0:
            if self._lut_start_cuda is None or self._lut_start_cuda.device != dev:
                self._lut_start_cuda = torch.as_tensor(
                    self.ascii_start_lut, dtype=torch.long, device=dev
                )
            # Reconstruct only column 0's ASCII byte for start-LUT lookup
            if not hasattr(self, "_lut_2bit_to_ascii_cuda") or self._lut_2bit_to_ascii_cuda is None \
                    or self._lut_2bit_to_ascii_cuda.device != dev:
                self._lut_2bit_to_ascii_cuda = torch.tensor(
                    [ord("A"), ord("C"), ord("G"), ord("T")], dtype=torch.long, device=dev,
                )
            ascii0 = self._lut_2bit_to_ascii_cuda[codes[:, 0].int()]
            ids[:, 0] = self._lut_start_cuda[ascii0]
        return ids

    # -- uint8 IDs on the bus ------------------------------------------
    def _ascii_lut_fits_uint8(self) -> bool:
        if self.ascii_lut is None:
            return False
        return int(self.ascii_lut.max(initial=0)) <= 255 and int(self.id_pad) <= 255

    def _ensure_uint8_lut_cuda(self, dev: torch.device) -> torch.Tensor:
        if not self._ascii_lut_fits_uint8():
            raise RuntimeError("uint8 LUT path requires vocab <= 256.")
        if (
            self._lut_u8_cuda is None
            or self._lut_u8_cuda.device != dev
        ):
            self._lut_u8_cuda = torch.as_tensor(
                self.ascii_lut, dtype=torch.uint8, device=dev
            )
        return self._lut_u8_cuda

    def _map_ascii_bytes_to_ids_cuda_u8(
        self, ascii_bytes_cpu: torch.Tensor, device: object
    ) -> torch.Tensor:
        """uint8 variant of the ASCII→ID mapping path. Promotes to int64 on
        device for downstream embed compute, but the LUT lookup itself
        materialises into a uint8 tensor so the LUT cache (and the kernel's
        memory traffic) is 8× smaller than the int64 LUT used elsewhere.
        """
        dev = _as_torch_device(device)
        if ascii_bytes_cpu.device.type != "cpu" or ascii_bytes_cpu.dtype != torch.uint8:
            raise TypeError("expected pinned uint8 CPU tensor")
        lut_u8 = self._ensure_uint8_lut_cuda(dev)
        ascii_dev = ascii_bytes_cpu.to(dev, non_blocking=True)
        # PyTorch treats uint8 indices as a mask, so int promotion is required.
        # int32 halves the transient index buffer relative to .long().
        ids_u8 = lut_u8[ascii_dev.int()]
        ids = ids_u8.to(torch.long)  # promote result for embed_tokens
        if self.ascii_start_lut is not None and ids.shape[1] > 0:
            if self._lut_start_cuda is None or self._lut_start_cuda.device != dev:
                self._lut_start_cuda = torch.as_tensor(
                    self.ascii_start_lut, dtype=torch.long, device=dev
                )
            ids[:, 0] = self._lut_start_cuda[ascii_dev[:, 0].int()]
        return ids

    # -- CUDA graph capture/replay -------------------------------------
    def _record_cuda_graph(
        self,
        seqs_template: List[str],
        device: object,
        emb_batch: int,
    ) -> None:
        """Capture a static-shape encode→embed graph using a real batch of
        sequences as the template. Subsequent inference can be replayed by
        copying new bytes into `_graph_static_in` and calling `_graph.replay()`.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require a CUDA device.")
        dev = _as_torch_device(device)

        ascii_cpu = self.encode_batch_to_ascii_bytes(seqs_template)
        B, T = int(ascii_cpu.shape[0]), int(ascii_cpu.shape[1])
        self._graph_shape = (B, T)
        self._graph_dev = dev

        # Persistent device input buffer for graph
        self._graph_static_in = torch.empty(
            (B, T), dtype=torch.uint8, device=dev
        )
        # Persistent output id tensor (graphs require fixed addresses)
        self._graph_static_out = torch.empty(
            (B, T), dtype=torch.long, device=dev
        )
        # Direct CPU->static device buffer copy (no temp allocation).
        self._graph_static_in.copy_(ascii_cpu, non_blocking=True)

        # Warmup before capture (CUDA graph requirement)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if self._lut_cuda is None or self._lut_cuda.device != dev:
                    self._lut_cuda = torch.as_tensor(
                        self.ascii_lut, dtype=torch.long, device=dev
                    )
                _ = self._lut_cuda[self._graph_static_in.int()]
        torch.cuda.current_stream().wait_stream(s)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            ids = self._lut_cuda[self._graph_static_in.int()]
            self._graph_static_out.copy_(ids)

    def _replay_cuda_graph(self, ascii_cpu: torch.Tensor) -> torch.Tensor:
        if (
            self._graph is None
            or self._graph_static_in is None
            or self._graph_static_out is None
        ):
            raise RuntimeError("CUDA graph not recorded.")
        if tuple(ascii_cpu.shape) != self._graph_shape:
            raise ValueError(
                f"Graph shape mismatch: recorded {self._graph_shape}, got {tuple(ascii_cpu.shape)}"
            )
        # Direct cross-device copy CPU(pinned) -> static device buffer.
        # An earlier revision wrote
        #     self._graph_static_in.copy_(ascii_cpu.to(dev, non_blocking=True))
        # which allocated a transient device tensor and did an extra D2D copy;
        # on Ampere that copy was visible against the LUT op itself.
        self._graph_static_in.copy_(ascii_cpu, non_blocking=True)
        self._graph.replay()
        return self._graph_static_out

    # -- Triton fused tokenize + gather --------------------------------
    def _try_import_triton(self) -> bool:
        if self._fused_kernel_available is not None:
            return self._fused_kernel_available
        try:
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401
            # The lg-asm recipe (TRITON_PTXAS_PATH=system ptxas +
            # TORCH_CUDA_ARCH_LIST="12.1+PTX") is applied by dnatok_compat at
            # import time; Triton compiles natively to sm_121 on the GB10.
            self._fused_kernel_available = True
        except Exception as e:
            self.log.warning("Triton unavailable (%s); fused kernel disabled.", e)
            self._fused_kernel_available = False
        return self._fused_kernel_available

    def _fused_tokenize_gather(
        self,
        ascii_bytes_cpu: torch.Tensor,
        embed_weight: torch.Tensor,
        device: object,
    ) -> torch.Tensor:
        """Fused (byte→id→embedding-gather) Triton kernel.

        Equivalent to: emb = embed_weight[ ascii_lut[ ascii_bytes ] ]
        but reads each byte once, computes the LUT lookup, and gathers the
        embedding row in a single kernel — saving one full pass over the
        intermediate id tensor.
        """
        if not self._try_import_triton():
            raise RuntimeError("Triton not available; cannot run fused kernel.")
        import triton
        import triton.language as tl

        if self.ascii_lut is None:
            raise RuntimeError("discover() must be called before fused kernel.")
        dev = _as_torch_device(device)
        ascii_dev = ascii_bytes_cpu.to(dev, non_blocking=True)
        if self._lut_cuda is None or self._lut_cuda.device != dev:
            self._lut_cuda = torch.as_tensor(
                self.ascii_lut, dtype=torch.long, device=dev
            )

        B, T = ascii_dev.shape
        D = int(embed_weight.shape[1])
        out = torch.empty((B, T, D), dtype=embed_weight.dtype, device=dev)

        @triton.jit
        def _fused_kernel(
            ascii_ptr, lut_ptr, emb_ptr, out_ptr,
            B, T, D, V,
            stride_emb_v, stride_emb_d,
            stride_out_b, stride_out_t, stride_out_d,
            BLOCK_D: tl.constexpr,
        ):
            pid_bt = tl.program_id(0)
            pid_d = tl.program_id(1)
            b = pid_bt // T
            t = pid_bt % T
            if b >= B:
                return
            byte_val = tl.load(ascii_ptr + b * T + t).to(tl.int32)
            token_id = tl.load(lut_ptr + byte_val).to(tl.int32)
            offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = offs_d < D
            emb_row = tl.load(
                emb_ptr + token_id * stride_emb_v + offs_d * stride_emb_d,
                mask=mask, other=0.0,
            )
            tl.store(
                out_ptr + b * stride_out_b + t * stride_out_t + offs_d * stride_out_d,
                emb_row, mask=mask,
            )

        BLOCK_D = 128
        grid = (B * T, (D + BLOCK_D - 1) // BLOCK_D)
        _fused_kernel[grid](
            ascii_dev, self._lut_cuda, embed_weight, out,
            B, T, D, int(embed_weight.shape[0]),
            embed_weight.stride(0), embed_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_D=BLOCK_D,
        )
        return out

    def free_buffers(self) -> None:
        """Explicitly release pinned memory buffers."""
        self._staging_ids_cpu = None
        self._staging_bytes_cpu = None
        self._packed_2bit_cpu = None
        self._packed_2bit_dev = None
        self._dev_ping_i = None
        self._dev_pong_i = None
        self._dev_ping_l = None
        self._dev_pong_l = None
        self._lut_cuda = None
        self._lut_start_cuda = None
        self._lut_u8_cuda = None
        # 2-bit-specific device caches (otherwise these leak across discover()
        # cycles or device changes).
        for attr in ("_lut_2bit_to_ascii_cuda", "_lut_2bit_to_id_cuda",
                     "_lut_2bit_to_id_u8_cuda"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        # K-mer device caches
        self._base5_lut_cuda = None
        self._kmer_lut_cuda = None
        self._kmer_offsets_cuda = None
        self._kmer_lengths_cuda = None
        self._kmer_flat_cuda = None
        self._kmer_weights_cuda = None
        self._graph = None
        self._graph_static_in = None
        self._graph_static_out = None
        self._graph_shape = None
        self._graph_dev = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
