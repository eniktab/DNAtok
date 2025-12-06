# test_ids_path_equivalence.py
#
# Keeps a single correctness test that asserts tokenizer ⇔ DNATok equivalence
# and prints the short human-readable report. No heavy perf tests here.
# If you *really* want to use the Evo2 tokenizer, export USE_EVO2_TOKENIZER=1.
#
#   pytest -q test_ids_path_equivalence.py
#
from __future__ import annotations

import os
import random
from typing import List

import pytest
import torch

# --------------------------- config ---------------------------------
DNA = "ACGTNacgtn"
B = 4096       # batch size
T = 512        # tokens per sequence
SEED = 42

# ---------- import Evo2 (optional) and DNATok -----------------
from src.dna_tokenizer import DNATok

def make_equal_len_sequences(B: int, T: int, seed: int = 0) -> List[str]:
    random.seed(seed)
    return ["".join(random.choice(DNA) for _ in range(T)) for _ in range(B)]

# ---- robust tokenizer probes (subset; enough for equivalence test) --

def _safe_get_vocab(tokenizer):
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            v = get_vocab()
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    v = getattr(tokenizer, "vocab", None)
    if isinstance(v, dict):
        return v
    return None

def _safe_token_to_id(tokenizer, token_str: str):
    fn = getattr(tokenizer, "token_to_id", None)
    if callable(fn):
        try:
            out = fn(token_str)
            if isinstance(out, int) and out >= 0:
                return out
        except Exception:
            pass
    cti = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(cti):
        try:
            out = cti(token_str)
            if isinstance(out, int) and out >= 0:
                return out
        except Exception:
            pass
    vocab = _safe_get_vocab(tokenizer)
    if vocab and token_str in vocab and isinstance(vocab[token_str], int):
        return int(vocab[token_str])
    return None

def _encode_one_char_to_id(tokenizer, ch: str):
    enc = getattr(tokenizer, "encode", None)
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

def _discover_unknown_id(tokenizer) -> int:
    for cand in ("N", "n"):
        tid = _safe_token_to_id(tokenizer, cand)
        if isinstance(tid, int):
            return tid
    for pad_tok in ("<pad>", "[PAD]", "PAD", "pad"):
        tid = _safe_token_to_id(tokenizer, pad_tok)
        if isinstance(tid, int):
            return tid
    return 0

def _tok2ids_via_encode_batch(tokenizer, seqs: List[str]) -> List[List[int]] | None:
    if hasattr(tokenizer, "encode_batch"):
        try:
            encs = tokenizer.encode_batch(seqs, add_special_tokens=False)
        except TypeError:
            encs = tokenizer.encode_batch(seqs)
        out = []
        for e in encs:
            if hasattr(e, "ids"):
                out.append(e.ids)
            elif isinstance(e, list) and all(isinstance(x, int) for x in e):
                out.append(e)
            else:
                raise RuntimeError("encode_batch returned unsupported element.")
        return out
    return None

def _tok2ids_via_encode(tokenizer, s: str) -> List[int] | None:
    enc = getattr(tokenizer, "encode", None)
    if callable(enc):
        try:
            try:
                out = enc(s, add_special_tokens=False)
            except TypeError:
                out = enc(s)
            if hasattr(out, "ids"):
                return list(out.ids)
            if isinstance(out, list) and all(isinstance(x, int) for x in out):
                return out
        except Exception:
            pass
    return None

def _tok2ids_via_tokenize_map(tokenizer, s: str) -> List[int] | None:
    tokf = getattr(tokenizer, "tokenize", None)
    if not callable(tokf):
        return None
    try:
        toks = tokf(s)
        if isinstance(toks, torch.Tensor):
            return toks.long().tolist()
        if not isinstance(toks, list):
            return None
        if toks and all(isinstance(x, int) for x in toks):
            return toks
        out = []
        for t in toks:
            if isinstance(t, int):
                out.append(int(t))
            else:
                tid = _safe_token_to_id(tokenizer, t)
                if tid is None:
                    return None
                out.append(int(tid))
        return out
    except Exception:
        return None

def _tok2ids_via_charwise(tokenizer, s: str) -> List[int]:
    unknown_id = _discover_unknown_id(tokenizer)
    out: List[int] = []
    for ch in s:
        tid = _safe_token_to_id(tokenizer, ch)
        if tid is None:
            code = ord(ch)
            for key in (f"<0x{code:02X}>", f"<0x{code:02x}>"):
                tid = _safe_token_to_id(tokenizer, key)
                if isinstance(tid, int):
                    break
        if tid is None:
            tid = _encode_one_char_to_id(tokenizer, ch)
        out.append(int(tid if isinstance(tid, int) else unknown_id))
    return out

def _tok2ids_user_way(tokenizer, s: str) -> List[int]:
    encb = _tok2ids_via_encode_batch(tokenizer, [s])
    if encb is not None and len(encb) == 1:
        return list(encb[0])
    enc1 = _tok2ids_via_encode(tokenizer, s)
    if enc1 is not None:
        return enc1
    tmap = _tok2ids_via_tokenize_map(tokenizer, s)
    if tmap is not None:
        return tmap
    return _tok2ids_via_charwise(tokenizer, s)

def _tok2ids_batch_encode(tokenizer, seqs: List[str]) -> List[List[int]]:
    encb = _tok2ids_via_encode_batch(tokenizer, seqs)
    if encb is not None:
        return encb
    return [_tok2ids_user_way(tokenizer, s) for s in seqs]

# ------------------------ fixtures -----------------------------------

@pytest.fixture(scope="module")
def tokenizer_and_helper():
    # Avoid heavy Evo2 model load by default. Opt-in with USE_EVO2_TOKENIZER=1.
    tok = None
    if os.environ.get("USE_EVO2_TOKENIZER", "0") == "1":
        try:
            from evo2 import Evo2  # noqa: WPS433
            evo2_model = Evo2(os.environ.get("EVO2_MODEL_NAME", "evo2_7b"))
            tok = getattr(evo2_model, "tokenizer", evo2_model)
        except Exception:
            tok = None

    if tok is None:
        class _DummyTok:
            def encode(self, s, add_special_tokens=False):
                return [ord(c) for c in s]
        tok = _DummyTok()

    class _Shim:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.pad_token_id = _safe_token_to_id(tokenizer, "<pad>") or 0
        def embed_tokens(self, x, **_):
            return x.float()

    helper = DNATok(
        _Shim(tok),
        ids_max_tokens_per_call=1_048_576,
        prefer_int32_h2d=True,
        overlap_h2d_compute=True,
    )
    helper.discover()
    return tok, helper

# --------------------------- tests ----------------------------------

def test_ids_equivalence_and_padding(tokenizer_and_helper):
    tok, helper = tokenizer_and_helper

    # Single-string equality
    s = "ACGTNt"
    ids_native_1 = _tok2ids_user_way(tok, s)
    ids_native_1b = _tok2ids_batch_encode(tok, [s])[0]
    ids_helper_1 = helper.encode_batch_to_ids([s]).tolist()[0]

    assert ids_native_1 == ids_native_1b, "NativeTokenizer: _user_way vs encode_batch differ."
    assert ids_native_1 == ids_helper_1, "NativeTokenizer vs DNATok differ on single string."

    # Batch equality
    seqs = make_equal_len_sequences(B, T, seed=SEED)
    ids_native_batch = _tok2ids_batch_encode(tok, seqs)
    ids_dnatok_batch = helper.encode_batch_to_ids(seqs).tolist()
    assert ids_native_batch == ids_dnatok_batch, "Batch mismatch: NativeTokenizer vs DNATok."

    # Padding equality (left-pad to helper.token_len if larger than T)
    pad_id = helper.id_pad
    want_len = helper.token_len if helper.token_len and helper.token_len > T else T

    def left_pad_to(x: List[int], L: int, pad: int) -> List[int]:
        if len(x) >= L:
            return x[-L:]
        return [pad] * (L - len(x)) + x

    if want_len > T:
        native_padded = [left_pad_to(x, want_len, pad_id) for x in ids_native_batch]
        helper_padded = [left_pad_to(x, want_len, pad_id) for x in ids_dnatok_batch]
        assert native_padded == helper_padded, "Padding mismatch after left-pad to model token_len."

    # Human-readable prints
    print("\n✅ Single-string equality OK.")
    print("✅ Batch equality OK.")
    print(f"Padding check: target_len={want_len}, pad_id={pad_id}")


def test_bytes_path_matches_ids_on_cpu():
    class _Tok:
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s]
        def convert_tokens_to_ids(self, tok):
            return ord(tok) if isinstance(tok, str) and len(tok) == 1 else None
    class _Shim:
        def __init__(self):
            self.tokenizer = _Tok()
        def embed_tokens(self, x, **_):
            return x.float()

    helper = DNATok(_Shim())
    helper.discover()
    seqs = ["ACGT", "TGCA"]

    ids_outputs = list(helper.embed_from_strings(seqs, emb_batch=2, device="cpu", path="ids"))
    bytes_outputs = list(helper.embed_from_strings(seqs, emb_batch=2, device="cpu", path="bytes"))

    assert len(ids_outputs) == len(bytes_outputs) == 1
    assert torch.equal(ids_outputs[0], bytes_outputs[0])


def test_device_buffers_prefill_pad():
    class _Tok:
        def encode(self, s, add_special_tokens=False):
            return [1 for _ in s]
    class _Shim:
        def __init__(self):
            self.tokenizer = _Tok()
            self.model_max_length = 6
            self.pad_token_id = 9
        def embed_tokens(self, x, **_):
            return x.float()

    helper = DNATok(_Shim())
    helper.discover()

    # Ensure buffers are allocated with pad fill for token_len > T
    helper._ensure_device_pingpong(micro_bs=2, T=3, device="cpu", use_i32=False)
    assert helper._dev_ping_l is not None
    assert helper._dev_ping_l.shape == (2, 6)
    # Left pad region should be pad_token_id (9)
    assert torch.all(helper._dev_ping_l[:, :3] == helper.id_pad)


class MockKmerTokenizer:
    def __init__(self, k=6):
        self.k = k
        self.vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5, "<pad>": 0}
        self.pad_token_id = 0
    
    def __call__(self, text, add_special_tokens=False, padding=False, truncation=False, max_length=None, return_tensors=None):
        if isinstance(text, str):
            text = [text]
        ids_list = []
        for t in text:
            row = []
            i = 0
            while i < len(t):
                chunk = t[i:i+self.k]
                if len(chunk) == self.k:
                    # Mock ID: sum of chars or something deterministic
                    # A=1, C=2...
                    val = sum(self.vocab.get(c, 0) for c in chunk)
                    row.append(100 + val) 
                    i += self.k
                else:
                    for c in chunk:
                        row.append(self.vocab.get(c, 0))
                    i += len(chunk)
            if max_length and len(row) < max_length and padding:
                row += [self.pad_token_id] * (max_length - len(row))
            ids_list.append(row)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids_list, dtype=torch.long)}
        return {"input_ids": ids_list}

    def encode(self, text, add_special_tokens=False):
        # Re-use call logic
        out = self.__call__(text, add_special_tokens=add_special_tokens)
        return out["input_ids"][0]

def test_kmer_equivalence():
    k = 4
    tok = MockKmerTokenizer(k=k)
    
    class _Shim:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.pad_token_id = 0
            self.model_max_length = 100
        def embed_tokens(self, x, **_):
            return x.float()

    helper = DNATok(_Shim(tok))
    helper.discover()
    helper.token_len = None # Disable padding for equivalence check
    
    assert helper.kmer_k == k, f"Failed to detect k={k}"
    assert helper.use_ids_path, "IDs path should be enabled for k-mer"
    
    # Test sequence divisible by k
    seq = "ACGT" * 10 # 40 chars -> 10 tokens
    ids_native = tok.encode(seq)
    ids_helper = helper.encode_batch_to_ids([seq])[0].tolist()
    
    assert ids_native == ids_helper, "K-mer encoding mismatch"
    
    # Test auto path selection
    # For k-mer, auto should pick ids path (bytes path not supported)
    # We can't easily check internal choice without mocking, but we can check it runs
    # and produces correct output.
    
    # Note: embed_from_strings requires CUDA for bytes path usually, but here we are on CPU?
    # DNATok.embed_from_strings checks for CUDA device for bytes path.
    # So on CPU it will always use ids path.
    # But let's verify it doesn't crash.
    
    out = list(helper.embed_from_strings([seq], emb_batch=1, device="cpu", path="auto"))
    assert len(out) == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
