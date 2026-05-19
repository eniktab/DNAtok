"""Correctness gate for the GPUTOK BPE backend.

For every BPE tokenizer in the model registry, assert that
GPUTokBPEBackend.encode_batch produces id sequences that are
bit-identical to HF native (after stripping HF padding).

If this gate fires the backend MUST NOT be used to report any speedup
number, period. Speed is a property; correctness is a precondition.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import dnatok_compat  # noqa: F401
from benchmarks.model_registry import MODEL_SPECS, resolve_model_path
from benchmarks.tokenizer_adapters import load_hf_tokenizer

# Skip the whole module cleanly if CUDA or GPUTOK isn't available here.
REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                                    reason="CUDA required")


def _gputok_available() -> bool:
    try:
        from src.gputok_bpe_backend import _find_gpu_tokenizer_dir
        return _find_gpu_tokenizer_dir() is not None
    except Exception:
        return False


REQUIRES_GPUTOK = pytest.mark.skipif(not _gputok_available(),
                                      reason="GPUTOK source not found")


BPE_MODELS = ["DNABERT2_117M", "GENA_LM_BERT_t2t", "METAGENE_1"]
# Engines: "gputok" = third-party BlockBPE baseline; "dnatok" = our
# entry-pool bucket-scheduling kernel.
ENGINES = ["gputok", "dnatok"]


def _build_backend(model_name: str, engine: str = "gputok"):
    spec = next((s for s in MODEL_SPECS if s.name == model_name), None)
    if spec is None:
        pytest.skip(f"{model_name} not in registry")
    p = resolve_model_path(spec)
    if p is None:
        pytest.skip(f"{model_name} cache missing")
    tok = load_hf_tokenizer(str(p))
    from src.gputok_bpe_backend import GPUTokBPEBackend
    try:
        backend = GPUTokBPEBackend(tok, engine=engine)
    except Exception as e:
        if engine == "dnatok":
            pytest.skip(f"{engine} kernel not buildable here: {e}")
        raise
    return backend, tok


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_supports_model(model_name, engine):
    """Backend must accept every registered BPE tokenizer without raising."""
    backend, _ = _build_backend(model_name, engine=engine)
    assert backend.vocab_size > 0
    assert backend.pad_id >= 0


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_bit_identical_to_hf(model_name, engine):
    """STRICT CORRECTNESS GATE.

    For 256 random DNA sequences spanning a wide length distribution, the
    backend output (after stripping HF pad) must equal what HF's tokenizer
    produces sequence-by-sequence with ``add_special_tokens=False``.

    Bumped from 64 to 256 sequences and the length distribution to cover
    short/medium/long inputs so that any rank-order edge case in the
    merge table or any normalizer interaction is more likely to be hit.
    """
    backend, hf_tok = _build_backend(model_name, engine=engine)

    random.seed(0)
    lens = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    seqs = [
        "".join(random.choices("ACGT", k=random.choice(lens)))
        for _ in range(256)
    ]

    # HF native, per-sequence (no padding) — the unambiguous reference.
    hf_per_seq = [hf_tok(s, add_special_tokens=False)["input_ids"] for s in seqs]

    # Backend output (padded batch).
    ids_dev, mask_dev = backend.encode_batch(seqs, device="cuda")
    assert ids_dev.shape == mask_dev.shape
    assert ids_dev.dtype == torch.long
    ids = ids_dev.cpu().tolist()
    mask = mask_dev.cpu().tolist()

    mismatches = 0
    first_diff = None
    for i in range(len(seqs)):
        valid = [ids[i][j] for j in range(len(mask[i])) if mask[i][j]]
        if valid != list(hf_per_seq[i]):
            mismatches += 1
            if first_diff is None:
                first_diff = (i, seqs[i], hf_per_seq[i], valid)
    if mismatches:
        i, s, hf, bk = first_diff
        raise AssertionError(
            f"{model_name}: {mismatches}/{len(seqs)} sequences differ from HF native.\n"
            f"  seq[{i}] ({s[:32]}...): hf({len(hf)})={hf[:8]}...  bk({len(bk)})={bk[:8]}..."
        )


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_with_n_bases(model_name, engine):
    """N bases exercise the LUT slot for code 'N' AND, for GENA-LM, the
    normalizer's ``N{10,} → -`` rule. The backend must still match HF
    exactly on inputs containing N runs of varying length.
    """
    backend, hf_tok = _build_backend(model_name, engine=engine)
    random.seed(11)
    cases = [
        "ACGTNACGT",                 # single N
        "ACGT" + "N" * 5 + "ACGT",   # 5 Ns (below GENA-LM's compression threshold)
        "ACGT" + "N" * 20 + "ACGT",  # 20 Ns (would trigger GENA-LM compression)
        "N" * 64,                    # all Ns
        "ACGTN" * 32,                # interleaved
    ]
    hf_per_seq = [hf_tok(s, add_special_tokens=False)["input_ids"] for s in cases]
    ids_dev, mask_dev = backend.encode_batch(cases, device="cuda")
    ids = ids_dev.cpu().tolist()
    mask = mask_dev.cpu().tolist()
    for i, s in enumerate(cases):
        valid = [ids[i][j] for j in range(len(mask[i])) if mask[i][j]]
        assert valid == list(hf_per_seq[i]), (
            f"{model_name}: N-content seq differs from HF native:\n"
            f"  seq: {s[:32]}\n  hf={hf_per_seq[i][:8]}...  bk={valid[:8]}..."
        )


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_mixed_case(model_name, engine):
    """If HF's pipeline lowercases (or preserves case), the backend must
    behave the same. This catches a regression where the backend bypasses
    HF's case-handling and ships raw lowercase bytes to GPUTOK."""
    backend, hf_tok = _build_backend(model_name, engine=engine)
    cases = ["acgtACGT", "AcGtAcGt", "actgnactgn"]
    hf_per_seq = [hf_tok(s, add_special_tokens=False)["input_ids"] for s in cases]
    ids_dev, mask_dev = backend.encode_batch(cases, device="cuda")
    ids = ids_dev.cpu().tolist()
    mask = mask_dev.cpu().tolist()
    for i, s in enumerate(cases):
        valid = [ids[i][j] for j in range(len(mask[i])) if mask[i][j]]
        assert valid == list(hf_per_seq[i]), (
            f"{model_name}: mixed-case seq {s!r} diverges:\n"
            f"  hf={hf_per_seq[i]}  bk={valid}"
        )


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_long_sequences(model_name, engine):
    """4 kbp and 8 kbp inputs — the depth regime where merge-iter count
    is largest. Asserts no max-iter truncation."""
    backend, hf_tok = _build_backend(model_name, engine=engine)
    random.seed(99)
    cases = [
        "".join(random.choices("ACGT", k=4000)),
        "".join(random.choices("ACGT", k=8000)),
    ]
    hf_per_seq = [hf_tok(s, add_special_tokens=False)["input_ids"] for s in cases]
    ids_dev, mask_dev = backend.encode_batch(cases, device="cuda")
    ids = ids_dev.cpu().tolist()
    mask = mask_dev.cpu().tolist()
    for i, s in enumerate(cases):
        valid = [ids[i][j] for j in range(len(mask[i])) if mask[i][j]]
        if valid != list(hf_per_seq[i]):
            n_diff = sum(1 for a, b in zip(valid, hf_per_seq[i]) if a != b)
            raise AssertionError(
                f"{model_name}: long seq {len(s)} bp diverges — {n_diff} token diffs, "
                f"backend produced {len(valid)} tokens vs HF {len(hf_per_seq[i])}"
            )


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_padding_matches_hf_pad_id(model_name, engine):
    """Pad positions in the backend output must use the HF pad_token_id."""
    backend, hf_tok = _build_backend(model_name, engine=engine)
    short = "ACGT"
    longer = "ACGTACGTACGTACGTACGTACGTACGTACGT"
    seqs = [short, longer]
    ids_dev, mask_dev = backend.encode_batch(seqs, device="cuda")
    ids = ids_dev.cpu().tolist()
    mask = mask_dev.cpu().tolist()
    # Every position where mask==0 must equal HF's pad_token_id.
    for row, m_row in zip(ids, mask):
        for tid, mv in zip(row, m_row):
            if not mv:
                assert tid == hf_tok.pad_token_id, (
                    f"{model_name}: padded position not hf.pad_token_id "
                    f"({tid} vs {hf_tok.pad_token_id})"
                )


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_stable_across_calls(model_name, engine):
    """Two identical calls must return identical ids (no nondeterminism)."""
    backend, _ = _build_backend(model_name, engine=engine)
    random.seed(7)
    seqs = ["".join(random.choices("ACGT", k=128)) for _ in range(8)]
    a = backend.encode_batch_ids_only(seqs).cpu()
    b = backend.encode_batch_ids_only(seqs).cpu()
    assert torch.equal(a, b), f"{model_name}: GPUTOK output is nondeterministic"


# ============================================================================
# Edge-case correctness gates
# ============================================================================
# These exercise corners the random-sequence gate doesn't reliably hit:
# very short sequences, B=1, all-identical bases, mixed empty/nonempty.

@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_batch_of_one(model_name, engine):
    """B=1 path — must not regress when the batch dimension is 1."""
    backend, hf_tok = _build_backend(model_name, engine=engine)
    seq = "ACGTACGTACGTACGT"
    hf_ids = list(hf_tok(seq, add_special_tokens=False)["input_ids"])
    ids_dev, mask_dev = backend.encode_batch([seq], device="cuda")
    assert ids_dev.shape[0] == 1
    valid = [int(ids_dev[0, j]) for j in range(ids_dev.shape[1]) if int(mask_dev[0, j])]
    assert valid == hf_ids, (
        f"{model_name} (B=1): valid={valid}  hf={hf_ids}"
    )


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_single_char(model_name, engine):
    """A 1-base sequence cannot merge anything; output must equal the
    single-byte HF tokenization."""
    backend, hf_tok = _build_backend(model_name, engine=engine)
    for c in ("A", "C", "G", "T", "N"):
        hf_ids = list(hf_tok(c, add_special_tokens=False)["input_ids"])
        ids_dev, mask_dev = backend.encode_batch([c], device="cuda")
        valid = [int(ids_dev[0, j]) for j in range(ids_dev.shape[1]) if int(mask_dev[0, j])]
        assert valid == hf_ids, (
            f"{model_name} single-char {c!r}: valid={valid}  hf={hf_ids}"
        )


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_homopolymer_runs(model_name, engine):
    """Long runs of a single base exercise the rank-batched selection's
    every-other rule (consecutive (A,A) pairs all share the lowest rank
    but only every other position can fire in one iteration)."""
    backend, hf_tok = _build_backend(model_name, engine=engine)
    cases = ["A" * 32, "C" * 32, "G" * 32, "T" * 32, "A" * 256, "G" * 1024]
    hf_per = [list(hf_tok(s, add_special_tokens=False)["input_ids"]) for s in cases]
    ids_dev, mask_dev = backend.encode_batch(cases, device="cuda")
    ids = ids_dev.cpu().tolist()
    mask = mask_dev.cpu().tolist()
    for i, s in enumerate(cases):
        valid = [ids[i][j] for j in range(len(mask[i])) if mask[i][j]]
        assert valid == hf_per[i], (
            f"{model_name} homopolymer {s[0]}×{len(s)}: "
            f"hf({len(hf_per[i])})={hf_per[i][:6]} valid({len(valid)})={valid[:6]}"
        )


# ============================================================================
# Normalizer compiler — unit tests
# ============================================================================
# These don't go through the GPU at all; they verify that our Python
# implementation of HF's normalizer pipeline reproduces HF semantics for
# the types we support. Catches drift if HF changes a definition or if
# someone adds a new type without testing.

def test_normalizer_strip():
    from src.gputok_bpe_backend import _compile_normalizer_step
    fn = _compile_normalizer_step({"type": "Strip", "strip_left": True, "strip_right": True})
    assert fn is not None
    assert fn("  ACGT  ") == "ACGT"
    assert fn("ACGT") == "ACGT"
    fn_l = _compile_normalizer_step({"type": "Strip", "strip_left": True, "strip_right": False})
    assert fn_l("  ACGT  ") == "ACGT  "


def test_normalizer_replace_string():
    from src.gputok_bpe_backend import _compile_normalizer_step
    fn = _compile_normalizer_step({
        "type": "Replace",
        "pattern": {"String": "N"},
        "content": "-",
    })
    assert fn is not None
    assert fn("ACGTNNACGT") == "ACGT--ACGT"


def test_normalizer_replace_regex():
    from src.gputok_bpe_backend import _compile_normalizer_step
    fn = _compile_normalizer_step({
        "type": "Replace",
        "pattern": {"Regex": "N{10,}"},
        "content": "-",
    })
    assert fn is not None
    # Below threshold: unchanged.
    assert fn("ACGT" + "N" * 9 + "ACGT") == "ACGT" + "N" * 9 + "ACGT"
    # At threshold: replaced.
    assert fn("ACGT" + "N" * 10 + "ACGT") == "ACGT-ACGT"


def test_normalizer_prepend():
    from src.gputok_bpe_backend import _compile_normalizer_step
    fn = _compile_normalizer_step({"type": "Prepend", "prepend": "_"})
    assert fn is not None
    assert fn("ACGT") == "_ACGT"
    assert fn("") == "_"


def test_normalizer_lowercase():
    from src.gputok_bpe_backend import _compile_normalizer_step
    fn = _compile_normalizer_step({"type": "Lowercase"})
    assert fn is not None
    assert fn("ACGTACGT") == "acgtacgt"


def test_normalizer_nfc_identity_for_ascii():
    from src.gputok_bpe_backend import _compile_normalizer_step
    for typ in ("NFC", "NFKC", "NFD", "NFKD", "StripAccents"):
        fn = _compile_normalizer_step({"type": typ})
        assert fn is not None, f"{typ} should be accepted as identity"
        assert fn("ACGT") == "ACGT"


def test_normalizer_sequence_strip_then_replace():
    from src.gputok_bpe_backend import _compile_normalizer_step
    fn = _compile_normalizer_step({
        "type": "Sequence",
        "normalizers": [
            {"type": "Strip", "strip_left": True, "strip_right": True},
            {"type": "Replace", "pattern": {"Regex": "N{10,}"}, "content": "-"},
        ],
    })
    assert fn is not None
    # Strip applies first, then the regex compresses Ns.
    assert fn("   " + "N" * 12 + "ACGT  ") == "-ACGT"


def test_normalizer_unsupported_type_rejected():
    """Genuine unknown types must return None — caller decides to reject."""
    from src.gputok_bpe_backend import _compile_normalizer_step
    assert _compile_normalizer_step({"type": "BertNormalizer"}) is None
    assert _compile_normalizer_step({"type": "Nmt"}) is None
    assert _compile_normalizer_step({"type": "Precompiled"}) is None
    assert _compile_normalizer_step({"type": "SomeNewThingHFAdds"}) is None


def test_normalizer_sequence_with_unsupported_inner_rejected():
    """If a Sequence contains an unsupported step, the whole sequence is
    rejected (None) — we cannot partially apply a normaliser pipeline."""
    from src.gputok_bpe_backend import _compile_normalizer_step
    fn = _compile_normalizer_step({
        "type": "Sequence",
        "normalizers": [
            {"type": "Strip", "strip_left": True, "strip_right": True},
            {"type": "BertNormalizer"},  # unsupported
        ],
    })
    assert fn is None


def test_supported_tokenizer_rejects_unsupported_normalizer():
    """The high-level _supported_tokenizer probe must reject configs with
    normalisers we cannot reproduce."""
    from src.gputok_bpe_backend import _supported_tokenizer
    cfg = {
        "model": {"type": "BPE", "merges": ["A A"]},
        "pre_tokenizer": {"type": "Whitespace"},
        "normalizer": {"type": "BertNormalizer"},
    }
    ok, reason = _supported_tokenizer(cfg)
    assert not ok
    assert "normalizer" in reason.lower()


@REQUIRES_CUDA
@REQUIRES_GPUTOK
def test_dnatok_kernel_view_aliasing_documented():
    """Without clone=True, the kernel returns views into a persistent
    workspace; the next call OVERWRITES them. With clone=True the
    returned tensors are independent. Verify both behaviours explicitly
    so the documented contract is testable.
    """
    spec = next(s for s in MODEL_SPECS if s.name == "DNABERT2_117M")
    if resolve_model_path(spec) is None:
        pytest.skip("DNABERT-2 cache missing")
    # Build through the backend so the merges file is materialised, then
    # access the underlying kernel directly.
    from src.gputok_bpe_backend import GPUTokBPEBackend
    tok = load_hf_tokenizer(str(resolve_model_path(spec)))
    backend = GPUTokBPEBackend(tok, engine="dnatok")
    kernel = backend._dnatok

    seqs_a = ["ACGTACGT"] * 4
    seqs_b = ["TTTTGGGG"] * 4

    # Without clone: second call overwrites first's view.
    ids1, lens1 = kernel.tokenize_batch(seqs_a)
    snap1_aliased = ids1.clone()  # snapshot for comparison
    ids2, lens2 = kernel.tokenize_batch(seqs_b)
    # ids1 now reflects seqs_b results because they share storage.
    # (Use slice-equal at column 0 only, since seqs_b has different content.)
    assert not torch.equal(ids1[:, :max(int(lens2.max()), 1)],
                            snap1_aliased[:, :max(int(lens2.max()), 1)]), (
        "view-aliasing contract broken: ids1 should track latest storage"
    )

    # With clone: second call does NOT overwrite cloned snapshot.
    ids3, lens3 = kernel.tokenize_batch(seqs_a, clone=True)
    snap3 = ids3.clone()
    ids4, lens4 = kernel.tokenize_batch(seqs_b, clone=True)
    assert torch.equal(ids3, snap3), (
        "clone=True did not produce an independent tensor"
    )


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_gputok_backend_mixed_lengths_in_batch(model_name, engine):
    """Mix of very-short, medium, and long inputs in one batch tests the
    padding semantics and (for dnatok) the routing split between GPU and
    HF fallback paths."""
    backend, hf_tok = _build_backend(model_name, engine=engine)
    random.seed(13)
    cases = [
        "AC",                                              # tiny
        "".join(random.choices("ACGT", k=64)),             # small
        "".join(random.choices("ACGT", k=256)),            # medium
        "".join(random.choices("ACGT", k=2048)),           # at-threshold
        "".join(random.choices("ACGT", k=4096)),           # routes to HF for both engines
    ]
    hf_per = [list(hf_tok(s, add_special_tokens=False)["input_ids"]) for s in cases]
    ids_dev, mask_dev = backend.encode_batch(cases, device="cuda")
    ids = ids_dev.cpu().tolist()
    mask = mask_dev.cpu().tolist()
    for i, s in enumerate(cases):
        valid = [ids[i][j] for j in range(len(mask[i])) if mask[i][j]]
        assert valid == hf_per[i], (
            f"{model_name} mixed-batch seq[{i}] (len={len(s)}): "
            f"hf({len(hf_per[i])}) vs valid({len(valid)})"
        )


# ============================================================================
# DNATok integration — high-level path through DNATok.encode_batch_to_ids
# ============================================================================
# When DNATok is initialised with a BPE tokenizer, discover() should build
# a bpe_backend and route encode_batch_to_ids / embed_from_strings through
# it. These tests verify that integration end-to-end.

@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_dnatok_discover_builds_bpe_backend(model_name):
    """For a supported BPE tokenizer, DNATok.discover() must produce a
    non-None bpe_backend even when the LUT/k-mer fast paths are disabled."""
    spec = next((s for s in MODEL_SPECS if s.name == model_name), None)
    if spec is None:
        pytest.skip(f"{model_name} not in registry")
    p = resolve_model_path(spec)
    if p is None:
        pytest.skip(f"{model_name} cache missing")
    tok = load_hf_tokenizer(str(p))

    class _Emb:
        def __init__(self, t):
            self.tokenizer = t
            self.embed_table = torch.nn.Embedding(len(t), 32).cuda()
        def embed_tokens(self, ids):
            return self.embed_table(ids)

    from src.dna_tokenizer import DNATok
    dna = DNATok(_Emb(tok), normalize_case=True, handle_invalid_chars=True)
    dna.discover()
    assert not dna.use_ids_path, "BPE tokenizer must NOT enable the LUT path"
    assert dna.bpe_backend is not None, (
        f"{model_name}: DNATok.discover did not build a BPE backend"
    )


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_dnatok_encode_batch_to_ids_via_bpe_backend(model_name):
    """DNATok.encode_batch_to_ids on a BPE tokenizer (with the backend in
    place) must produce ids that match HF native after stripping
    DNATok-side padding."""
    spec = next((s for s in MODEL_SPECS if s.name == model_name), None)
    if spec is None:
        pytest.skip(f"{model_name} not in registry")
    p = resolve_model_path(spec)
    if p is None:
        pytest.skip(f"{model_name} cache missing")
    tok = load_hf_tokenizer(str(p))

    class _Emb:
        def __init__(self, t):
            self.tokenizer = t
            self.embed_table = torch.nn.Embedding(len(t), 32).cuda()
        def embed_tokens(self, ids):
            return self.embed_table(ids)

    from src.dna_tokenizer import DNATok
    dna = DNATok(_Emb(tok), normalize_case=True, handle_invalid_chars=True)
    dna.discover()
    assert dna.bpe_backend is not None

    random.seed(21)
    seqs = ["".join(random.choices("ACGT", k=random.choice([16, 32, 128]))) for _ in range(8)]
    ids_pinned = dna.encode_batch_to_ids(seqs)
    assert ids_pinned.dtype == torch.long
    # Strip on the side DNATok chose to pad (default: left).
    pad_id = int(dna.id_pad)
    rows = ids_pinned.tolist()
    for i, s in enumerate(seqs):
        row = rows[i]
        if dna.padding_side == "left":
            j = 0
            while j < len(row) and row[j] == pad_id:
                j += 1
            valid = row[j:]
        else:
            j = len(row)
            while j > 0 and row[j - 1] == pad_id:
                j -= 1
            valid = row[:j]
        hf_ids = list(tok(s, add_special_tokens=False)["input_ids"])
        assert valid == hf_ids, (
            f"{model_name} seq[{i}]: hf={hf_ids[:8]}  valid={valid[:8]}"
        )


@REQUIRES_CUDA
@REQUIRES_GPUTOK
@pytest.mark.parametrize("model_name", BPE_MODELS)
def test_dnatok_embed_from_strings_uses_backend(model_name):
    """End-to-end embed: encode + embed must produce a finite tensor and
    the correct number of output rows for a BPE model."""
    spec = next((s for s in MODEL_SPECS if s.name == model_name), None)
    if spec is None:
        pytest.skip(f"{model_name} not in registry")
    p = resolve_model_path(spec)
    if p is None:
        pytest.skip(f"{model_name} cache missing")
    tok = load_hf_tokenizer(str(p))

    class _Emb:
        def __init__(self, t):
            self.tokenizer = t
            self.embed_table = torch.nn.Embedding(len(t), 32).cuda()
        def embed_tokens(self, ids):
            return self.embed_table(ids)

    from src.dna_tokenizer import DNATok
    dna = DNATok(_Emb(tok), normalize_case=True, handle_invalid_chars=True)
    dna.discover()
    seqs = ["ACGTACGT" * 8 for _ in range(4)]
    chunks = list(dna.embed_from_strings(seqs, emb_batch=2, device="cuda", path="ids"))
    total_rows = sum(c.shape[0] for c in chunks)
    assert total_rows == len(seqs), (
        f"{model_name}: embed_from_strings yielded {total_rows} rows, expected {len(seqs)}"
    )
    for c in chunks:
        assert c.device.type == "cuda"
        assert torch.isfinite(c).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
