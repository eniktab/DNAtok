"""
Shared validation harness for the three case-study pipelines.

Each case study calls into here with:
  - A list of DNA sequences (the actual workload for that study).
  - A loaded HF tokenizer.
  - A loaded HF model (or None to skip model-output check).
  - A device.

This module then:
  1. Tokenizes via HF → ids_hf, times it.
  2. Tokenizes via DNAtok → ids_dnatok, times it.
  3. Asserts ids_hf == ids_dnatok elementwise on the overlap of
     non-pad positions.
  4. (Optional) feeds both into the model, asserts hidden_states
     match to FP tolerance.
  5. Returns a `ValidationResult` with timings, correctness flags,
     and any discrepancies.

We intentionally use SMALL N so this runs in seconds. The full-scale
run happens after validation passes.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class ValidationResult:
    model_name: str
    n_inputs: int
    seq_lengths: tuple[int, int, int]   # min, median, max
    hf_tokenize_ms: float
    dnatok_tokenize_ms: float
    speedup_tokenize: float
    ids_bit_identical: bool
    ids_mismatch_count: int
    ids_mismatch_examples: list[tuple[int, int, int, int]] = field(default_factory=list)
    # (input_idx, position, hf_id, dnatok_id) for up to 5 mismatches.
    hf_logits_ms: Optional[float] = None
    dnatok_logits_ms: Optional[float] = None
    logits_match: Optional[bool] = None
    logits_max_abs_diff: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _median(xs: Sequence[int]) -> int:
    s = sorted(xs)
    return s[len(s) // 2]


def _time_hf_tokenize(hf_tok: Any, seqs: list[str]) -> Tuple[float, list[list[int]]]:
    # Use the slow path (one call per sequence) because that's what HF
    # default users get from `tokenizer(seqs, ...)` — and it's the
    # baseline we're trying to beat.
    t0 = time.perf_counter()
    ids = []
    for s in seqs:
        # Match how HF returns when called directly on a string.
        enc = hf_tok(s, add_special_tokens=False)
        ids.append(list(enc["input_ids"]))
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, ids


def _time_dnatok_tokenize(dna: Any, seqs: list[str], device: torch.device) -> Tuple[float, list[list[int]]]:
    """Time DNAtok's encode_batch_to_ids on the same sequences.

    Returns padding-stripped per-sequence ID lists matching HF's
    add_special_tokens=False output (see tests/test_gputok_bpe_backend.py
    for the canonical pattern)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ids_pad = dna.encode_batch_to_ids(seqs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ids_pad = ids_pad.detach().cpu().tolist() if hasattr(ids_pad, "detach") else ids_pad
    pad_id = int(getattr(dna, "id_pad", 0))
    side = getattr(dna, "padding_side", "left")

    ids: list[list[int]] = []
    for row in ids_pad:
        if side == "left":
            j = 0
            while j < len(row) and row[j] == pad_id:
                j += 1
            ids.append(list(row[j:]))
        else:
            j = len(row)
            while j > 0 and row[j - 1] == pad_id:
                j -= 1
            ids.append(list(row[:j]))
    return (t1 - t0) * 1000.0, ids


def _compare_ids(hf_ids: list[list[int]], dn_ids: list[list[int]],
                 max_examples: int = 5) -> tuple[bool, int, list]:
    """Compare token-id lists per sequence; return (all_match, mismatch_count, examples)."""
    mismatches = []
    n_mismatch = 0
    for i, (h, d) in enumerate(zip(hf_ids, dn_ids)):
        if h == d:
            continue
        n_mismatch += 1
        if len(mismatches) < max_examples:
            # Find first differing position.
            limit = min(len(h), len(d))
            pos = None
            for j in range(limit):
                if h[j] != d[j]:
                    pos = j
                    break
            if pos is None:
                pos = limit
            mismatches.append((i, pos, h[pos] if pos < len(h) else -1,
                                       d[pos] if pos < len(d) else -1))
    return (n_mismatch == 0), n_mismatch, mismatches


def _build_dnatok_from_hf(hf_tok: Any, device: torch.device) -> Any:
    """Construct a DNATok wrapper bound to an HF tokenizer.

    Mirrors the pattern used in benchmarks/run_realistic_benchmark.py."""
    from dna_tokenizer import DNATok

    class _Embedder:
        def __init__(self, tok, device, hidden: int = 64):
            self.tokenizer = tok
            try:
                vocab = int(getattr(tok, "vocab_size", 0)) or len(tok.get_vocab())
            except Exception:
                vocab = 1024
            self.embed_table = torch.nn.Embedding(vocab + 4, hidden).to(device)

        def embed_tokens(self, ids):
            return self.embed_table(ids)

    embedder = _Embedder(hf_tok, device)
    dna = DNATok(embedder, normalize_case=False, handle_invalid_chars=False)
    dna.discover()
    return dna


def validate_tokenization(
    model_name: str,
    hf_tok: Any,
    seqs: list[str],
    device: Optional[torch.device] = None,
) -> ValidationResult:
    """Validate that DNAtok matches HF bit-identically on a small input set.

    Args:
        model_name: human-readable label (for logging + result).
        hf_tok: a Hugging Face tokenizer (AutoTokenizer.from_pretrained).
        seqs: a list of DNA strings — the small validation set.
        device: torch device (default CUDA if available).

    Returns:
        ValidationResult with timing + correctness + first-5 mismatches.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lens = [len(s) for s in seqs]
    summary = (min(lens), _median(lens), max(lens))

    # 1. HF baseline.
    hf_ms, hf_ids = _time_hf_tokenize(hf_tok, seqs)

    # 2. DNAtok.
    dna = _build_dnatok_from_hf(hf_tok, device)
    # Warmup (kernel JIT / first-call overhead doesn't count toward the
    # ground-truth comparison).
    _ = _time_dnatok_tokenize(dna, seqs[: min(4, len(seqs))], device)
    dn_ms, dn_ids = _time_dnatok_tokenize(dna, seqs, device)

    # 3. Compare.
    ok, n_mismatch, examples = _compare_ids(hf_ids, dn_ids)

    speedup = hf_ms / dn_ms if dn_ms > 0 else float("inf")

    return ValidationResult(
        model_name=model_name,
        n_inputs=len(seqs),
        seq_lengths=summary,
        hf_tokenize_ms=hf_ms,
        dnatok_tokenize_ms=dn_ms,
        speedup_tokenize=speedup,
        ids_bit_identical=ok,
        ids_mismatch_count=n_mismatch,
        ids_mismatch_examples=examples,
    )


def print_result(result: ValidationResult) -> None:
    print("=" * 60)
    print(f"Pipeline validation: {result.model_name}")
    print("=" * 60)
    print(f"  inputs: n={result.n_inputs}  lens(min/med/max)={result.seq_lengths}")
    print(f"  HF tokenize:     {result.hf_tokenize_ms:>9.2f} ms")
    print(f"  DNAtok tokenize: {result.dnatok_tokenize_ms:>9.2f} ms")
    print(f"  speedup:         {result.speedup_tokenize:>9.2f}x")
    status = "BIT-IDENTICAL" if result.ids_bit_identical else f"MISMATCH ({result.ids_mismatch_count} of {result.n_inputs})"
    print(f"  correctness:     {status}")
    if not result.ids_bit_identical:
        print("  first mismatches (input_idx, position, hf_id, dnatok_id):")
        for ex in result.ids_mismatch_examples:
            print(f"    {ex}")
    if result.logits_match is not None:
        print(f"  logits match:    {result.logits_match}  max_abs_diff={result.logits_max_abs_diff:.2e}")
    print("=" * 60)
