"""Shared timing helpers for the bio pipelines.

Goal: report tokens/sec and bp/sec for both HF (single + multi-thread)
and DNAtok on the same real biological inputs. No proxy, no hidden
fallback — every measurement here corresponds to a real tokenizer
call on real data.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, List, Tuple


@dataclass
class TokenizerTiming:
    """One tokenizer × one workload measurement."""
    label: str                          # e.g. "hf_t1", "hf_t8", "dnatok"
    n_sequences: int
    total_bp: int
    total_tokens: int                   # cumulative tokens produced
    wall_seconds: float
    bp_per_second: float                # input throughput
    tokens_per_second: float            # OUTPUT throughput — what feeds the GPU
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _set_rayon(n: int) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(n)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if n > 1 else "false"


def time_hf_batched(
    hf_tok: Any,
    seqs: List[str],
    *,
    chunk: int,
    n_threads: int,
    max_length: int,
) -> TokenizerTiming:
    """Time the HF Rust tokenizer in batched calls (engages Rayon)."""
    _set_rayon(n_threads)

    # Warmup: one batch through the tokenizer so Rayon initializes.
    if seqs:
        _ = hf_tok(list(seqs[: min(chunk, len(seqs))]),
                    add_special_tokens=False, padding="longest",
                    truncation=True, max_length=max_length,
                    return_tensors=None)

    total_tokens = 0
    t0 = time.perf_counter()
    for i in range(0, len(seqs), chunk):
        batch = seqs[i : i + chunk]
        enc = hf_tok(list(batch), add_special_tokens=False,
                      padding=False, truncation=False,
                      return_tensors=None)
        for ids in enc["input_ids"]:
            total_tokens += len(ids)
    elapsed = time.perf_counter() - t0

    total_bp = sum(len(s) for s in seqs)
    return TokenizerTiming(
        label=f"hf_t{n_threads}",
        n_sequences=len(seqs),
        total_bp=total_bp,
        total_tokens=total_tokens,
        wall_seconds=elapsed,
        bp_per_second=total_bp / elapsed if elapsed > 0 else 0.0,
        tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0.0,
    )


def time_dnatok(
    dnatok: Any,
    seqs: List[str],
    *,
    chunk: int,
    device: Any,
) -> Tuple[TokenizerTiming, List[List[int]]]:
    """Time DNAtok on the same workload. Returns timing + padded IDs."""
    import torch  # local import

    if seqs:
        _ = dnatok.encode_batch_to_ids(seqs[: min(chunk, len(seqs))])
        if device.type == "cuda":
            torch.cuda.synchronize()

    pad_id = int(dnatok.id_pad)
    side = dnatok.padding_side
    total_tokens = 0
    valid_ids: List[List[int]] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(seqs), chunk):
        batch = seqs[i : i + chunk]
        ids = dnatok.encode_batch_to_ids(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        rows = ids.cpu().tolist()
        for row in rows:
            if side == "left":
                j = 0
                while j < len(row) and row[j] == pad_id:
                    j += 1
                v = row[j:]
            else:
                j = len(row)
                while j > 0 and row[j - 1] == pad_id:
                    j -= 1
                v = row[:j]
            valid_ids.append(v)
            total_tokens += len(v)
    elapsed = time.perf_counter() - t0

    total_bp = sum(len(s) for s in seqs)
    return (
        TokenizerTiming(
            label="dnatok",
            n_sequences=len(seqs),
            total_bp=total_bp,
            total_tokens=total_tokens,
            wall_seconds=elapsed,
            bp_per_second=total_bp / elapsed if elapsed > 0 else 0.0,
            tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0.0,
        ),
        valid_ids,
    )


def time_dnatok_native(
    dnatok: Any,
    seqs: List[str],
    *,
    chunk: int,
    device: Any,
) -> Tuple[TokenizerTiming, List[List[int]]]:
    """V1-style DNAtok timing: persistent pinned staging buffer +
    int32 H2D path. Returns timing + padded IDs (decoded back to int64
    valid tokens for the bit-identical check).

    Differences from time_dnatok:
      * uses encode_batch_to_ids_staging (reuses pinned buffer)
      * sets prefer_int32_h2d=True so H2D moves int32 (half the bytes),
        then on the caller side promote to int64 for the embedding lookup
        (V1 §3.2 — "nearly free promotion to int64").
    """
    import torch  # local import

    # Set int32 H2D preference on DNATok instance for staging output
    prev_pref = getattr(dnatok, "prefer_int32_h2d", False)
    dnatok.prefer_int32_h2d = True
    try:
        # Warm-up: prime staging buffer + any kernels
        if seqs:
            _ = dnatok.encode_batch_to_ids_staging(
                seqs[: min(chunk, len(seqs))]
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

        pad_id = int(dnatok.id_pad)
        side = dnatok.padding_side
        total_tokens = 0
        valid_ids: List[List[int]] = []

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(seqs), chunk):
            batch = seqs[i : i + chunk]
            ids = dnatok.encode_batch_to_ids_staging(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            # Bit-identical check needs int64 logical content. Convert.
            rows = ids.to(torch.long).cpu().tolist()
            for row in rows:
                if side == "left":
                    j = 0
                    while j < len(row) and row[j] == pad_id:
                        j += 1
                    v = row[j:]
                else:
                    j = len(row)
                    while j > 0 and row[j - 1] == pad_id:
                        j -= 1
                    v = row[:j]
                valid_ids.append(v)
                total_tokens += len(v)
        elapsed = time.perf_counter() - t0

        total_bp = sum(len(s) for s in seqs)
        return (
            TokenizerTiming(
                label="dnatok_native",
                n_sequences=len(seqs),
                total_bp=total_bp,
                total_tokens=total_tokens,
                wall_seconds=elapsed,
                bp_per_second=total_bp / elapsed if elapsed > 0 else 0.0,
                tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0.0,
                notes="staging+int32_H2D",
            ),
            valid_ids,
        )
    finally:
        dnatok.prefer_int32_h2d = prev_pref


def assert_bit_identical(
    hf_tok: Any, dnatok_ids: List[List[int]], seqs: List[str], n_check: int = 50
) -> Tuple[int, list]:
    """Compare DNAtok's stripped IDs against HF single-call per sequence.

    Returns (mismatch_count, first_3_mismatch_examples)."""
    _set_rayon(1)  # use plain Python loop semantics for comparison
    mismatch = 0
    examples = []
    n_check = min(n_check, len(dnatok_ids), len(seqs))
    for i in range(n_check):
        hf_out = hf_tok(seqs[i], add_special_tokens=False)["input_ids"]
        h = hf_out.tolist() if hasattr(hf_out, "tolist") else list(hf_out)
        d = list(dnatok_ids[i])
        if h != d:
            mismatch += 1
            if len(examples) < 3:
                examples.append({
                    "idx": i,
                    "hf_len": len(h),
                    "dnatok_len": len(d),
                    "hf_first8": h[:8],
                    "dnatok_first8": d[:8],
                })
    return mismatch, examples


def format_table(rows: List[TokenizerTiming]) -> str:
    """Pretty-print the headline metric table."""
    if not rows:
        return ""
    dn = next((r for r in rows if r.label == "dnatok"), rows[0])
    out = []
    out.append("=" * 100)
    out.append(f"  {'tokenizer':<14}{'n_seqs':>8}"
               f"{'bp/sec':>14}{'tokens/sec':>14}{'wall (s)':>10}"
               f"{'vs dnatok':>12}")
    out.append("=" * 100)
    for r in rows:
        slowdown = r.wall_seconds / dn.wall_seconds if dn.wall_seconds else 1.0
        out.append(
            f"  {r.label:<14}{r.n_sequences:>8}"
            f"{r.bp_per_second / 1e6:>10.2f} Mbp"
            f"{r.tokens_per_second / 1e6:>10.2f} Mt "
            f"{r.wall_seconds:>9.2f} "
            f"{slowdown:>10.2f}x"
        )
    out.append("=" * 100)
    return "\n".join(out)
