#!/usr/bin/env python3
"""
DNAtok case study 03: DNABERT-2 tokenisation across ENCODE CTCF peaks.

Real biological data:
  - ENCODE GM12878 CTCF ChIP-seq narrowPeak (peak coordinates).
  - hg38 chr21 sequence (UCSC).

For each CTCF peak on chr21, extract a 1 kbp window centred on the
peak; build a matched negative set from random non-peak regions.
Tokenize both sets with HF and DNAtok; assert bit-identical; report
timing.

DNABERT-2 is the BERT-era BPE compatibility case study. Demonstrates
DNAtok works plug-and-play on a 2023 BPE model, not just the newest
single-base ones.

No new biology claimed; we run the standard sequence-classification
input preparation on real ChIP-seq coordinates.

Run from the project root:

    # Validation (50 peaks + 50 background, ~10 s):
    python3 bio_examples/03_dnabert2_encode_ctcf/run.py --n-peaks 50

    # Full chr21 sweep:
    python3 bio_examples/03_dnabert2_encode_ctcf/run.py
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

# Engage Rayon threads in the HF Rust tokenizer before transformers
# imports — must come before any tokenizer load.
os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bio_examples"))


def load_narrowpeak(path: Path, chrom: str = "chr21"):
    """Yield (chrom, start, end, signal) from a (gz) narrowPeak BED file."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0] != chrom:
                continue
            yield fields[0], int(fields[1]), int(fields[2]), float(fields[6])


def build_negative_set(seq: str, positives: list[tuple[int, int]],
                       n_neg: int, win_bp: int, rng: random.Random):
    """Sample non-overlapping background windows from non-peak regions."""
    L = len(seq)
    pos_intervals = [(s - 5_000, e + 5_000) for s, e in positives]
    pos_intervals.sort()
    neg: list[tuple[int, int, str]] = []
    while len(neg) < n_neg:
        start = rng.randint(0, L - win_bp)
        end = start + win_bp
        # Reject if it overlaps any padded peak interval (binary search).
        if any(end > s and start < e for s, e in pos_intervals):
            continue
        win = seq[start:end]
        if win.count("N") / win_bp > 0.5:
            continue
        neg.append((start, end, win))
    return neg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chrom", default="chr21")
    ap.add_argument("--window-bp", type=int, default=1_000,
                    help="1 kbp window centred on each peak.")
    ap.add_argument("--n-peaks", type=int, default=None,
                    help="Limit positives to first N peaks on chrom.")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-threads", type=int, nargs="+", default=[1, 8],
                    help="RAYON_NUM_THREADS values to compare (default 1 8).")
    ap.add_argument("--end-to-end", action="store_true", default=True,
                    help="Run the model forward pass too (default on).")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "case_03_dnabert2_ctcf")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(run_dir / "log.txt", "w")

    def log(msg: str):
        print(msg); log_handle.write(msg + "\n"); log_handle.flush()

    log(f"\n=== DNAtok case 03: DNABERT-2 / ENCODE CTCF on {args.chrom} ===")

    # --- Step 1: real biological data ---
    log("[1/5] Downloading hg38 chromosome + ENCODE CTCF narrowPeak ...")
    from _data.download import load_chrom_sequence, download_encode_ctcf_gm12878
    seq = load_chrom_sequence(args.chrom)
    log(f"  {args.chrom}: {len(seq):,} bp")
    bed_path = download_encode_ctcf_gm12878()
    log(f"  CTCF narrowPeak: {bed_path}")

    # --- Step 2: build positive set (peak-centred windows) ---
    log("[2/5] Building peak-centred positive windows ...")
    peaks = list(load_narrowpeak(bed_path, chrom=args.chrom))
    if args.n_peaks is not None:
        peaks = peaks[: args.n_peaks]
    pos_windows = []
    win_bp = args.window_bp
    for chrom_, s, e, signal in peaks:
        centre = (s + e) // 2
        ws = max(0, centre - win_bp // 2)
        we = ws + win_bp
        if we > len(seq):
            continue
        w = seq[ws:we]
        if w.count("N") / win_bp > 0.5:
            continue
        pos_windows.append((ws, we, w))
    log(f"  {len(pos_windows):,} positive windows")

    # --- Step 3: matched negatives ---
    log("[3/5] Sampling matched negative windows ...")
    rng = random.Random(args.seed)
    pos_intervals = [(ws, we) for ws, we, _ in pos_windows]
    neg_windows = build_negative_set(seq, pos_intervals,
                                       n_neg=len(pos_windows), win_bp=win_bp,
                                       rng=rng)
    log(f"  {len(neg_windows):,} negative windows")

    all_windows = pos_windows + neg_windows
    n_win = len(all_windows)
    total_bp = n_win * win_bp
    log(f"  combined: {n_win:,} windows × {win_bp} bp = {total_bp / 1e6:.3f} Mbp")

    # --- Step 4: load DNABERT-2 (real BertModel; skips custom Triton flash-attn) ---
    log("[4/5] Loading DNABERT-2 tokenizer + REAL BertModel ...")
    import torch
    import dnatok_compat  # noqa: F401 — must come before transformers (HF hub version shim)
    from transformers import AutoTokenizer
    from dna_tokenizer import DNATok
    from _common.models import load_dnabert2
    hf_tok = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_obj, forward_fn_model, info = load_dnabert2("zhihan1996/DNABERT-2-117M")
    n_params = info["params"]
    log(f"  REAL model: {n_params/1e6:.1f} M params  loader={info['loader']}")

    class _Embedder:
        def __init__(self, tok, mod):
            self.tokenizer = tok
            self.model = mod
            embed_module = getattr(mod, "embeddings", None)
            if embed_module is not None and hasattr(embed_module, "word_embeddings"):
                self._embed = embed_module.word_embeddings
            else:
                v = int(getattr(tok, "vocab_size", 0)) or len(tok.get_vocab())
                self._embed = torch.nn.Embedding(v + 4, 16).to(device)
        def embed_tokens(self, ids):
            return self._embed(ids)

    dnatok = DNATok(_Embedder(hf_tok, model_obj),
                    normalize_case=False, handle_invalid_chars=False)
    dnatok.discover()

    # --- Step 5: end-to-end timing ---
    log("[5/5] End-to-end timing (tokenise + GPU forward pass) ...")
    seqs = [w[2] for w in all_windows]
    max_len = args.window_bp + 8

    def run_forward(ids_tensor):
        return forward_fn_model(ids_tensor)

    def time_hf(seqs, chunk, label):
        _ = hf_tok(list(seqs[: min(chunk, len(seqs))]),
                    add_special_tokens=False, padding="longest",
                    truncation=True, max_length=max_len, return_tensors="pt")
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        tok_s = 0.0; fwd_s = 0.0
        for i in range(0, len(seqs), chunk):
            batch = seqs[i : i + chunk]
            t = time.perf_counter()
            enc = hf_tok(list(batch), add_special_tokens=False,
                          padding="longest", truncation=True,
                          max_length=max_len, return_tensors="pt")
            tok_s += time.perf_counter() - t
            t = time.perf_counter()
            _ = run_forward(enc["input_ids"])
            if device.type == "cuda": torch.cuda.synchronize()
            fwd_s += time.perf_counter() - t
        return {"label": label, "tok_s": tok_s, "fwd_s": fwd_s,
                "total_s": time.perf_counter() - t0}

    def time_dn(seqs, chunk):
        _ = dnatok.encode_batch_to_ids(seqs[: min(chunk, len(seqs))])
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        tok_s = 0.0; fwd_s = 0.0
        for i in range(0, len(seqs), chunk):
            batch = seqs[i : i + chunk]
            t = time.perf_counter()
            ids = dnatok.encode_batch_to_ids(batch)
            if device.type == "cuda": torch.cuda.synchronize()
            tok_s += time.perf_counter() - t
            t = time.perf_counter()
            _ = run_forward(ids)
            if device.type == "cuda": torch.cuda.synchronize()
            fwd_s += time.perf_counter() - t
        return {"label": "dnatok", "tok_s": tok_s, "fwd_s": fwd_s,
                "total_s": time.perf_counter() - t0}

    results = []
    for n_threads in args.hf_threads:
        os.environ["RAYON_NUM_THREADS"] = str(n_threads)
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if n_threads > 1 else "false"
        label = f"hf_t{n_threads}"
        log(f"  {label} ...")
        r = time_hf(seqs, args.chunk, label)
        log(f"    tok={r['tok_s']:.2f}s  fwd={r['fwd_s']:.2f}s  total={r['total_s']:.2f}s")
        results.append(r)
    log("  dnatok ...")
    r = time_dn(seqs, args.chunk)
    log(f"    tok={r['tok_s']:.2f}s  fwd={r['fwd_s']:.2f}s  total={r['total_s']:.2f}s")
    results.append(r)

    # Bit-identical check (single-thread)
    os.environ["RAYON_NUM_THREADS"] = "1"
    hf_ids_check = [hf_tok(s, add_special_tokens=False)["input_ids"]
                     for s in seqs[: min(50, n_win)]]
    dn_check = dnatok.encode_batch_to_ids(seqs[: min(50, n_win)]).cpu().tolist()
    pad_id = int(dnatok.id_pad); side = dnatok.padding_side
    mismatch = 0
    for h, row in zip(hf_ids_check, dn_check):
        if side == "left":
            j = 0
            while j < len(row) and row[j] == pad_id: j += 1
            valid = row[j:]
        else:
            j = len(row)
            while j > 0 and row[j - 1] == pad_id: j -= 1
            valid = row[:j]
        h_list = h.tolist() if hasattr(h, "tolist") else list(h)
        if list(valid) != h_list: mismatch += 1

    log("")
    log("=" * 90)
    log(f"  {'pipeline':<14}{'tok (s)':>10}{'fwd (s)':>10}{'total (s)':>12}"
        f"{'tok %':>8}{'GPU util':>12}{'vs dnatok':>12}")
    log("=" * 90)
    dn_total = next(r["total_s"] for r in results if r["label"] == "dnatok")
    for r in results:
        tok_pct = r["tok_s"] / r["total_s"] * 100 if r["total_s"] else 0.0
        gpu_pct = r["fwd_s"] / r["total_s"] * 100 if r["total_s"] else 0.0
        slowdown = r["total_s"] / dn_total
        log(f"  {r['label']:<14}{r['tok_s']:>9.2f} {r['fwd_s']:>9.2f} "
            f"{r['total_s']:>11.2f} {tok_pct:>7.1f}% {gpu_pct:>11.1f}% "
            f"{slowdown:>10.2f}x")
    log("=" * 90)
    log(f"  bit-identical (n=50): {mismatch == 0}")

    res = {
        "case_study": "03_dnabert2_encode_ctcf",
        "model": "zhihan1996/DNABERT-2-117M",
        "model_params": n_params,
        "chrom": args.chrom,
        "window_bp": win_bp,
        "n_positives": len(pos_windows),
        "n_negatives": len(neg_windows),
        "total_bp": total_bp,
        "pipelines": results,
        "bit_identical": mismatch == 0,
        "device": str(device),
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2)
    log(f"\n  Results: {run_dir / 'results.json'}")
    log_handle.close()
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
