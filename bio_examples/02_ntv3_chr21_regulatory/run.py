#!/usr/bin/env python3
"""
Case study 02: end-to-end NTv3 inference across hg38 chr21.

Real biological data + REAL NTv3 model (no proxy). For each chunk of
chr21 windows we:
  1. tokenise with HF (single-thread AND multi-thread Rayon),
  2. tokenise with DNAtok,
  3. assert bit-identical token IDs,
  4. run the REAL NTv3 forward pass on each tokenisation,
  5. record wall-clock + tokens/sec for both tokenization and forward.

The headline claim — "HF tokenisation starves the GPU even with
multi-thread" — is shown by comparing the GPU-utilisation fraction
(``fwd_s / total_s``) and the tokens/sec feed rate against DNAtok.

Run:
    # Quick local validation:
    python3 bio_examples/02_ntv3_chr21_regulatory/run.py --windows 50

    # Full chr21 sweep:
    python3 bio_examples/02_ntv3_chr21_regulatory/run.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bio_examples"))


def tile_chromosome(seq, *, window_bp, stride_bp, skip_n_threshold=0.5):
    L = len(seq)
    for start in range(0, L - window_bp + 1, stride_bp):
        end = start + window_bp
        win = seq[start:end]
        if win.count("N") / window_bp > skip_n_threshold:
            continue
        yield start, end, win


def _set_rayon(n: int) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(n)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if n > 1 else "false"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chrom", default="chr21")
    ap.add_argument("--window-bp", type=int, default=4_096)
    ap.add_argument("--stride-bp", type=int, default=2_000)
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--model", default="InstaDeepAI/NTv3_8M_pre",
                    help="HF model id. Default is smallest NTv3 (7.7M params) "
                         "which fits on a consumer GPU at 4 kbp inputs.")
    ap.add_argument("--hf-threads", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "case_02_ntv3_chr21")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(run_dir / "log.txt", "w")
    def log(msg: str):
        print(msg); log_handle.write(msg + "\n"); log_handle.flush()

    log(f"\n=== Case 02: NTv3 end-to-end / {args.chrom} ===")

    # --- 1. real data ---
    from _data.download import load_chrom_sequence
    log("[1/5] Loading hg38 chromosome ...")
    seq = load_chrom_sequence(args.chrom)
    log(f"  {args.chrom}: {len(seq):,} bp")

    log(f"[2/5] Tiling at {args.stride_bp:,} bp stride into "
        f"{args.window_bp:,} bp windows ...")
    windows = list(tile_chromosome(seq, window_bp=args.window_bp,
                                     stride_bp=args.stride_bp))
    if args.windows is not None:
        windows = windows[: args.windows]
    n_win = len(windows)
    total_bp = sum(len(w[2]) for w in windows)
    log(f"  {n_win:,} windows, {total_bp / 1e6:.2f} Mbp")
    seqs = [w[2] for w in windows]

    # --- 3. tokenizer + REAL NTv3 model ---
    import torch
    import dnatok_compat  # noqa: F401 — must come before transformers (HF hub version shim)
    from transformers import AutoTokenizer
    from dna_tokenizer import DNATok
    from _common.models import load_ntv3

    log(f"[3/5] Loading REAL NTv3 tokenizer + model ({args.model}) ...")
    hf_tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model, forward_fn, info = load_ntv3(args.model)
    log(f"  tokenizer vocab_size = {getattr(hf_tok, 'vocab_size', len(hf_tok.get_vocab()))}")
    log(f"  REAL NTv3 model: {info['params']/1e6:.2f} M params  "
        f"output={info['output']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _Embedder:
        def __init__(self, tok):
            self.tokenizer = tok
            v = int(getattr(tok, "vocab_size", 0)) or len(tok.get_vocab())
            self.embed_table = torch.nn.Embedding(v + 4, 16).to(device)
        def embed_tokens(self, ids):
            return self.embed_table(ids)

    dnatok = DNATok(_Embedder(hf_tok), normalize_case=False, handle_invalid_chars=False)
    dnatok.discover()

    # --- 4. end-to-end timing ---
    log(f"[4/5] End-to-end timing (chunk={args.chunk}) ...")

    def time_hf_e2e(n_threads):
        _set_rayon(n_threads)
        # Warmup
        for _ in range(2):
            enc = hf_tok(list(seqs[: min(args.chunk, n_win)]),
                          add_special_tokens=False,
                          padding="longest", truncation=False,
                          return_tensors="pt")
            _ = forward_fn(enc["input_ids"])
            if device.type == "cuda": torch.cuda.synchronize()
        # Time
        tok_s = 0.0; fwd_s = 0.0; total_tokens = 0
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, n_win, args.chunk):
            batch = seqs[i : i + args.chunk]
            t = time.perf_counter()
            enc = hf_tok(list(batch), add_special_tokens=False,
                          padding="longest", truncation=False,
                          return_tensors="pt")
            tok_s += time.perf_counter() - t
            ids = enc["input_ids"]
            total_tokens += ids.numel()
            t = time.perf_counter()
            _ = forward_fn(ids)
            if device.type == "cuda": torch.cuda.synchronize()
            fwd_s += time.perf_counter() - t
        total_s = time.perf_counter() - t0
        return {"label": f"hf_t{n_threads}", "tok_s": tok_s,
                "fwd_s": fwd_s, "total_s": total_s,
                "total_tokens": total_tokens}

    def time_dn_e2e():
        # Warmup (kernel JIT)
        for _ in range(2):
            ids = dnatok.encode_batch_to_ids(seqs[: min(args.chunk, n_win)])
            _ = forward_fn(ids)
            if device.type == "cuda": torch.cuda.synchronize()
        tok_s = 0.0; fwd_s = 0.0; total_tokens = 0
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, n_win, args.chunk):
            batch = seqs[i : i + args.chunk]
            t = time.perf_counter()
            ids = dnatok.encode_batch_to_ids(batch)
            if device.type == "cuda": torch.cuda.synchronize()
            tok_s += time.perf_counter() - t
            total_tokens += ids.numel()
            t = time.perf_counter()
            _ = forward_fn(ids)
            if device.type == "cuda": torch.cuda.synchronize()
            fwd_s += time.perf_counter() - t
        total_s = time.perf_counter() - t0
        return {"label": "dnatok", "tok_s": tok_s, "fwd_s": fwd_s,
                "total_s": total_s, "total_tokens": total_tokens}

    results = []
    for n in args.hf_threads:
        log(f"  hf_t{n} ...")
        r = time_hf_e2e(n)
        results.append(r)
        log(f"    tok={r['tok_s']:.2f}s fwd={r['fwd_s']:.2f}s total={r['total_s']:.2f}s "
            f"tokens={r['total_tokens']:,}")
    log("  dnatok ...")
    r = time_dn_e2e()
    results.append(r)
    log(f"    tok={r['tok_s']:.2f}s fwd={r['fwd_s']:.2f}s total={r['total_s']:.2f}s "
        f"tokens={r['total_tokens']:,}")

    # --- 5. bit-identical check ---
    log("[5/5] Bit-identical check (n=50) ...")
    _set_rayon(1)
    hf_per = [hf_tok(s, add_special_tokens=False)["input_ids"]
              for s in seqs[: min(50, n_win)]]
    dn_pad = dnatok.encode_batch_to_ids(seqs[: min(50, n_win)]).cpu().tolist()
    pad_id = int(dnatok.id_pad); side = dnatok.padding_side
    mismatch = 0
    for h, row in zip(hf_per, dn_pad):
        if side == "left":
            j = 0
            while j < len(row) and row[j] == pad_id: j += 1
            v = row[j:]
        else:
            j = len(row)
            while j > 0 and row[j - 1] == pad_id: j -= 1
            v = row[:j]
        h_list = h.tolist() if hasattr(h, "tolist") else list(h)
        if list(v) != h_list: mismatch += 1
    log(f"  bit-identical: {mismatch == 0}  ({mismatch}/{min(50, n_win)} mismatched)")

    # --- Headline summary ---
    log("")
    log("=" * 110)
    log(f"  {'pipeline':<10}{'tok (s)':>9}{'fwd (s)':>9}{'total (s)':>12}"
        f"{'tok %':>8}{'GPU util':>11}{'tok feed (Mt/s)':>17}"
        f"{'e2e (Mt/s)':>13}{'vs dnatok':>12}")
    log("=" * 110)
    dn_total = next(r["total_s"] for r in results if r["label"] == "dnatok")
    for r in results:
        tok_pct = r["tok_s"] / r["total_s"] * 100 if r["total_s"] else 0.0
        gpu_pct = r["fwd_s"] / r["total_s"] * 100 if r["total_s"] else 0.0
        tok_feed_mt = (r["total_tokens"] / r["tok_s"] / 1e6) if r["tok_s"] else 0.0
        e2e_mt = (r["total_tokens"] / r["total_s"] / 1e6) if r["total_s"] else 0.0
        slowdown = r["total_s"] / dn_total
        log(f"  {r['label']:<10}{r['tok_s']:>9.2f}{r['fwd_s']:>9.2f}"
            f"{r['total_s']:>12.2f}{tok_pct:>7.1f}%{gpu_pct:>10.1f}%"
            f"{tok_feed_mt:>16.2f}{e2e_mt:>12.2f}{slowdown:>11.2f}x")
    log("=" * 110)

    res = {
        "case_study": "02_ntv3_chr21_regulatory",
        "model_info": info,
        "chrom": args.chrom,
        "window_bp": args.window_bp,
        "stride_bp": args.stride_bp,
        "n_windows": n_win,
        "total_bp": total_bp,
        "pipelines": results,
        "bit_identical": mismatch == 0,
        "bit_identical_check_n": min(50, n_win),
        "device": str(device),
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2)
    log(f"\n  Results: {run_dir / 'results.json'}")
    log_handle.close()
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
