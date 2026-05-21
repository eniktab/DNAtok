#!/usr/bin/env python3
"""
Sustained-throughput sweep at MAX BATCH SIZE per model.

Production deployment regime: each model is given the largest batch
that fits in GPU memory (probed with a few-step doubling search), then
Pipeline 04 runs sustained throughput at that batch. This tests
whether the GPU-starvation finding persists when the framework is
configured to maximise per-batch GPU utilisation (the regime where
Rayon-threaded HF should have its best chance to keep up).

Output: per-model results.json under
  results/case_04_sustained_throughput/max_batch_sweep_<ts>/<model>/.

Each model gets its own JSON with:
  - tested_batches: which batch sizes we tried
  - max_batch_fit: largest batch that didn't OOM
  - results: timing at max_batch_fit (sequential + pipelined × HF/DNAtok)

Honest behaviour on OOM: we catch torch.cuda.OutOfMemoryError, log
it explicitly, and back off — no silent failures.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Models + per-model max sequence length to attempt the sweep at.
# Batch sizes are doubled from `min_batch` until OOM or `max_batch_cap`.
# This is bench design, not the real run — the inner Pipeline 04 logic
# handles the actual benchmark with the chosen batch.
DEFAULT_SPEC = [
    # (model_id, window_bp, min_batch, batch_cap, n_seqs)
    ("LongSafari/hyenadna-tiny-1k-seqlen-hf",        1024, 32,  1024,  5000),
    ("LongSafari/hyenadna-small-32k-seqlen-hf",     4096, 8,    128, 1000),
    ("InstaDeepAI/NTv3_8M_pre",                      4096, 16,   256, 2000),
    ("InstaDeepAI/NTv3_100M_pre",                    2048, 8,    128, 1000),
    ("InstaDeepAI/NTv3_650M_post",                   1024, 4,     64,  500),
    ("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
                                                      1024, 16,   256, 5000),
    ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                                                      1024, 4,     64, 1000),
    ("zhihan1996/DNABERT-2-117M",                    500, 16,   256, 5000),
    ("AIRI-Institute/gena-lm-bert-base-t2t",         1024, 8,    128, 2000),
    ("metagene-ai/METAGENE-1",                       1024, 4,     64, 1000),
    # Caduceus needs mamba_ssm
    ("kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
                                                      1024, 16,   256, 2000),
    # Evo2 — 1B params, big memory
    ("arcinstitute/evo2_1b_base",                     1024, 4,     64,  500),
    ("arcinstitute/evo2_1b_base",                     8192, 1,      8,  200),
    # Evo2-7b only if explicitly requested (heavy)
]

EVO2_7B_SPEC = [
    ("arcinstitute/evo2_7b",                          1024, 2,     16,  200),
    ("arcinstitute/evo2_7b",                          8192, 1,      4,  100),
]


def probe_max_batch(model_id, window_bp, min_batch, batch_cap, *,
                    dry_n=2):
    """Doubling search for the largest batch that doesn't OOM.

    Spawns a subprocess per batch size so each OOM cleanly resets the
    CUDA state. Returns the last successful batch.
    """
    print(f"  Probing max batch for {model_id} @ win={window_bp} ...",
          flush=True)
    last_ok = None
    b = min_batch
    while b <= batch_cap:
        # Run a 2-sample mini-benchmark just to test memory fit.
        cmd = [
            sys.executable,
            str(ROOT / "bio_examples" / "04_sustained_throughput" / "run.py"),
            "--model", model_id, "--window-bp", str(window_bp),
            "--n-sequences", str(max(dry_n, b)),
            "--chunk", str(b), "--hf-threads", "1",
            "--out-dir", "/tmp/probe_oom_dummy",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=600)
        out = (proc.stdout or "") + (proc.stderr or "")
        if "OutOfMemoryError" in out or "out of memory" in out.lower():
            print(f"    batch={b}: OOM", flush=True)
            break
        if proc.returncode != 0:
            print(f"    batch={b}: rc={proc.returncode}, "
                  f"err={out[-200:]}", flush=True)
            break
        print(f"    batch={b}: OK", flush=True)
        last_ok = b
        if b >= batch_cap:
            break
        b *= 2
    return last_ok


def run_at_batch(model_id, window_bp, batch, n_seqs, out_dir):
    """Run Pipeline 04 at the chosen batch with HF-t1, HF-t8, DNAtok
    sequential + pipelined."""
    cmd = [
        sys.executable,
        str(ROOT / "bio_examples" / "04_sustained_throughput" / "run.py"),
        "--model", model_id, "--window-bp", str(window_bp),
        "--n-sequences", str(n_seqs),
        "--chunk", str(batch), "--hf-threads", "1", "8",
        "--out-dir", str(out_dir),
    ]
    print(f"  Running at batch={batch}, n={n_seqs} ...", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=False, timeout=3600)
    return {"rc": proc.returncode, "wall_s": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=None,
                    help="Override the default sweep. Use HF model IDs.")
    ap.add_argument("--include-evo2-7b", action="store_true",
                    help="Add Evo2-7b to the sweep (heavy; only on H200).")
    ap.add_argument("--skip-probe", action="store_true",
                    help="Skip the OOM probe; use min_batch directly.")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "case_04_max_batch_sweep")
    args = ap.parse_args()

    sweep = list(DEFAULT_SPEC)
    if args.include_evo2_7b:
        sweep.extend(EVO2_7B_SPEC)
    if args.models is not None:
        sweep = [t for t in sweep if t[0] in args.models]

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    summary = []
    for model_id, window_bp, min_b, cap_b, n_seqs in sweep:
        model_dir = run_dir / model_id.replace("/", "__") / f"win{window_bp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        print()
        print("=" * 80)
        print(f">>> {model_id}  win={window_bp}")
        print("=" * 80)
        if args.skip_probe:
            chosen = min_b
        else:
            chosen = probe_max_batch(model_id, window_bp, min_b, cap_b)
            if chosen is None:
                print(f"  no batch fits between {min_b} and {cap_b}; "
                      f"skipping")
                summary.append({
                    "model_id": model_id, "window_bp": window_bp,
                    "max_batch_fit": None, "status": "no_fit",
                })
                continue
        info = run_at_batch(model_id, window_bp, chosen, n_seqs, model_dir)
        info["model_id"] = model_id
        info["window_bp"] = window_bp
        info["batch"] = chosen
        info["n_sequences"] = n_seqs
        summary.append(info)

    with open(run_dir / "summary.json", "w") as f:
        json.dump({"timestamp": ts, "models": summary}, f, indent=2)
    print("\n=== sweep summary ===")
    print(f"  {'model':<58}{'win':>6}{'batch':>8}{'rc':>4}{'wall':>10}")
    print("-" * 90)
    for s in summary:
        b = s.get("batch", "n/a"); rc = s.get("rc", s.get("status", "?"))
        w = f"{s.get('wall_s', 0):.1f}s" if "wall_s" in s else "n/a"
        print(f"  {s['model_id'][:57]:<58}{s['window_bp']:>6}"
              f"{str(b):>8}{str(rc):>4}{w:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
