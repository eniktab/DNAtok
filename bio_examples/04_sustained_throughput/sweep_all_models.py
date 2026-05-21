#!/usr/bin/env python3
"""
Run the sustained-throughput test across every supported model family.

Each model gets per-model parameters (window length, batch size, n) so
the test fits in the available GPU memory and finishes in a reasonable
wall-clock. Models that fail to load are reported with the error but
do not abort the sweep.

Output:
    results/case_04_sustained_throughput/sweep_<ts>/<model>/results.json
    results/case_04_sustained_throughput/sweep_<ts>/summary.json
    results/case_04_sustained_throughput/sweep_<ts>/summary.txt

Run:
    python3 bio_examples/04_sustained_throughput/sweep_all_models.py \\
        [--models MODEL1 MODEL2 ...]  # default: representative subset

The sweep is intentionally a SUBSET by default — five representative
models, one per family, sized to fit on a single 16 GB GPU. Override
with --models for the full 21-variant matrix on bigger hardware.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (model_id, window_bp, n_sequences, chunk).
# Picked so each completes in ~2-5 min on GB10 and fits in 16 GB.
# Memory rule of thumb: chunk × window × params × 4 bytes ≤ 4 GB.
DEFAULT_MODELS = [
    # Char / single-base (the headline regime — biggest starvation).
    ("LongSafari/hyenadna-tiny-1k-seqlen-hf",        1024, 5000, 32),
    ("LongSafari/hyenadna-small-32k-seqlen-hf",     4096, 1000,  8),
    ("kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
                                                     1024, 2000, 16),
    ("InstaDeepAI/NTv3_8M_pre",                     4096, 2000,  8),
    ("InstaDeepAI/NTv3_100M_pre",                   2048, 1000,  4),
    # k-mer (NTv2 6-mer). At single-mer 6, max tokens are ~166 per kbp.
    ("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
                                                     1024, 5000, 16),
    ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                                                     1024, 2000,  8),
    # BPE (the BERT-era family — small relative tokeniser).
    # DNABERT-2 max_position_embeddings=512 → keep dna window short so
    # BPE output stays under 512 tokens.
    ("zhihan1996/DNABERT-2-117M",                    500, 5000, 32),
    ("AIRI-Institute/gena-lm-bert-base-t2t",        1024, 2000, 16),
    # Evo2 — single-nucleotide, big model. Smaller N + chunk.
    ("arcinstitute/evo2_1b_base",                    1024,  200,  2),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=None,
                    help="Override the default model sweep. Use one of "
                         "the HF model IDs; per-model params are inferred "
                         "from the defaults when present.")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "case_04_sustained_throughput")
    ap.add_argument("--hf-threads", type=int, nargs="+", default=[1, 8])
    args = ap.parse_args()

    if args.models is not None:
        # Filter the defaults to the requested set; if a user passes
        # something not in the defaults, use sensible fallback params.
        chosen = []
        for m in args.models:
            match = next((t for t in DEFAULT_MODELS if t[0] == m), None)
            chosen.append(match if match else (m, 1024, 2000, 16))
        sweep = chosen
    else:
        sweep = DEFAULT_MODELS

    ts = time.strftime("%Y%m%d-%H%M%S")
    sweep_dir = args.out_dir / f"sweep_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running sustained-throughput sweep across {len(sweep)} models")
    print(f"Output: {sweep_dir}")

    summary = {"timestamp": ts, "models": []}
    for model_id, window_bp, n_seqs, chunk in sweep:
        model_dir = sweep_dir / model_id.replace("/", "__")
        model_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "bio_examples" / "04_sustained_throughput" / "run.py"),
            "--model", model_id,
            "--window-bp", str(window_bp),
            "--n-sequences", str(n_seqs),
            "--chunk", str(chunk),
            "--hf-threads", *[str(t) for t in args.hf_threads],
            "--out-dir", str(model_dir),
        ]
        print()
        print("=" * 80)
        print(f">>> {model_id}  win={window_bp}  n={n_seqs}  chunk={chunk}")
        print("=" * 80)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=False, timeout=1800)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        elapsed = time.perf_counter() - t0
        # Pick up the most recent results.json under model_dir.
        rj_files = sorted(model_dir.glob("*/results.json"))
        rj = rj_files[-1] if rj_files else None
        summary["models"].append({
            "model_id": model_id,
            "window_bp": window_bp,
            "n_sequences": n_seqs,
            "chunk": chunk,
            "rc": rc,
            "wall_s": elapsed,
            "results_json": str(rj) if rj else None,
        })

    # Aggregate.
    with open(sweep_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty-print summary text.
    lines = ["", "=" * 100,
             f"  {'model':<58}{'rc':>4}{'wall (s)':>10}",
             "=" * 100]
    for m in summary["models"]:
        lines.append(f"  {m['model_id'][:57]:<58}"
                     f"{str(m['rc']):>4}{m['wall_s']:>9.1f}")
    lines.append("=" * 100)
    lines.append(f"Sweep output: {sweep_dir}")
    text = "\n".join(lines)
    print(text)
    with open(sweep_dir / "summary.txt", "w") as f:
        f.write(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
