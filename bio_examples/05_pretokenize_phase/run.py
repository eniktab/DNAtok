#!/usr/bin/env python3
"""
Case study 05: pre-tokenization phase speedup at lg-asm scale.

This is the workflow that lg-asm actually uses (see
``lg-asm/src/training/generate_embeddings.py``): tokenize DNA strings
into shards offline FIRST, then load token IDs and feed them to the
model. The tokenization phase is CPU-only — the GPU is idle, but the
total pipeline wall-clock includes this preprocessing time.

When you have lg-asm-scale data (hundreds of thousands of long
sequences), the tokenization phase alone can take HOURS on HF native.
DNAtok puts that step on the GPU, collapsing it to seconds.

This script measures **tokenization-only throughput** (Mbp/s, tokens/s)
without any model forward pass — exactly the metric that matters for
the pre-tokenization phase of a streaming pipeline.

Run from the project root:

    # All models at moderate scale:
    python3 bio_examples/05_pretokenize_phase/run.py --n-sequences 10000

    # One model at production scale (50k seqs):
    python3 bio_examples/05_pretokenize_phase/run.py \\
        --models InstaDeepAI/NTv3_8M_pre --n-sequences 50000
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


def _set_rayon(n: int) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(n)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if n > 1 else "false"


# (model_id, window_bp, n_sequences, chunk). Bigger N than Pipeline 04
# because no model forward — we can stress the tokenizer harder.
DEFAULT_MODELS = [
    # Single-base / char — the regime where pre-tokenization dominates.
    ("InstaDeepAI/NTv3_8M_pre",                       4096, 20000, 16),
    ("LongSafari/hyenadna-tiny-1k-seqlen-hf",        1024, 50000, 32),
    # 6-mer (NTv2).
    ("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
                                                       1024, 20000, 32),
    # BPE.
    ("zhihan1996/DNABERT-2-117M",                     1024, 20000, 64),
    ("AIRI-Institute/gena-lm-bert-base-t2t",          2048, 10000, 32),
    # Single-nucleotide byte-level — Evo2.
    ("arcinstitute/evo2_1b_base",                     8192,  1000,  8),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-sequences", type=int, default=None,
                    help="Override per-model N.")
    ap.add_argument("--chrom", default="chr21")
    ap.add_argument("--hf-threads", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "case_05_pretokenize")
    args = ap.parse_args()

    if args.models:
        chosen = []
        for m in args.models:
            match = next((t for t in DEFAULT_MODELS if t[0] == m), None)
            chosen.append(match if match else (m, 1024, 10000, 32))
        sweep = chosen
    else:
        sweep = DEFAULT_MODELS
    if args.n_sequences is not None:
        sweep = [(m, w, args.n_sequences, c) for (m, w, _, c) in sweep]

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(run_dir / "log.txt", "w")
    def log(msg: str):
        print(msg); log_handle.write(msg + "\n"); log_handle.flush()

    log(f"\n=== Case 05: pre-tokenization throughput at lg-asm scale ===")

    # --- 1. real biological data: chr21 ---
    from _data.download import load_chrom_sequence
    log(f"[1/3] Loading {args.chrom} ...")
    seq = load_chrom_sequence(args.chrom)
    log(f"  {args.chrom}: {len(seq):,} bp")

    # We need separate windows per model (different window_bp), so we
    # produce them inside the per-model loop. For simplicity here we
    # tile the chromosome to whatever each model needs.
    import dnatok_compat  # noqa: F401 — must come before transformers
    import torch
    from transformers import AutoTokenizer
    from dna_tokenizer import DNATok
    from benchmarks.tokenizer_adapters import load_hf_tokenizer

    summary = []
    for model_id, window_bp, n_seqs, chunk in sweep:
        log(f"\n[2/3] {model_id}  win={window_bp}  n={n_seqs}  chunk={chunk}")
        windows = []
        pos = 0
        while pos + window_bp <= len(seq) and len(windows) < n_seqs:
            w = seq[pos : pos + window_bp]
            if w.count("N") / window_bp <= 0.5:
                windows.append(w)
            pos += window_bp
        n = len(windows)
        total_bp = n * window_bp
        log(f"  {n:,} windows, {total_bp / 1e6:.1f} Mbp")

        # tokenizer
        hf_tok = (load_hf_tokenizer(model_id) if "evo2" in model_id.lower()
                   else AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Emb:
            def __init__(self, tok):
                self.tokenizer = tok
                v = int(getattr(tok, "vocab_size", 0)) or len(tok.get_vocab())
                self.embed_table = torch.nn.Embedding(v + 4, 16).to(device)
            def embed_tokens(self, ids): return self.embed_table(ids)

        dnatok = DNATok(_Emb(hf_tok), normalize_case=False, handle_invalid_chars=False)
        dnatok.discover()

        results = []

        for n_threads in args.hf_threads:
            _set_rayon(n_threads)
            # Warmup
            _ = hf_tok(list(windows[: min(chunk, n)]),
                        add_special_tokens=False, padding="longest",
                        truncation=False, return_tensors=None)
            total_tokens = 0
            t0 = time.perf_counter()
            for i in range(0, n, chunk):
                batch = windows[i : i + chunk]
                enc = hf_tok(list(batch), add_special_tokens=False,
                              padding=False, truncation=False,
                              return_tensors=None)
                for ids in enc["input_ids"]:
                    total_tokens += len(ids)
            elapsed = time.perf_counter() - t0
            results.append({
                "label": f"hf_t{n_threads}", "wall_s": elapsed,
                "total_tokens": total_tokens,
                "bp_per_s": total_bp / elapsed if elapsed > 0 else 0,
                "tokens_per_s": total_tokens / elapsed if elapsed > 0 else 0,
            })
            log(f"  hf_t{n_threads}: {elapsed:.2f}s  "
                f"{total_bp / elapsed / 1e6:.2f} Mbp/s  "
                f"{total_tokens / elapsed / 1e6:.2f} Mtok/s")

        # DNAtok (legacy API: encode_batch_to_ids, alloc-per-call)
        _ = dnatok.encode_batch_to_ids(windows[: min(chunk, n)])
        if device.type == "cuda": torch.cuda.synchronize()
        total_tokens = 0
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, n, chunk):
            batch = windows[i : i + chunk]
            ids = dnatok.encode_batch_to_ids(batch)
            if device.type == "cuda": torch.cuda.synchronize()
            total_tokens += ids.numel()
        elapsed = time.perf_counter() - t0
        results.append({
            "label": "dnatok", "wall_s": elapsed,
            "total_tokens": total_tokens,
            "bp_per_s": total_bp / elapsed if elapsed > 0 else 0,
            "tokens_per_s": total_tokens / elapsed if elapsed > 0 else 0,
        })
        log(f"  dnatok: {elapsed:.2f}s  "
            f"{total_bp / elapsed / 1e6:.2f} Mbp/s  "
            f"{total_tokens / elapsed / 1e6:.2f} Mtok/s")

        # DNAtok GPU BPE backend path (when available; BPE models only).
        # The default dispatch in encode_batch_to_ids tries CachedLMM
        # first and returns; bpe_backend is only reached on CachedLMM
        # failure. To isolate the GPU BPE kernel's throughput we
        # temporarily disable lmm_bpe — the original instance is
        # restored after the timing loop.
        if getattr(dnatok, "bpe_backend", None) is not None:
            saved_lmm = dnatok.lmm_bpe
            dnatok.lmm_bpe = None
            try:
                _ = dnatok.encode_batch_to_ids(windows[: min(chunk, n)])
                if device.type == "cuda": torch.cuda.synchronize()
                total_tokens_be = 0
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                for i in range(0, n, chunk):
                    batch = windows[i : i + chunk]
                    ids = dnatok.encode_batch_to_ids(batch)
                    if device.type == "cuda": torch.cuda.synchronize()
                    total_tokens_be += ids.numel()
                elapsed_be = time.perf_counter() - t0
                results.append({
                    "label": "dnatok_gpu_bpe", "wall_s": elapsed_be,
                    "total_tokens": total_tokens_be,
                    "bp_per_s": total_bp / elapsed_be if elapsed_be > 0 else 0,
                    "tokens_per_s": total_tokens_be / elapsed_be if elapsed_be > 0 else 0,
                    "notes": "GPUTokBPEBackend (engine=dnatok)",
                })
                log(f"  dnatok_gpu_bpe: {elapsed_be:.2f}s  "
                    f"{total_bp / elapsed_be / 1e6:.2f} Mbp/s  "
                    f"{total_tokens_be / elapsed_be / 1e6:.2f} Mtok/s")
            finally:
                dnatok.lmm_bpe = saved_lmm

        # DNAtok-native (V1 stack: encode_batch_to_ids_staging + int32 H2D)
        # This path skips both CachedLMM and bpe_backend for BPE models
        # (it calls _tokenize_batch_cpu = HF tokenizer with int32 pinning).
        # For byte/char/k-mer models it exercises the ASCII LUT / k-mer
        # fast paths and is meaningful as a "native LUT" measurement.
        prev_pref = getattr(dnatok, "prefer_int32_h2d", False)
        dnatok.prefer_int32_h2d = True
        try:
            _ = dnatok.encode_batch_to_ids_staging(windows[: min(chunk, n)])
            if device.type == "cuda": torch.cuda.synchronize()
            total_tokens_native = 0
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(0, n, chunk):
                batch = windows[i : i + chunk]
                ids = dnatok.encode_batch_to_ids_staging(batch)
                if device.type == "cuda": torch.cuda.synchronize()
                total_tokens_native += ids.numel()
            elapsed_native = time.perf_counter() - t0
            results.append({
                "label": "dnatok_native", "wall_s": elapsed_native,
                "total_tokens": total_tokens_native,
                "bp_per_s": total_bp / elapsed_native if elapsed_native > 0 else 0,
                "tokens_per_s": total_tokens_native / elapsed_native if elapsed_native > 0 else 0,
                "notes": "staging+int32_H2D",
            })
            log(f"  dnatok_native: {elapsed_native:.2f}s  "
                f"{total_bp / elapsed_native / 1e6:.2f} Mbp/s  "
                f"{total_tokens_native / elapsed_native / 1e6:.2f} Mtok/s")
        finally:
            dnatok.prefer_int32_h2d = prev_pref

        # Bit-identical sanity check
        _set_rayon(1)
        n_check = min(50, n)
        hf_per = [hf_tok(s, add_special_tokens=False)["input_ids"]
                   for s in windows[:n_check]]
        dn_pad = dnatok.encode_batch_to_ids(windows[:n_check]).cpu().tolist()
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
        log(f"  bit-identical (n={n_check}): {mismatch == 0}")

        # Cleanup
        del hf_tok, dnatok
        if device.type == "cuda":
            torch.cuda.empty_cache()

        summary.append({
            "model_id": model_id,
            "window_bp": window_bp,
            "n_sequences": n,
            "total_bp": total_bp,
            "chunk": chunk,
            "results": results,
            "bit_identical": mismatch == 0,
        })

    # Headline table — report the best HF variant vs the best DNAtok variant
    # (DNAtok-native uses the V1 staging path + int32 H2D; for BPE models
    # the legacy `dnatok` label allocates per call and is slower. The
    # *best* row is what users would actually deploy.)
    log("")
    log("[3/3] Pre-tokenization speedup at lg-asm scale")
    log("=" * 124)
    log(f"  {'model':<58}{'HF (Mbp/s)':>12}{'HF (Mt/s)':>11}"
        f"{'DNAtok (Mbp/s)':>16}{'DNAtok (Mt/s)':>15}{'speedup':>10}{'variant':>14}")
    log("=" * 124)
    for s in summary:
        hf_best = max((r for r in s["results"] if r["label"].startswith("hf")),
                       key=lambda r: r["bp_per_s"])
        dn_candidates = [r for r in s["results"] if r["label"].startswith("dnatok")]
        dn_best = max(dn_candidates, key=lambda r: r["bp_per_s"])
        speedup = dn_best["bp_per_s"] / max(hf_best["bp_per_s"], 1e-9)
        log(f"  {s['model_id'][:57]:<58}"
            f"{hf_best['bp_per_s']/1e6:>11.2f} "
            f"{hf_best['tokens_per_s']/1e6:>10.2f} "
            f"{dn_best['bp_per_s']/1e6:>15.2f} "
            f"{dn_best['tokens_per_s']/1e6:>14.2f} "
            f"{speedup:>9.2f}x"
            f"  {dn_best['label']:>12}")
    log("=" * 124)

    with open(run_dir / "summary.json", "w") as f:
        json.dump({"summary": summary, "timestamp": ts}, f, indent=2)
    log(f"\nResults: {run_dir / 'summary.json'}")
    log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
