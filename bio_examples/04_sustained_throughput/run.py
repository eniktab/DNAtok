#!/usr/bin/env python3
"""
Case study 04: SUSTAINED throughput at lg-asm-scale workload.

This is the experimental design the user (a previous lg-asm contributor)
spelled out: in real production we don't run a single batch — we feed
THOUSANDS of sequences through the GFM in a pipelined fashion, and the
question is which step rate-limits sustained GPU utilisation.

Two regimes per tokenizer:

  (a) SEQUENTIAL — for batch i: tokenise then forward; CPU and GPU
      take turns. wall = sum(tok_i) + sum(fwd_i). Any time the
      tokenizer is slow, the GPU sits idle.

  (b) PIPELINED  — producer thread tokenises batch i+1 on the CPU
      while the main thread runs forward on batch i. wall =
      max(sum(tok_i), sum(fwd_i)) + small startup. If tokenize > GPU,
      the GPU still stalls waiting for the queue to refill.

The metric that matters: **sustained tokens/sec** at steady state.
This equals min(producer_rate, consumer_rate) under pipelining. It
quantifies "tokens per second of GPU inference", which is what the
user asked for.

We also sample GPU utilisation throughout the run via nvidia-smi.
"""
from __future__ import annotations
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bio_examples"))


# ---------------------------------------------------------------------
# Light-weight GPU sampler
# ---------------------------------------------------------------------

class GPUSampler:
    """Sample nvidia-smi GPU utilisation + memory in a background thread."""

    def __init__(self, period_s: float = 0.1):
        self.period_s = period_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        n_errors = 0
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.0,
                ).stdout.strip().split(",")

                def _i(s: str):
                    s = s.strip()
                    if not s or s == "[N/A]":
                        return None
                    try: return int(s)
                    except ValueError: return None

                self.samples.append({
                    "t": time.perf_counter(),
                    "util_pct": _i(out[0]),
                    "mem_used_mib": _i(out[1]),
                    "mem_total_mib": _i(out[2]),
                })
            except Exception as e:
                n_errors += 1
                # Print the first error so the user sees the cause; then
                # silently count further errors so we don't spam stdout.
                if n_errors == 1:
                    print(f"[GPUSampler] nvidia-smi sample failed: "
                          f"{type(e).__name__}: {e}", flush=True)
            self._stop.wait(self.period_s)
        if n_errors:
            print(f"[GPUSampler] {n_errors} sample errors (continued silently after the first)",
                  flush=True)

    def summary(self) -> dict:
        utils = sorted(s["util_pct"] for s in self.samples
                        if s["util_pct"] is not None)
        mems = [s["mem_used_mib"] for s in self.samples
                 if s["mem_used_mib"] is not None]
        n_u = len(utils)
        total_mib = next((s["mem_total_mib"] for s in self.samples
                          if s["mem_total_mib"] is not None), None)
        return {
            "util_mean": (sum(utils) / n_u) if n_u else None,
            "util_p50": utils[n_u // 2] if n_u else None,
            "util_p95": utils[min(n_u - 1, int(0.95 * n_u))] if n_u else None,
            "mem_peak_mib": max(mems) if mems else None,
            "mem_total_mib": total_mib,
            "n_samples": len(self.samples),
            "n_util_samples": n_u,
            "n_mem_samples": len(mems),
        }


# ---------------------------------------------------------------------
# Pipelined producer/consumer
# ---------------------------------------------------------------------

def run_pipelined(seqs, tokenize_fn, forward_fn, *, chunk: int,
                   queue_size: int = 4) -> dict:
    """Producer-consumer with overlapping tokenize / forward.

    tokenize_fn(batch_strs) -> Tensor of token IDs on CPU
    forward_fn(ids_tensor) -> any  (caller does GPU sync internally)
    Returns dict with tok_s/fwd_s/wait_s/total_s/total_tokens.
    """
    import torch
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    tok_s_total = 0.0
    fwd_s_total = 0.0
    wait_s_total = 0.0
    tok_lock = threading.Lock()

    def producer():
        nonlocal tok_s_total
        for i in range(0, len(seqs), chunk):
            batch = seqs[i : i + chunk]
            t = time.perf_counter()
            ids = tokenize_fn(batch)
            elapsed = time.perf_counter() - t
            with tok_lock:
                tok_s_total += elapsed
            q.put(ids)
        q.put(None)  # sentinel

    total_tokens = 0
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_prod = threading.Thread(target=producer, daemon=True)
    t_prod.start()
    while True:
        t_w = time.perf_counter()
        ids = q.get()
        wait_s_total += time.perf_counter() - t_w
        if ids is None:
            break
        total_tokens += ids.numel()
        t = time.perf_counter()
        _ = forward_fn(ids)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd_s_total += time.perf_counter() - t
    t_prod.join()
    total_s = time.perf_counter() - t0
    return {
        "tok_s": tok_s_total, "fwd_s": fwd_s_total,
        "wait_s": wait_s_total, "total_s": total_s,
        "total_tokens": total_tokens,
    }


def run_sequential_dnatok_native(
    seqs, dnatok, forward_fn, *, chunk: int
) -> dict:
    """V1 stack — sequential. Uses encode_batch_to_ids_staging (persistent
    pinned buffer reuse) + int32 H2D + on-device int64 cast for embedding.

    The cast is what V1 §3.2 calls "nearly free": the int32 staging halves
    the PCIe bytes while int32→int64 on device runs at SM bandwidth.
    """
    import torch
    prev_pref = getattr(dnatok, "prefer_int32_h2d", False)
    dnatok.prefer_int32_h2d = True
    tok_s_total = 0.0
    fwd_s_total = 0.0
    total_tokens = 0
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(seqs), chunk):
            batch = seqs[i : i + chunk]
            t = time.perf_counter()
            ids_cpu = dnatok.encode_batch_to_ids_staging(batch)
            tok_s_total += time.perf_counter() - t
            total_tokens += ids_cpu.numel()
            t = time.perf_counter()
            # int32 pinned → device (non-blocking H2D) → int64 cast on device
            ids_dev_i32 = ids_cpu.to("cuda", non_blocking=True)
            ids_dev = ids_dev_i32.to(torch.long)
            _ = forward_fn(ids_dev)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_s_total += time.perf_counter() - t
        total_s = time.perf_counter() - t0
    finally:
        dnatok.prefer_int32_h2d = prev_pref
    return {
        "tok_s": tok_s_total, "fwd_s": fwd_s_total,
        "wait_s": 0.0, "total_s": total_s,
        "total_tokens": total_tokens,
        "engine": "dnatok_native_seq",
    }


def run_pipelined_dnatok_native(
    seqs, dnatok, forward_fn, *, chunk: int, queue_size: int = 4
) -> dict:
    """V1 stack — pipelined. Producer thread runs the staging tokenizer
    into the persistent pinned buffer; consumer thread pulls the staging
    tensor, does int32 non-blocking H2D, casts to int64 on device, then
    forwards.

    NOTE: the producer races on the SAME staging buffer, so the producer
    must wait for the consumer to consume each tensor before overwriting.
    We achieve this with a `queue.Queue(maxsize=1)` so put() blocks until
    get() drains. That gives the CUDA driver a chance to launch H2D
    asynchronously on the consumer side while the producer prepares the
    next batch's CPU-side ASCII→IDs work.
    """
    import torch
    prev_pref = getattr(dnatok, "prefer_int32_h2d", False)
    dnatok.prefer_int32_h2d = True

    # Each producer step replaces the staging buffer's contents. To avoid
    # the consumer reading stale data we hand it a *clone* of the staging
    # tensor (still pinned) so the producer can immediately overwrite.
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    tok_s_total = 0.0
    fwd_s_total = 0.0
    wait_s_total = 0.0
    tok_lock = threading.Lock()

    def producer():
        nonlocal tok_s_total
        try:
            for i in range(0, len(seqs), chunk):
                batch = seqs[i : i + chunk]
                t = time.perf_counter()
                ids_staging = dnatok.encode_batch_to_ids_staging(batch)
                # Clone into a fresh pinned tensor so the staging buffer
                # can be reused on the next iteration without racing.
                ids_cpu = torch.empty_like(ids_staging).pin_memory()
                ids_cpu.copy_(ids_staging)
                elapsed = time.perf_counter() - t
                with tok_lock:
                    tok_s_total += elapsed
                q.put(ids_cpu)
            q.put(None)
        except Exception as e:
            q.put(("ERR", e))

    total_tokens = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_prod = threading.Thread(target=producer, daemon=True)
    t_prod.start()
    try:
        while True:
            t_w = time.perf_counter()
            item = q.get()
            wait_s_total += time.perf_counter() - t_w
            if item is None:
                break
            if isinstance(item, tuple) and item[0] == "ERR":
                raise item[1]
            ids_cpu = item
            total_tokens += ids_cpu.numel()
            t = time.perf_counter()
            ids_dev_i32 = ids_cpu.to("cuda", non_blocking=True)
            ids_dev = ids_dev_i32.to(torch.long)
            _ = forward_fn(ids_dev)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_s_total += time.perf_counter() - t
    finally:
        t_prod.join()
        dnatok.prefer_int32_h2d = prev_pref
    total_s = time.perf_counter() - t0
    return {
        "tok_s": tok_s_total, "fwd_s": fwd_s_total,
        "wait_s": wait_s_total, "total_s": total_s,
        "total_tokens": total_tokens,
        "engine": "dnatok_native_pipelined",
    }


def run_sequential(seqs, tokenize_fn, forward_fn, *, chunk: int) -> dict:
    import torch
    tok_s_total = 0.0
    fwd_s_total = 0.0
    total_tokens = 0
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(seqs), chunk):
        batch = seqs[i : i + chunk]
        t = time.perf_counter()
        ids = tokenize_fn(batch)
        tok_s_total += time.perf_counter() - t
        total_tokens += ids.numel()
        t = time.perf_counter()
        _ = forward_fn(ids)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd_s_total += time.perf_counter() - t
    total_s = time.perf_counter() - t0
    return {"tok_s": tok_s_total, "fwd_s": fwd_s_total,
            "wait_s": 0.0, "total_s": total_s,
            "total_tokens": total_tokens}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chrom", default="chr21")
    ap.add_argument("--n-sequences", type=int, default=5000,
                    help="High-throughput regime: many sequences.")
    ap.add_argument("--window-bp", type=int, default=4_096)
    ap.add_argument("--stride-bp", type=int, default=4_000)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--model", default="InstaDeepAI/NTv3_8M_pre",
                    help="Use the small NTv3-8M by default — the regime "
                         "where tokenisation is the bottleneck. For very "
                         "large models (e.g. Evo2-1b) the GPU itself "
                         "dominates and we'd expect smaller pipelining "
                         "benefit; honest reporting.")
    ap.add_argument("--hf-threads", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "case_04_sustained_throughput")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(run_dir / "log.txt", "w")
    def log(msg: str):
        print(msg); log_handle.write(msg + "\n"); log_handle.flush()

    log(f"\n=== Case 04: sustained pipelined throughput / {args.chrom} ===")
    log("Question: at lg-asm-scale workload (thousands of reads, batched, "
        "pipelined), what's the sustained tokens/sec the GPU can ingest? "
        "Sequential AND overlapped regimes reported.\n")

    # --- 1. real data ---
    from _data.download import load_chrom_sequence
    log("[1/4] Loading hg38 chromosome ...")
    seq = load_chrom_sequence(args.chrom)
    log(f"  {args.chrom}: {len(seq):,} bp")

    log(f"[2/4] Tiling to ~{args.n_sequences} usable {args.window_bp:,} bp windows ...")
    L = len(seq)
    windows = []
    pos = 0
    while pos + args.window_bp <= L and len(windows) < args.n_sequences:
        win = seq[pos : pos + args.window_bp]
        if win.count("N") / args.window_bp <= 0.5:
            windows.append(win)
        pos += args.stride_bp
    n = len(windows)
    total_bp = n * args.window_bp
    log(f"  {n:,} windows ({total_bp / 1e6:.1f} Mbp total)")

    # --- 3. real tokenizer + real model ---
    log(f"[3/4] Loading REAL tokenizer + model ({args.model}) ...")
    # dnatok_compat MUST import before transformers (patches the
    # huggingface-hub version metadata so the in-container 1.2.x
    # passes transformers' <1.0 gate).
    import dnatok_compat  # noqa: F401
    import torch
    from transformers import AutoTokenizer
    from dna_tokenizer import DNATok
    from _common.models import load_model_auto
    from benchmarks.tokenizer_adapters import load_hf_tokenizer

    # load_hf_tokenizer handles Evo2's byte tokenizer; falls back to
    # AutoTokenizer otherwise.
    hf_tok = load_hf_tokenizer(args.model) if "evo2" in args.model.lower() else \
             AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_obj, forward_fn, info = load_model_auto(args.model)
    log(f"  REAL model: {info['params']/1e6:.2f} M params")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _Embedder:
        def __init__(self, tok):
            self.tokenizer = tok
            v = int(getattr(tok, "vocab_size", 0)) or len(tok.get_vocab())
            self.embed_table = torch.nn.Embedding(v + 4, 16).to(device)
        def embed_tokens(self, ids): return self.embed_table(ids)

    dnatok = DNATok(_Embedder(hf_tok), normalize_case=False, handle_invalid_chars=False)
    dnatok.discover()

    def hf_tok_fn(batch):
        return hf_tok(list(batch), add_special_tokens=False, padding="longest",
                      truncation=False, return_tensors="pt")["input_ids"]

    def dn_tok_fn(batch):
        return dnatok.encode_batch_to_ids(batch)

    def fwd(ids):
        return forward_fn(ids)

    # Warmup
    log("  Warming up ...")
    for _ in range(3):
        fwd(hf_tok_fn(windows[: args.chunk]))
        fwd(dn_tok_fn(windows[: args.chunk]))
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # --- 4. measurement ---
    log(f"[4/4] Timing {n:,} sequences × {args.window_bp:,} bp "
        f"(chunk={args.chunk}, queue=4) ...")

    def measure(label, tokenize_fn, mode):
        sampler = GPUSampler(period_s=0.1)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        sampler.start()
        if mode == "sequential":
            r = run_sequential(windows, tokenize_fn, fwd, chunk=args.chunk)
        else:
            r = run_pipelined(windows, tokenize_fn, fwd, chunk=args.chunk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        sampler.stop()
        r["label"] = label
        r["mode"] = mode
        r["gpu"] = sampler.summary()
        r["torch_peak_mem_mib"] = (
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
        )
        # Derived metric: tokens / second of full pipeline.
        r["e2e_tokens_per_s"] = (
            r["total_tokens"] / r["total_s"] if r["total_s"] > 0 else 0.0
        )
        return r

    results = []
    for n_threads in args.hf_threads:
        os.environ["RAYON_NUM_THREADS"] = str(n_threads)
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if n_threads > 1 else "false"
        for mode in ("sequential", "pipelined"):
            label = f"hf_t{n_threads}_{mode}"
            log(f"  {label} ...")
            r = measure(label, hf_tok_fn, mode)
            results.append(r)
            log(f"    total={r['total_s']:.2f}s  tok={r['tok_s']:.2f}s "
                f"fwd={r['fwd_s']:.2f}s  e2e={r['e2e_tokens_per_s']/1e6:.2f} Mt/s  "
                f"GPU util mean={r['gpu']['util_mean']}%")
    for mode in ("sequential", "pipelined"):
        label = f"dnatok_{mode}"
        log(f"  {label} ...")
        r = measure(label, dn_tok_fn, mode)
        results.append(r)
        log(f"    total={r['total_s']:.2f}s  tok={r['tok_s']:.2f}s "
            f"fwd={r['fwd_s']:.2f}s  e2e={r['e2e_tokens_per_s']/1e6:.2f} Mt/s  "
            f"GPU util mean={r['gpu']['util_mean']}%")

    # ---- DNAtok-NATIVE (V1 stack: staging + int32 H2D + on-device cast) ----
    # These two extra rows exercise the V1 systems-layer (persistent pinned
    # buffer reuse, int32 H2D, ping-pong-style overlap) which the legacy
    # dnatok_* rows above do NOT use. They are the apples-to-apples
    # comparison for "what is DNAtok actually capable of at production
    # rate with its full V1 stack engaged."
    def measure_native(label, mode):
        sampler = GPUSampler(period_s=0.1)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        sampler.start()
        if mode == "sequential":
            r = run_sequential_dnatok_native(
                windows, dnatok, fwd, chunk=args.chunk
            )
        else:
            r = run_pipelined_dnatok_native(
                windows, dnatok, fwd, chunk=args.chunk
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        sampler.stop()
        r["label"] = label
        r["mode"] = mode
        r["gpu"] = sampler.summary()
        r["torch_peak_mem_mib"] = (
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
        )
        r["e2e_tokens_per_s"] = (
            r["total_tokens"] / r["total_s"] if r["total_s"] > 0 else 0.0
        )
        return r

    for mode in ("sequential", "pipelined"):
        label = f"dnatok_native_{mode}"
        log(f"  {label} ...")
        try:
            r = measure_native(label, mode)
            results.append(r)
            log(f"    total={r['total_s']:.2f}s  tok={r['tok_s']:.2f}s "
                f"fwd={r['fwd_s']:.2f}s  e2e={r['e2e_tokens_per_s']/1e6:.2f} Mt/s  "
                f"GPU util mean={r['gpu']['util_mean']}%")
        except Exception as e:
            log(f"    dnatok_native_{mode} FAILED: {e}")

    # --- Summary table ---
    log("")
    log("=" * 120)
    log(f"  {'pipeline':<22}{'mode':<11}{'total':>9}"
        f"{'tok_s':>9}{'fwd_s':>9}{'wait_s':>9}"
        f"{'e2e Mt/s':>11}{'GPU mean':>11}{'GPU p95':>10}{'peak mem (MiB)':>16}")
    log("=" * 120)
    dn_pipe = next(r for r in results
                    if r["label"] == "dnatok_pipelined")
    dn_e2e = dn_pipe["e2e_tokens_per_s"]
    for r in results:
        g = r["gpu"]
        ratio = (r["e2e_tokens_per_s"] / dn_e2e) if dn_e2e > 0 else 0.0
        u_mean = f"{g['util_mean']:.1f}%" if g['util_mean'] is not None else "n/a"
        u_p95 = f"{g['util_p95']}%" if g['util_p95'] is not None else "n/a"
        mem = f"{r['torch_peak_mem_mib']:.1f}" if r['torch_peak_mem_mib'] else "n/a"
        log(f"  {r['label'][:21]:<22}{r['mode']:<11}{r['total_s']:>8.2f}s"
            f"{r['tok_s']:>8.2f}s{r['fwd_s']:>8.2f}s{r['wait_s']:>8.2f}s"
            f"{r['e2e_tokens_per_s']/1e6:>10.2f}"
            f"{u_mean:>11}{u_p95:>10}"
            f"{mem:>16}")
    log("=" * 120)
    log("Key claim: sustained e2e tokens/sec is what the GPU actually "
        "ingests. The pipelined dnatok value is the upper bound; any HF "
        "row below it is the starvation gap.")

    res = {
        "case_study": "04_sustained_throughput",
        "model_info": info,
        "chrom": args.chrom,
        "n_sequences": n,
        "window_bp": args.window_bp,
        "chunk": args.chunk,
        "queue_size": 4,
        "results": results,
        "device": str(device),
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2)
    log(f"\n  Results: {run_dir / 'results.json'}")
    log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
