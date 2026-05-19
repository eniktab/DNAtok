#!/usr/bin/env python3
"""Publication-grade realistic-workload DNA tokenisation benchmark.

This script simulates the workloads a genomic-foundation-model pipeline
actually sees in production, not the synthetic fixed-shape sweeps used
for kernel microbenchmarks.

Workload scenarios
------------------
Each scenario draws thousands of sequences from a length distribution
that matches a specific data source:

* **illumina_short**  — short-read sequencing (NovaSeq / NextSeq).
    n=5 000 reads, length ~ Normal(150, 30), clipped to [50, 300].
    Representative of variant-calling, ChIP-seq, RNA-seq quantification.

* **pacbio_hifi**     — PacBio HiFi / CCS long reads.
    n=1 000 reads, length ~ Normal(15 000, 3 000), clipped to [5 kb, 25 kb].
    Representative of de novo assembly, structural-variant calling.

* **nanopore_long**   — Oxford Nanopore reads.
    n=  500 reads, length ~ LogNormal(mu=ln(10000), sigma=0.7),
    clipped to [1 kb, 100 kb]. Heavy-tailed; some very long reads.
    Representative of methylation calling, plant genomes, ultra-long
    sequencing.

* **gene_models**     — RefSeq-like protein-coding gene length distribution.
    n=2 000 genes, length log-uniform in [500, 50 000].
    Representative of annotation, ortholog detection, prediction.

* **mixed_clinical**  — heterogeneous batch: 60% short, 30% medium,
    10% long. n=2 000 sequences. Closer to a real clinical
    diagnostics pipeline mixing read types.

For each scenario we report wall-clock to tokenise the full set, plus
throughput (sequences/sec and bases/sec). HF native is the baseline.

Models exercised: every BPE / k-mer / char-level genomic foundation
model in `benchmarks/model_registry.py` that is locally available.

Output: results_hpc/realistic/<timestamp>/{bench.csv, bench.json, log.txt}.
The CSV is wide and tidy for downstream plotting.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
import numpy as np  # noqa: E402

import dnatok_compat  # noqa: F401,E402
from benchmarks.model_registry import MODEL_SPECS, resolve_model_path  # noqa: E402
from benchmarks.tokenizer_adapters import load_hf_tokenizer  # noqa: E402

try:
    from src.gputok_bpe_backend import GPUTokBPEBackend  # noqa: E402
except Exception:
    GPUTokBPEBackend = None  # type: ignore[assignment]

from src.dna_tokenizer import DNATok  # noqa: E402


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

# Realistic DNA composition: ~25% each of ACGT with sparse Ns at sequencing
# error / repeat-masking positions. Real human genome is ~41% GC, so we
# bias slightly when generating "genomic" sequences below.
_DNA_BASES = "ACGT"
_DNA_WEIGHTS_BALANCED   = (1, 1, 1, 1)
_DNA_WEIGHTS_HUMAN_GC41 = (29.5, 20.5, 20.5, 29.5)  # A,C,G,T

# Some workloads include the N base where the sequencer wasn't confident.
# Real Illumina has < 0.1% N typically; nanopore raw can have a few %.
_N_RATE_ILLUMINA = 0.001
_N_RATE_NANOPORE = 0.02


def _gen_dna(L: int, rng: random.Random, *, n_rate: float = 0.0,
             gc_bias: bool = False,
             custom_weights: Optional[Tuple[float, ...]] = None) -> str:
    """Draw a single DNA sequence of length L.

    custom_weights: optional (A, C, G, T) weights to override the
    gc_bias / balanced defaults. Used by the GC-content scenarios.
    """
    if L <= 0:
        return ""
    if custom_weights is not None:
        weights = custom_weights
    else:
        weights = _DNA_WEIGHTS_HUMAN_GC41 if gc_bias else _DNA_WEIGHTS_BALANCED
    s = rng.choices(_DNA_BASES, weights=weights, k=L)
    if n_rate > 0.0:
        # Sprinkle Ns at random positions.
        nN = max(1, int(L * n_rate))
        for _ in range(nN):
            s[rng.randrange(L)] = "N"
    return "".join(s)


def _gen_poly_a(L: int, rng: random.Random, *, polya_len: int = 200) -> str:
    """Generate a sequence with a poly-A stretch embedded in random DNA.
    Mirrors mRNA poly-A tails / microsatellites — important edge case
    because every adjacent (A,A) pair maps to the same low rank, so the
    BPE schedule sees a long run of same-rank merges.
    """
    if L <= 0:
        return ""
    polya = "A" * min(polya_len, L)
    if L <= polya_len:
        return polya
    # Random GC-biased flanks around the polyA.
    flank_total = L - polya_len
    left_n = rng.randint(0, flank_total)
    right_n = flank_total - left_n
    left = "".join(rng.choices(_DNA_BASES, weights=_DNA_WEIGHTS_HUMAN_GC41, k=left_n))
    right = "".join(rng.choices(_DNA_BASES, weights=_DNA_WEIGHTS_HUMAN_GC41, k=right_n))
    return left + polya + right


@dataclass
class WorkloadScenario:
    name: str
    description: str
    n_reads: int
    length_sampler: Callable[[random.Random], int]
    n_rate: float = 0.0
    gc_bias: bool = False
    # Override (A, C, G, T) sampling weights for GC-content scenarios.
    custom_weights: Optional[Tuple[float, float, float, float]] = None
    # If set, use this dedicated per-sequence generator instead of the
    # default (length_sampler + _gen_dna).
    seq_generator: Optional[Callable[[int, random.Random], str]] = None

    def generate(self, seed: int = 0) -> List[str]:
        rng = random.Random(seed)
        seqs: List[str] = []
        for _ in range(self.n_reads):
            L = self.length_sampler(rng)
            if self.seq_generator is not None:
                seqs.append(self.seq_generator(L, rng))
            else:
                seqs.append(_gen_dna(L, rng,
                                      n_rate=self.n_rate,
                                      gc_bias=self.gc_bias,
                                      custom_weights=self.custom_weights))
        return seqs

    def length_stats(self, seqs: List[str]) -> Dict[str, float]:
        lens = np.array([len(s) for s in seqs], dtype=np.int64)
        return dict(
            n=int(lens.size),
            min=int(lens.min()),
            p25=int(np.percentile(lens, 25)),
            p50=int(np.median(lens)),
            p75=int(np.percentile(lens, 75)),
            max=int(lens.max()),
            total_bases=int(lens.sum()),
        )


def _clip(lo: int, hi: int, x: int) -> int:
    return max(lo, min(hi, x))


def _make_scenarios() -> List[WorkloadScenario]:
    return [
        WorkloadScenario(
            name="illumina_short",
            description="Short-read sequencing (NovaSeq / NextSeq). "
                        "Length ~ Normal(150, 30), clipped [50, 300].",
            n_reads=5000,
            length_sampler=lambda r: _clip(50, 300,
                                            int(r.gauss(150, 30))),
            n_rate=_N_RATE_ILLUMINA,
            gc_bias=True,
        ),
        WorkloadScenario(
            name="pacbio_hifi",
            description="PacBio HiFi long reads. "
                        "Length ~ Normal(15 000, 3 000), clipped [5k, 25k].",
            n_reads=1000,
            length_sampler=lambda r: _clip(5000, 25000,
                                            int(r.gauss(15000, 3000))),
            n_rate=0.0,
            gc_bias=True,
        ),
        WorkloadScenario(
            name="nanopore_long",
            description="Nanopore long reads. "
                        "Length ~ LogNormal(ln(10000), 0.7), clipped [1k, 100k].",
            n_reads=500,
            length_sampler=lambda r: _clip(1000, 100000,
                                            int(r.lognormvariate(
                                                np.log(10000), 0.7))),
            n_rate=_N_RATE_NANOPORE,
            gc_bias=True,
        ),
        WorkloadScenario(
            name="gene_models",
            description="RefSeq-style gene-length distribution. "
                        "Log-uniform in [500, 50 000].",
            n_reads=2000,
            length_sampler=lambda r: int(
                np.exp(r.uniform(np.log(500), np.log(50000)))),
            n_rate=0.0,
            gc_bias=True,
        ),
        WorkloadScenario(
            name="mixed_clinical",
            description="Heterogeneous clinical batch — 60% short, "
                        "30% medium, 10% long.",
            n_reads=2000,
            length_sampler=lambda r: (
                _clip(50, 300, int(r.gauss(150, 30)))      if r.random() < 0.60
                else _clip(500, 5000, int(r.uniform(500, 5000))) if r.random() < 0.75
                else _clip(5000, 30000, int(r.gauss(15000, 5000)))
            ),
            n_rate=0.005,
            gc_bias=True,
        ),
        # GC-content sweep — different organisms have very different
        # GC%. M. tuberculosis ~65%, P. falciparum (malaria) ~20%,
        # human ~41%, E. coli ~50%. The fast paths must be GC-agnostic.
        WorkloadScenario(
            name="gc_low_20",
            description="AT-rich genome (e.g. Plasmodium, ~20% GC). "
                        "n=2000 reads at 1 kbp.",
            n_reads=2000,
            length_sampler=lambda r: 1000,
            n_rate=0.0,
            custom_weights=(40, 10, 10, 40),  # A,C,G,T → 20% GC
        ),
        WorkloadScenario(
            name="gc_high_65",
            description="GC-rich genome (e.g. Mycobacterium, ~65% GC). "
                        "n=2000 reads at 1 kbp.",
            n_reads=2000,
            length_sampler=lambda r: 1000,
            n_rate=0.0,
            custom_weights=(17.5, 32.5, 32.5, 17.5),  # 65% GC
        ),
        # Special bio patterns — common in real DNA, can hit edge cases
        # in any tokeniser that exploits frequency assumptions.
        WorkloadScenario(
            name="repeats_polyA",
            description="Low-complexity poly-A runs (e.g. mRNA poly-A "
                        "tail, microsatellites). n=1000 reads at 1 kbp, "
                        "each containing a 200-bp poly-A stretch.",
            n_reads=1000,
            length_sampler=lambda r: 1000,
            seq_generator=lambda L, r: _gen_poly_a(L, r, polya_len=200),
        ),
    ]


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------

def _process_in_chunks(seqs: List[str], chunk: int, fn) -> None:
    """Process a long list of sequences in chunks (mimics how a real
    pipeline batches its work). Each chunk is a single tokeniser call.
    """
    for i in range(0, len(seqs), chunk):
        fn(seqs[i : i + chunk])


def _time_total(fn: Callable[[], None], warmup: int, iters: int) -> Dict[str, float]:
    """Run fn `iters` times after `warmup` warmups; return p50/p25/p75 (ms)."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    n = len(times)
    return dict(
        p50=times[n // 2],
        p25=times[n // 4],
        p75=times[(3 * n) // 4],
        min=times[0],
        max=times[-1],
        mean=float(sum(times) / n),
    )


# ---------------------------------------------------------------------------
# Per-model encoder builders
# ---------------------------------------------------------------------------

class _MockEmbedder:
    def __init__(self, tok, hidden: int = 64, device: str = "cuda"):
        self.tokenizer = tok
        try:
            vocab = int(getattr(tok, "vocab_size", 0)) or len(tok.get_vocab())
        except Exception:
            vocab = 1024
        self.embed_table = torch.nn.Embedding(vocab + 4, hidden).to(device)

    def embed_tokens(self, ids):
        return self.embed_table(ids)


def _build_dna_tokeniser(hf_tok, device: torch.device) -> DNATok:
    embedder = _MockEmbedder(hf_tok, device=device)
    dna = DNATok(embedder, normalize_case=False, handle_invalid_chars=False)
    dna.discover()
    return dna


# ---------------------------------------------------------------------------
# Per-model bench
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    model: str
    model_class: str
    scenario: str
    method: str             # hf_native | hf_native_mt | dnatok | gputok_baseline
    chunk: int              # batch size used for the tokeniser calls
    p50_ms: float           # whole-scenario p50 wall-clock
    p25_ms: float
    p75_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    n_reads: int
    total_bases: int
    seqs_per_s: float
    bases_per_s: float
    peak_gpu_mem_mb: float = 0.0  # 0.0 = CPU-only method
    cold_start_ms: float = 0.0     # first call (cold) wall-clock
    notes: str = ""


def _classify(name: str) -> str:
    if name.startswith("NTv2"):
        return "LUT-kmer"
    if name.startswith("DNABERT") or name.startswith("GENA") or name.startswith("METAGENE"):
        return "BPE"
    return "LUT-char"


def _time_cold(fn: Callable[[], None]) -> float:
    """Time a single first call (cold). Resets CUDA memory stats around it."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _peak_gpu_mem_mb() -> float:
    """Peak CUDA allocated memory (MB) since last reset."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024.0 ** 2)


def _set_rayon_threads(n: int) -> None:
    """HuggingFace fast tokenizers parallelise across input strings via
    a Rust thread pool whose width is controlled by RAYON_NUM_THREADS.
    Setting this enables fair comparison against a multi-threaded HF
    baseline (1 thread = single-core baseline; N threads = scaling)."""
    os.environ["RAYON_NUM_THREADS"] = str(n)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _run_scenario(
    model_name: str,
    scenario: WorkloadScenario,
    seqs: List[str],
    *,
    chunks: List[int],
    warmup: int,
    iters: int,
    device: torch.device,
    hf_threads: List[int],
    include_gputok: bool,
    log,
) -> List[TrialResult]:
    spec = next((s for s in MODEL_SPECS if s.name == model_name), None)
    if spec is None:
        return []
    path = resolve_model_path(spec)
    if path is None:
        return []
    try:
        hf = load_hf_tokenizer(str(path))
    except Exception as e:
        log(f"  {model_name}: load failed: {e}")
        return []
    try:
        dna = _build_dna_tokeniser(hf, device)
    except Exception as e:
        log(f"  {model_name}: DNATok discover failed: {e}")
        return []

    # Try to also stand up a GPUTOK-baseline backend for BPE models —
    # the same code path as DNAtok but with the engine flag flipped to
    # the third-party BlockBPE-baseline kernel.
    gputok_backend = None
    if include_gputok and GPUTokBPEBackend is not None:
        try:
            if GPUTokBPEBackend.is_supported(hf):
                gputok_backend = GPUTokBPEBackend(hf, engine="gputok")
        except Exception:
            gputok_backend = None

    cls = _classify(model_name)
    stats = scenario.length_stats(seqs)
    results: List[TrialResult] = []

    for chunk in chunks:
        # 1. HF baselines at varying thread counts.
        for n_threads in hf_threads:
            _set_rayon_threads(n_threads)

            def _hf():
                _process_in_chunks(seqs, chunk,
                                    lambda batch: hf(batch, padding=True,
                                                     return_tensors="pt"))

            try:
                cold = _time_cold(_hf)
                t = _time_total(_hf, warmup=warmup, iters=iters)
            except Exception as e:
                log(f"  {model_name}/{scenario.name}/HF[chunk={chunk},t={n_threads}]: "
                    f"{type(e).__name__}: {e}")
                continue

            label = "hf_native" if n_threads == 1 else f"hf_native_t{n_threads}"
            results.append(TrialResult(
                model=model_name, model_class=cls, scenario=scenario.name,
                method=label, chunk=chunk,
                p50_ms=t["p50"], p25_ms=t["p25"], p75_ms=t["p75"],
                min_ms=t["min"], max_ms=t["max"], mean_ms=t["mean"],
                n_reads=stats["n"], total_bases=stats["total_bases"],
                seqs_per_s=1000.0 * stats["n"] / t["p50"],
                bases_per_s=1000.0 * stats["total_bases"] / t["p50"],
                peak_gpu_mem_mb=0.0,
                cold_start_ms=cold,
                notes=f"RAYON_NUM_THREADS={n_threads}",
            ))

        # 2. Optional: BlockBPE-style GPUTOK baseline.
        if gputok_backend is not None:
            def _gpu_baseline():
                _process_in_chunks(seqs, chunk,
                                    lambda batch: gputok_backend.encode_batch(
                                        batch, device="cuda"))
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                cold = _time_cold(_gpu_baseline)
                t = _time_total(_gpu_baseline, warmup=warmup, iters=iters)
                peak = _peak_gpu_mem_mb()
            except Exception as e:
                log(f"  {model_name}/{scenario.name}/GPUTOK[chunk={chunk}]: "
                    f"{type(e).__name__}: {e}")
                t = None
            if t is not None:
                results.append(TrialResult(
                    model=model_name, model_class=cls, scenario=scenario.name,
                    method="gputok_baseline", chunk=chunk,
                    p50_ms=t["p50"], p25_ms=t["p25"], p75_ms=t["p75"],
                    min_ms=t["min"], max_ms=t["max"], mean_ms=t["mean"],
                    n_reads=stats["n"], total_bases=stats["total_bases"],
                    seqs_per_s=1000.0 * stats["n"] / t["p50"],
                    bases_per_s=1000.0 * stats["total_bases"] / t["p50"],
                    peak_gpu_mem_mb=peak,
                    cold_start_ms=cold,
                    notes="engine=gputok (BlockBPE-style baseline kernel)",
                ))

        # 3. DNATok (our kernel).
        def _bk():
            _process_in_chunks(seqs, chunk,
                                lambda batch: dna.encode_batch_to_ids(batch))
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            cold = _time_cold(_bk)
            t = _time_total(_bk, warmup=warmup, iters=iters)
            peak = _peak_gpu_mem_mb()
        except Exception as e:
            log(f"  {model_name}/{scenario.name}/DNATok[chunk={chunk}]: "
                f"{type(e).__name__}: {e}")
            continue
        results.append(TrialResult(
            model=model_name, model_class=cls, scenario=scenario.name,
            method="dnatok", chunk=chunk,
            p50_ms=t["p50"], p25_ms=t["p25"], p75_ms=t["p75"],
            min_ms=t["min"], max_ms=t["max"], mean_ms=t["mean"],
            n_reads=stats["n"], total_bases=stats["total_bases"],
            seqs_per_s=1000.0 * stats["n"] / t["p50"],
            bases_per_s=1000.0 * stats["total_bases"] / t["p50"],
            peak_gpu_mem_mb=peak,
            cold_start_ms=cold,
        ))
    return results


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results_hpc" / "realistic")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--chunks", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--models", type=str, nargs="+", default=None,
                    help="Model-name filter; defaults to all available.")
    ap.add_argument("--scenarios", type=str, nargs="+", default=None,
                    help="Scenario-name filter; defaults to all.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-threads", type=int, nargs="+", default=[1],
                    help="HF Rust-tokeniser thread counts to bench "
                         "(via RAYON_NUM_THREADS). E.g. '1 4 8' for "
                         "single, 4-thread, and 8-thread baselines.")
    ap.add_argument("--include-gputok", action="store_true", default=False,
                    help="Also bench the third-party BlockBPE-style "
                         "kernel (engine='gputok' inside our backend) "
                         "for BPE models. Apples-to-apples GPU "
                         "tokeniser comparison.")
    args = ap.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    log_handle = open(log_path, "w")
    def log(msg: str):
        print(msg)
        log_handle.write(msg + "\n")
        log_handle.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}")
    log(f"Output: {out_dir}")

    scenarios = _make_scenarios()
    if args.scenarios:
        scenarios = [s for s in scenarios if s.name in args.scenarios]

    # Generate workloads once per scenario (deterministic with --seed).
    workloads: Dict[str, Tuple[WorkloadScenario, List[str]]] = {}
    for sc in scenarios:
        seqs = sc.generate(seed=args.seed)
        stats = sc.length_stats(seqs)
        log(f"\nScenario {sc.name}: {sc.description}")
        log(f"  n={stats['n']}  total_bases={stats['total_bases']:,}  "
            f"min={stats['min']}  p25={stats['p25']}  p50={stats['p50']}  "
            f"p75={stats['p75']}  max={stats['max']}")
        workloads[sc.name] = (sc, seqs)

    # Resolve models.
    available = []
    for spec in MODEL_SPECS:
        if args.models and spec.name not in args.models:
            continue
        if resolve_model_path(spec) is None:
            continue
        available.append(spec.name)
    log(f"\nModels: {len(available)} available — {', '.join(available)}")

    # Run.
    all_results: List[TrialResult] = []
    for model_name in available:
        log(f"\n=== {model_name} ===")
        for scenario_name, (sc, seqs) in workloads.items():
            log(f"  scenario {scenario_name} (n={len(seqs)})")
            res = _run_scenario(model_name, sc, seqs,
                                chunks=args.chunks,
                                warmup=args.warmup, iters=args.iters,
                                device=device,
                                hf_threads=args.hf_threads,
                                include_gputok=args.include_gputok,
                                log=log)
            for r in res:
                mem = f" mem={r.peak_gpu_mem_mb:>5.1f}MB" if r.peak_gpu_mem_mb > 0 else ""
                log(f"    {r.method:<20} chunk={r.chunk:<4} "
                    f"p50={r.p50_ms:>8.2f}ms  "
                    f"{r.seqs_per_s:>10.1f} seq/s  "
                    f"{r.bases_per_s/1e6:>7.2f} Mbase/s"
                    f"{mem}  cold={r.cold_start_ms:>7.1f}ms")
            all_results.extend(res)

    # Persist CSV + JSON.
    csv_path = out_dir / "bench.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        w.writeheader()
        for r in all_results:
            w.writerow(asdict(r))
    log(f"\nWrote {csv_path}")

    json_path = out_dir / "bench.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    log(f"Wrote {json_path}")

    # Summary table (median per (model_class, scenario)).
    log("\n=== Speedup summary (median DNATok p50 vs HF p50) ===")
    from collections import defaultdict
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"hf": [], "bk": []})
    for r in all_results:
        key = (r.model_class, r.scenario)
        if r.method == "hf_native":
            grouped[key]["hf"].append(r.p50_ms)
        elif r.method == "dnatok":
            grouped[key]["bk"].append(r.p50_ms)
    log(f"  {'class':<10} {'scenario':<20} {'HF p50':>9} {'DNAtok p50':>11} {'speedup':>9}")
    for (cls, scn), ts in sorted(grouped.items()):
        if not ts["hf"] or not ts["bk"]:
            continue
        hf_med = sorted(ts["hf"])[len(ts["hf"]) // 2]
        bk_med = sorted(ts["bk"])[len(ts["bk"]) // 2]
        log(f"  {cls:<10} {scn:<20} {hf_med:>7.2f}ms {bk_med:>9.2f}ms "
            f"{hf_med / bk_med:>7.2f}x")

    log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
