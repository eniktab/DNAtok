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
             gc_bias: bool = False) -> str:
    """Draw a single DNA sequence of length L."""
    if L <= 0:
        return ""
    weights = _DNA_WEIGHTS_HUMAN_GC41 if gc_bias else _DNA_WEIGHTS_BALANCED
    s = rng.choices(_DNA_BASES, weights=weights, k=L)
    if n_rate > 0.0:
        # Sprinkle Ns at random positions.
        nN = max(1, int(L * n_rate))
        for _ in range(nN):
            s[rng.randrange(L)] = "N"
    return "".join(s)


@dataclass
class WorkloadScenario:
    name: str
    description: str
    n_reads: int
    length_sampler: Callable[[random.Random], int]
    n_rate: float = 0.0
    gc_bias: bool = False

    def generate(self, seed: int = 0) -> List[str]:
        rng = random.Random(seed)
        return [
            _gen_dna(self.length_sampler(rng), rng,
                     n_rate=self.n_rate, gc_bias=self.gc_bias)
            for _ in range(self.n_reads)
        ]

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
    method: str             # hf_native | dnatok
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
    notes: str = ""


def _classify(name: str) -> str:
    if name.startswith("NTv2"):
        return "LUT-kmer"
    if name.startswith("DNABERT") or name.startswith("GENA") or name.startswith("METAGENE"):
        return "BPE"
    return "LUT-char"


def _run_scenario(
    model_name: str,
    scenario: WorkloadScenario,
    seqs: List[str],
    *,
    chunks: List[int],
    warmup: int,
    iters: int,
    device: torch.device,
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

    cls = _classify(model_name)
    stats = scenario.length_stats(seqs)
    results: List[TrialResult] = []

    for chunk in chunks:
        def _hf():
            _process_in_chunks(seqs, chunk,
                                lambda batch: hf(batch, padding=True,
                                                 return_tensors="pt"))

        def _bk():
            _process_in_chunks(seqs, chunk,
                                lambda batch: dna.encode_batch_to_ids(batch))

        try:
            t_hf = _time_total(_hf, warmup=warmup, iters=iters)
        except Exception as e:
            log(f"  {model_name}/{scenario.name}/HF[chunk={chunk}]: {type(e).__name__}: {e}")
            continue
        try:
            t_bk = _time_total(_bk, warmup=warmup, iters=iters)
        except Exception as e:
            log(f"  {model_name}/{scenario.name}/DNATok[chunk={chunk}]: {type(e).__name__}: {e}")
            continue

        for label, t in (("hf_native", t_hf), ("dnatok", t_bk)):
            results.append(TrialResult(
                model=model_name,
                model_class=cls,
                scenario=scenario.name,
                method=label,
                chunk=chunk,
                p50_ms=t["p50"],
                p25_ms=t["p25"],
                p75_ms=t["p75"],
                min_ms=t["min"],
                max_ms=t["max"],
                mean_ms=t["mean"],
                n_reads=stats["n"],
                total_bases=stats["total_bases"],
                seqs_per_s=1000.0 * stats["n"] / t["p50"],
                bases_per_s=1000.0 * stats["total_bases"] / t["p50"],
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
                                device=device, log=log)
            for r in res:
                log(f"    {r.method:<10} chunk={r.chunk:<4} "
                    f"p50={r.p50_ms:>8.2f}ms  "
                    f"{r.seqs_per_s:>10.1f} seq/s  "
                    f"{r.bases_per_s/1e6:>7.2f} Mbase/s")
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
