"""Publication-grade cross-platform table across GB10, H200, A100.

For each model x scenario x phase, computes speedup of the best dnatok method vs hf_native.
Then aggregates per-platform median, geometric mean, and per-class breakdown.
"""
import csv, glob, os, statistics
from collections import defaultdict
from pathlib import Path

def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                r["p50_ms"] = float(r["p50_ms"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows

def find_latest(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None

gb10 = find_latest("results_hpc/publication/__pub_grade_gb10/*/bench_summary.csv")
h200 = find_latest("results_hpc/publication_h200/*/bench_summary.csv")
a100 = find_latest("results_hpc/publication_a100/*/bench_summary.csv")

platforms = {"GB10": gb10, "H200": h200, "A100": a100}
data = {p: (load(path) if path else []) for p, path in platforms.items()}

# (plat, model, scenario, phase, method) -> p50
T = {}
for plat, rows in data.items():
    for r in rows:
        T[(plat, r["model"], r["scenario"], r["phase"], r["method"])] = r["p50_ms"]

def best_dnatok(plat, model, scenario, phase):
    candidates = [
        "dnatok_cuda_graph", "dnatok_dna_kernel", "dnatok_default",
        "dnatok_pack2bit_uint8", "dnatok_gputok_bpe", "dnatok_fused_triton",
    ]
    out = None
    for m in candidates:
        v = T.get((plat, model, scenario, phase, m))
        if v is None:
            continue
        if out is None or v < out:
            out = v
    return out

MODELS_BPE = ["DNABERT2_117M", "GENA_LM_BERT_t2t", "METAGENE_1"]
MODELS_LUT_KMER = ["NTv2_50M", "NTv2_500M"]
MODELS_LUT_CHAR = [
    "HyenaDNA_tiny_1k", "HyenaDNA_small_32k", "HyenaDNA_medium_160k",
    "HyenaDNA_medium_450k", "HyenaDNA_large_1m",
    "Caduceus_ph_1k_4L", "Caduceus_ph_131k_16L", "Caduceus_ps_131k_16L",
    "Evo2_1b_base",
    "NTv3_8M_pre", "NTv3_100M_pre", "NTv3_100M_post",
    "NTv3_650M_post", "NTv3_650M_post_131kb",
]

SCENARIOS = ["standard", "short", "long", "large_batch", "ultra_long"]
PHASE = "encode"

print(f"GB10: {gb10}")
print(f"H200: {h200}")
print(f"A100: {a100}")
print()

def gmean(xs):
    if not xs:
        return None
    import math
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

print(f"=== Speedup vs hf_native (best dnatok method) | phase={PHASE} ===\n")
for scenario in SCENARIOS:
    print(f"--- scenario: {scenario} ---")
    print(f"{'model':30s} | {'GB10':>10s} {'H200':>10s} {'A100':>10s}  hf_p50 (ms) GB10/H200/A100")
    per_plat_speedups = defaultdict(list)
    per_class = defaultdict(lambda: defaultdict(list))
    for model in MODELS_BPE + MODELS_LUT_KMER + MODELS_LUT_CHAR:
        cells = []
        hf_cells = []
        cls = "BPE" if model in MODELS_BPE else ("LUT-kmer" if model in MODELS_LUT_KMER else "LUT-char")
        for plat in ("GB10", "H200", "A100"):
            dn = best_dnatok(plat, model, scenario, PHASE)
            hfv = T.get((plat, model, scenario, PHASE, "hf_native"))
            if dn is None or hfv is None:
                cells.append("     -")
                hf_cells.append("     -")
            else:
                sp = hfv / dn
                cells.append(f"{sp:8.2f}x")
                hf_cells.append(f"{hfv:6.2f}")
                per_plat_speedups[plat].append(sp)
                per_class[cls][plat].append(sp)
        print(f"{model:30s} | {' '.join(cells)}   {' / '.join(hf_cells)}")
    # per-platform summary
    print()
    for plat in ("GB10", "H200", "A100"):
        xs = per_plat_speedups[plat]
        if xs:
            print(f"  {plat}: n={len(xs):3d}  median={statistics.median(xs):6.1f}x  gmean={gmean(xs):6.1f}x  min={min(xs):5.2f}x  max={max(xs):6.1f}x")
    print()
    for cls in ("BPE", "LUT-kmer", "LUT-char"):
        for plat in ("GB10", "H200", "A100"):
            xs = per_class[cls][plat]
            if xs:
                print(f"  [{cls:8s}] {plat}: n={len(xs):2d}  median={statistics.median(xs):6.1f}x  gmean={gmean(xs):6.1f}x")
    print()
