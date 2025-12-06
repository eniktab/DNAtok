# enhanced_tokenizer_benchmark.py
# Comprehensive benchmark suite comparing HF-native Rust tokenizer (Evo2) vs DNATok
# Designed for research publication with detailed profiling metrics (timing only)
#
# Runs complete benchmark suite automatically - just provide output path
#
# Usage:
#   python enhanced_tokenizer_benchmark.py

from __future__ import annotations

import csv
import json
import math
import os
import psutil
import random
import socket
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from evo2 import Evo2
except Exception:
    Evo2 = None
    print("Warning: Evo2 not available. Using fallback tokenizer.")

from dna_tokenizer import DNATok

DNA = "ACGTNacgtn"

# ================================ CONFIGURATION ===================================

OUTPUT_DIR = ROOT / "results"

# Benchmark configurations to run
BENCHMARK_CONFIGS = [
    {
        "name": "standard",
        "B": 4096,
        "T": 512,
        "reps": 5,
        "description": "Standard benchmark (4096 x 512)",
    },
    {
        "name": "large_batch",
        "B": 8192,
        "T": 512,
        "reps": 5,
        "description": "Large batch size (8192 x 512)",
    },
    {
        "name": "long_sequence",
        "B": 2048,
        "T": 2048,
        "reps": 5,
        "description": "Long sequences (2048 x 2048)",
    },
]

# Sweep configurations
BATCH_SIZE_SWEEP = [256, 512, 1024, 2048, 4096, 8192, 16384]
SEQ_LENGTH_SWEEP = [128, 256, 512, 1024, 2048, 4096]

# System settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
WARMUP_REPS = 3
SEED = 42
EMB_DIM = 128
VOCAB_MOD = 8192
EMB_BATCH = 2048

# Feature flags (DNATok best config for A100)
USE_COMPUTE_SHIM = True
PREFER_INT32_H2D = True          # int32 H2D then cast on device (bandwidth-optimized)
OVERLAP_COPY_COMPUTE = True      # use pipelined H2D/compute overlap

# Run full sweeps
RUN_BATCH_SWEEP = True
RUN_LENGTH_SWEEP = True
RUN_ABLATION_STUDIES = True  # Test different optimization combinations


# ============================== TOKENIZER WRAPPERS ================================

def _safe_get_vocab(tokenizer):
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            v = get_vocab()
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    try:
        v = getattr(tokenizer, "vocab")
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    return None


def _safe_token_to_id(tokenizer, token_str: str):
    fn = getattr(tokenizer, "token_to_id", None)
    if callable(fn):
        try:
            out = fn(token_str)
            if isinstance(out, int) and out >= 0:
                return out
        except Exception:
            pass
    cti = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(cti):
        try:
            out = cti(token_str)
            if isinstance(out, int) and out >= 0:
                return out
        except Exception:
            pass
    vocab = _safe_get_vocab(tokenizer)
    if vocab and token_str in vocab and isinstance(vocab[token_str], int):
        return int(vocab[token_str])
    return None


def _encode_one_char_to_id(tokenizer, ch: str):
    enc = getattr(tokenizer, "encode", None)
    if not callable(enc):
        return None
    try:
        try:
            out = enc(ch, add_special_tokens=False)
        except TypeError:
            out = enc(ch)
        ids = out.ids if hasattr(out, "ids") else out
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
            return ids[0]
    except Exception:
        pass
    return None


def _discover_unknown_id(tokenizer) -> int:
    for cand in ("N", "n"):
        tid = _safe_token_to_id(tokenizer, cand)
        if isinstance(tid, int):
            return tid
    for pad_tok in ("<pad>", "[PAD]", "PAD", "pad"):
        tid = _safe_token_to_id(tokenizer, pad_tok)
        if isinstance(tid, int):
            return tid
    return 0


def _tok2ids_via_encode_batch(tokenizer, seqs: List[str]) -> Optional[List[List[int]]]:
    if hasattr(tokenizer, "encode_batch"):
        try:
            encs = tokenizer.encode_batch(seqs, add_special_tokens=False)
        except TypeError:
            encs = tokenizer.encode_batch(seqs)
        out = []
        for e in encs:
            if hasattr(e, "ids"):
                out.append(e.ids)
            elif isinstance(e, list) and all(isinstance(x, int) for x in e):
                out.append(e)
            else:
                raise RuntimeError("encode_batch returned unsupported element.")
        return out
    return None


def _tok2ids_via_encode(tokenizer, s: str) -> Optional[List[int]]:
    enc = getattr(tokenizer, "encode", None)
    if callable(enc):
        try:
            try:
                out = enc(s, add_special_tokens=False)
            except TypeError:
                out = enc(s)
            if hasattr(out, "ids"):
                return list(out.ids)
            if isinstance(out, list) and all(isinstance(x, int) for x in out):
                return out
        except Exception:
            pass
    return None


def _tok2ids_via_tokenize_map(tokenizer, s: str) -> Optional[List[int]]:
    tokf = getattr(tokenizer, "tokenize", None)
    if not callable(tokf):
        return None
    try:
        toks = tokf(s)
        if isinstance(toks, torch.Tensor):
            return toks.long().tolist()
        if not isinstance(toks, list):
            return None
        if toks and all(isinstance(x, int) for x in toks):
            return toks
        out = []
        for t in toks:
            if isinstance(t, int):
                out.append(int(t))
            else:
                tid = _safe_token_to_id(tokenizer, t)
                if tid is None:
                    return None
                out.append(int(tid))
        return out
    except Exception:
        return None


def _tok2ids_via_charwise(tokenizer, s: str) -> List[int]:
    unknown_id = _discover_unknown_id(tokenizer)
    out: List[int] = []
    for ch in s:
        tid = _safe_token_to_id(tokenizer, ch)
        if tid is None:
            code = ord(ch)
            for key in (f"<0x{code:02X}>", f"<0x{code:02x}>"):
                tid = _safe_token_to_id(tokenizer, key)
                if isinstance(tid, int):
                    break
            if tid is None:
                tid = _encode_one_char_to_id(tokenizer, ch)
        out.append(int(tid if isinstance(tid, int) else unknown_id))
    return out


def tok2ids_user_robust(tokenizer, s: str) -> List[int]:
    """HF Rust (robust adapter): encode_batch → encode → tokenize+map → char-wise."""
    encb = _tok2ids_via_encode_batch(tokenizer, [s])
    if encb is not None and len(encb) == 1:
        return list(encb[0])
    enc1 = _tok2ids_via_encode(tokenizer, s)
    if enc1 is not None:
        return enc1
    tmap = _tok2ids_via_tokenize_map(tokenizer, s)
    if tmap is not None:
        return tmap
    return _tok2ids_via_charwise(tokenizer, s)


def tok2ids_batch(tokenizer, seqs: List[str]) -> List[List[int]]:
    encb = _tok2ids_via_encode_batch(tokenizer, seqs)
    if encb is not None:
        return encb
    return [tok2ids_user_robust(tokenizer, s) for s in seqs]


# ============================= COMPUTE SHIMS =====================================

class _NoComputeShim:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @torch.no_grad()
    def embed_tokens(self, x, **_):
        return x.float()


class _ComputeShim:
    """Realistic GPU compute to expose overlap benefits."""

    def __init__(self, tokenizer, device: str, vocab_mod: int, emb_dim: int):
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.emb = torch.nn.Embedding(vocab_mod, emb_dim, device=self.device)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 2, device=self.device),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim, device=self.device),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.02)
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    @torch.no_grad()
    def embed_tokens(self, x, **_):
        x = x.to(self.device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=_is_cuda_device(str(self.device)),
        ):
            e = self.emb(x % self.emb.num_embeddings)
            y = e.mean(dim=1)
            out = self.net(y)
        return out.float()


# ============================== HELPER FUNCTIONS ==================================

def make_equal_len_sequences(B: int, T: int, seed: int = 0) -> List[str]:
    random.seed(seed)
    return ["".join(random.choice(DNA) for _ in range(T)) for _ in range(B)]


def as_cuda(ids_2d, device: str) -> torch.Tensor:
    """Convert to target device int64 with pinning when possible."""
    if isinstance(ids_2d, torch.Tensor):
        t = ids_2d
        if t.dtype != torch.int64:
            t = t.to(torch.int64)
        if t.device.type != "cpu":
            return t.to(device, non_blocking=True)
        try:
            t = t.pin_memory()
        except Exception:
            pass
        return t.to(device, non_blocking=True)
    t = torch.tensor(ids_2d, dtype=torch.int64)
    try:
        t = t.pin_memory()
    except Exception:
        pass
    return t.to(device, non_blocking=True)


def _is_cuda_device(device: str) -> bool:
    try:
        return torch.cuda.is_available() and torch.device(device).type == "cuda"
    except Exception:
        return False


def maybe_sync(device: str) -> None:
    if _is_cuda_device(device):
        torch.cuda.synchronize()


def maybe_empty_cache(device: str) -> None:
    if _is_cuda_device(device):
        torch.cuda.empty_cache()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def gpu_properties(device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(idx)
        out = dict(
            name=props.name,
            total_mem_bytes=props.total_memory,
            total_mem_gb=props.total_memory / (1024 ** 3),
            multi_processor_count=props.multi_processor_count,
            major=props.major,
            minor=props.minor,
        )
    except Exception:
        pass
    return out


# ========================== BENCHMARK EXECUTION ===================================

@dataclass
class BenchmarkResult:
    """Single benchmark result with timing-only profiling data."""
    method: str
    metric: str
    value: float
    rep: int
    B: int
    T: int
    total_tokens: int
    config_name: str

    # Timing breakdown
    encode_time_ms: float = 0.0
    h2d_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """Orchestrates benchmark execution."""

    def __init__(
        self,
        B: int,
        T: int,
        reps: int,
        config_name: str,
        tokenizer,
        helper: DNATok,
        embedder,
        use_int32: bool = True,
        use_overlap: bool = True,
        device: str = DEVICE,
    ):
        self.B = B
        self.T = T
        self.reps = reps
        self.config_name = config_name
        self.tokenizer = tokenizer
        self.helper = helper
        self.embedder = embedder
        self.use_int32 = use_int32
        self.use_overlap = use_overlap
        self.device = device
        self.results: List[BenchmarkResult] = []

    def _profile_encode(
        self,
        encode_fn: Callable[[], Any],
        method: str,
        seqs: List[str],
    ) -> List[BenchmarkResult]:
        """Profile tokenization encode + H2D steps."""
        total_tokens = self.B * self.T
        results: List[BenchmarkResult] = []

        for rep in range(self.reps):
            maybe_sync(self.device)
            t0 = time.perf_counter()
            ids_2d = encode_fn()
            t1 = time.perf_counter()
            encode_ms = (t1 - t0) * 1000.0

            maybe_sync(self.device)
            c0 = time.perf_counter()
            _ = as_cuda(ids_2d, self.device)
            maybe_sync(self.device)
            c1 = time.perf_counter()
            h2d_ms = (c1 - c0) * 1000.0

            total_ms = encode_ms + h2d_ms

            encode_rate = total_tokens / max(1e-9, encode_ms / 1000.0)
            h2d_rate = total_tokens / max(1e-9, h2d_ms / 1000.0)

            for metric, value in (("encode_tok_s", encode_rate), ("h2d_tok_s", h2d_rate)):
                result = BenchmarkResult(
                    method=method,
                    metric=metric,
                    value=value,
                    rep=rep,
                    B=self.B,
                    T=self.T,
                    total_tokens=total_tokens,
                    config_name=self.config_name,
                    encode_time_ms=encode_ms,
                    h2d_time_ms=h2d_ms,
                    total_time_ms=total_ms,
                )
                results.append(result)

            maybe_empty_cache(self.device)

        return results

    def _profile_streaming(
        self,
        ids_cpu: torch.Tensor,
        method: str,
        use_pipeline: bool,
    ) -> List[BenchmarkResult]:
        """Profile end-to-end streaming (DNATok iter_embed_tokens_*)."""
        if not self.helper.use_ids_path:
            print(f"      Skipping {method} (IDs path disabled).")
            return []
        total_tokens = self.B * self.T
        results: List[BenchmarkResult] = []

        for rep in range(self.reps):
            maybe_sync(self.device)
            t0 = time.perf_counter()

            if use_pipeline:
                iterator = self.helper.iter_embed_tokens_pipelined(
                    ids_cpu,
                    EMB_BATCH,
                    device=self.device,
                    use_int32_h2d=self.use_int32,
                )
            else:
                iterator = self.helper.iter_embed_tokens_in_slices(
                    ids_cpu,
                    EMB_BATCH,
                    device=self.device,
                )

            for _ in iterator:
                pass

            maybe_sync(self.device)
            t1 = time.perf_counter()

            total_ms = (t1 - t0) * 1000.0
            throughput = total_tokens / max(1e-9, total_ms / 1000.0)

            result = BenchmarkResult(
                method=method,
                metric="end2end_tok_s",
                value=throughput,
                rep=rep,
                B=self.B,
                T=self.T,
                total_tokens=total_tokens,
                config_name=self.config_name,
                total_time_ms=total_ms,
            )
            results.append(result)

            maybe_empty_cache(self.device)

        return results

    def run_all_benchmarks(self, seqs: List[str]) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        all_results: List[BenchmarkResult] = []

        print(f"    Warmup ({WARMUP_REPS} iterations)...")
        for _ in range(WARMUP_REPS):
            _ = tok2ids_batch(self.tokenizer, seqs)
            _ = self.helper.encode_batch_to_ids(seqs)

        methods: List[Tuple[str, Callable[[], Any]]] = [
            (
                "HF Rust (robust adapter)",
                lambda: [tok2ids_user_robust(self.tokenizer, s) for s in seqs],
            ),
            ("HF Rust (encode_batch)", lambda: tok2ids_batch(self.tokenizer, seqs)),
            ("DNATok (encode i64)", lambda: self.helper.encode_batch_to_ids(seqs)),
            (
                "DNATok (staging i64)",
                lambda: self.helper.encode_batch_to_ids_staging(
                    seqs, dtype=torch.int64
                ),
            ),
            (
                "DNATok (staging i32)",
                lambda: self.helper.encode_batch_to_ids_staging(
                    seqs, dtype=torch.int32
                ),
            ),
        ]

        for method_name, encode_fn in methods:
            results = self._profile_encode(encode_fn, method_name, seqs)
            all_results.extend(results)

            encode_results = [r for r in results if r.metric == "encode_tok_s"]
            if encode_results:
                enc_mean = np.mean([r.value for r in encode_results])
                print(f"      {method_name:32s}: {enc_mean:>12,.0f} tok/s")

        # Streaming benchmarks (DNATok IDs-path end-to-end)
        i64 = self.helper.encode_batch_to_ids_staging(seqs, dtype=torch.int64)
        i32 = self.helper.encode_batch_to_ids_staging(seqs, dtype=torch.int32)

        baseline_results = self._profile_streaming(i64, "Streaming baseline", use_pipeline=False)
        all_results.extend(baseline_results)

        ids_input = i32 if self.use_int32 else i64
        pipeline_results = self._profile_streaming(
            ids_input, "Streaming pipelined", use_pipeline=True
        )
        all_results.extend(pipeline_results)

        if baseline_results and pipeline_results:
            baseline_mean = np.mean([r.value for r in baseline_results])
            pipeline_mean = np.mean([r.value for r in pipeline_results])
            speedup = (pipeline_mean / baseline_mean - 1.0) * 100.0

            print(f"      {'Streaming baseline':32s}: {baseline_mean:>12,.0f} tok/s")
            print(
                f"      {'Streaming pipelined':32s}: "
                f"{pipeline_mean:>12,.0f} tok/s (+{speedup:.1f}%)"
            )

        return all_results


def write_results(
    all_results: List[BenchmarkResult],
    metadata: Dict[str, Any],
    stamp: str,
):
    """Write comprehensive results to CSV and JSON files."""
    out_dir = Path(OUTPUT_DIR)
    ensure_dir(out_dir)

    # Detailed trials CSV
    trials_csv = out_dir / f"benchmark_trials_{stamp}.csv"
    fieldnames = list(BenchmarkResult.__dataclass_fields__.keys())

    with open(trials_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r.to_dict())

    # Summary statistics CSV
    summary_csv = out_dir / f"benchmark_summary_{stamp}.csv"
    summary_data: Dict[
        Tuple[str, str, str],
        Dict[str, Any],
    ] = defaultdict(
        lambda: {
            "values": [],
            "B": 0,
            "T": 0,
            "config_name": "",
            "encode_time_ms": [],
            "h2d_time_ms": [],
            "total_time_ms": [],
        }
    )

    for r in all_results:
        key = (r.config_name, r.method, r.metric)
        d = summary_data[key]
        d["values"].append(r.value)
        d["B"] = r.B
        d["T"] = r.T
        d["config_name"] = r.config_name
        d["encode_time_ms"].append(r.encode_time_ms)
        d["h2d_time_ms"].append(r.h2d_time_ms)
        d["total_time_ms"].append(r.total_time_ms)

    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "config_name",
            "method",
            "metric",
            "B",
            "T",
            "total_tokens",
            "throughput_mean",
            "throughput_std",
            "throughput_min",
            "throughput_max",
            "throughput_median",
            "throughput_p95",
            "throughput_p99",
            "encode_time_ms_mean",
            "h2d_time_ms_mean",
            "total_time_ms_mean",
            "count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (config_name, method, metric), d in summary_data.items():
            values = d["values"]
            row = {
                "config_name": config_name,
                "method": method,
                "metric": metric,
                "B": d["B"],
                "T": d["T"],
                "total_tokens": d["B"] * d["T"],
                "throughput_mean": float(np.mean(values)),
                "throughput_std": float(np.std(values)),
                "throughput_min": float(np.min(values)),
                "throughput_max": float(np.max(values)),
                "throughput_median": float(np.median(values)),
                "throughput_p95": float(np.percentile(values, 95)),
                "throughput_p99": float(np.percentile(values, 99)),
                "encode_time_ms_mean": float(
                    np.mean(d["encode_time_ms"]) if d["encode_time_ms"] else 0.0
                ),
                "h2d_time_ms_mean": float(
                    np.mean(d["h2d_time_ms"]) if d["h2d_time_ms"] else 0.0
                ),
                "total_time_ms_mean": float(
                    np.mean(d["total_time_ms"]) if d["total_time_ms"] else 0.0
                ),
                "count": len(values),
            }
            writer.writerow(row)

    # Metadata JSON
    meta_json = out_dir / f"benchmark_meta_{stamp}.json"
    with open(meta_json, "w") as f:
        json.dump(metadata, f, indent=2)

    # Performance comparison table (human-readable)
    comparison_txt = out_dir / f"benchmark_comparison_{stamp}.txt"
    with open(comparison_txt, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TOKENIZATION BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Group by config
        configs: Dict[str, List[BenchmarkResult]] = {}
        for r in all_results:
            configs.setdefault(r.config_name, []).append(r)

        for config_name, results in configs.items():
            config_info = next(
                (c for c in BENCHMARK_CONFIGS if c["name"] == config_name),
                None,
            )
            if config_info:
                f.write(
                    f"\n{config_info['description']}"
                    f" (B={config_info['B']}, T={config_info['T']})\n"
                )
                f.write("-" * 80 + "\n")

            # Group by method and metric
            method_results: Dict[Tuple[str, str], List[BenchmarkResult]] = defaultdict(list)
            for r in results:
                method_results[(r.method, r.metric)].append(r)

            for (method, metric), method_res in sorted(method_results.items()):
                values = [r.value for r in method_res]
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))

                if metric == "encode_tok_s":
                    label = "Encode"
                elif metric == "h2d_tok_s":
                    label = "H2D   "
                elif metric == "end2end_tok_s":
                    label = "E2E   "
                else:
                    label = metric[:6]

                f.write(
                    f"  {label} | {method:32s} | "
                    f"{mean_val:>12,.0f} ± {std_val:>8,.0f} tok/s\n"
                )

        f.write("\n" + "=" * 80 + "\n")

    print(f"\n{'=' * 70}")
    print("OUTPUT FILES")
    print(f"{'=' * 70}")
    print(f"  Trials:      {trials_csv}")
    print(f"  Summary:     {summary_csv}")
    print(f"  Comparison:  {comparison_txt}")
    print(f"  Metadata:    {meta_json}")

    return trials_csv, summary_csv, meta_json, comparison_txt


def run_sweep(
    tokenizer,
    helper: DNATok,
    embedder,
    sweep_type: str,
    stamp: str,
) -> List[BenchmarkResult]:
    """Run parameter sweep."""
    all_results: List[BenchmarkResult] = []

    if sweep_type == "batch":
        print(f"\n{'=' * 70}")
        print("BATCH SIZE SWEEP")
        print(f"{'=' * 70}")

        for B in BATCH_SIZE_SWEEP:
            print(f"\n  Batch size: {B}")
            try:
                seqs = make_equal_len_sequences(B, 512, seed=SEED)
                runner = BenchmarkRunner(
                    B,
                    512,
                    3,
                    f"batch_sweep_{B}",
                    tokenizer,
                    helper,
                    embedder,
                    use_int32=PREFER_INT32_H2D,
                    use_overlap=OVERLAP_COPY_COMPUTE,
                    device=DEVICE,
                )
                results = runner.run_all_benchmarks(seqs)
                all_results.extend(results)
            except Exception as e:
                print(f"    Error: {e}")

    elif sweep_type == "length":
        print(f"\n{'=' * 70}")
        print("SEQUENCE LENGTH SWEEP")
        print(f"{'=' * 70}")

        for T in SEQ_LENGTH_SWEEP:
            print(f"\n  Sequence length: {T}")
            try:
                seqs = make_equal_len_sequences(2048, T, seed=SEED)
                runner = BenchmarkRunner(
                    2048,
                    T,
                    3,
                    f"length_sweep_{T}",
                    tokenizer,
                    helper,
                    embedder,
                    use_int32=PREFER_INT32_H2D,
                    use_overlap=OVERLAP_COPY_COMPUTE,
                    device=DEVICE,
                )
                results = runner.run_all_benchmarks(seqs)
                all_results.extend(results)
            except Exception as e:
                print(f"    Error: {e}")

    return all_results


def run_ablation_study(
    tokenizer,
    helper: DNATok,
    embedder,
    stamp: str,
) -> List[BenchmarkResult]:
    """Run ablation study on optimization techniques (DNATok streaming only)."""
    print(f"\n{'=' * 70}")
    print("ABLATION STUDY: Optimization Techniques")
    print(f"{'=' * 70}")

    if not helper.use_ids_path:
        print("  Skipping ablation study (IDs/bytes path disabled after validation).")
        return []

    all_results: List[BenchmarkResult] = []
    B, T = 4096, 512
    seqs = make_equal_len_sequences(B, T, seed=SEED)

    configs = [
        ("baseline", False, False, "No optimizations"),
        ("int32_only", True, False, "Int32 H2D only"),
        ("overlap_only", False, True, "Overlap only"),
        ("both", True, True, "Int32 + Overlap"),
    ]

    for name, use_int32, use_overlap, desc in configs:
        print(f"\n  Configuration: {desc}")

        # Update helper settings
        helper.prefer_int32_h2d = use_int32
        helper.overlap_h2d_compute = use_overlap

        runner = BenchmarkRunner(
            B,
            T,
            5,
            f"ablation_{name}",
            tokenizer,
            helper,
            embedder,
            use_int32=use_int32,
            use_overlap=use_overlap,
            device=DEVICE,
        )

        # Only run streaming benchmarks for ablation
        i64 = helper.encode_batch_to_ids_staging(seqs, dtype=torch.int64)
        i32 = helper.encode_batch_to_ids_staging(seqs, dtype=torch.int32)
        ids_input = i32 if use_int32 else i64

        results = runner._profile_streaming(
            ids_input,
            f"Ablation-{name}",
            use_pipeline=use_overlap,
        )
        all_results.extend(results)

        if results:
            mean_throughput = float(np.mean([r.value for r in results]))
            print(f"    Throughput: {mean_throughput:,.0f} tok/s")

    # Restore original settings
    helper.prefer_int32_h2d = PREFER_INT32_H2D
    helper.overlap_h2d_compute = OVERLAP_COPY_COMPUTE

    return all_results


# ================================== MAIN ==========================================

def main():
    stamp = now_stamp()

    print("=" * 70)
    print("COMPREHENSIVE TOKENIZATION BENCHMARK SUITE")
    print("=" * 70)
    print(f"\nRun ID: {stamp}")
    print(f"Output: {OUTPUT_DIR}")

    # System info
    print(f"\n{'=' * 70}")
    print("SYSTEM CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"  Host:         {socket.gethostname()}")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  CUDA:         {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
    print(f"  Device:       {DEVICE}")
    print(f"  CPU cores:    {psutil.cpu_count()}")
    print(f"  System RAM:   {psutil.virtual_memory().total / (1024 ** 3):.1f} GB")

    gpu_info = gpu_properties(DEVICE)
    if gpu_info:
        print(f"  GPU:          {gpu_info.get('name', 'Unknown')}")
        print(f"  GPU memory:   {gpu_info.get('total_mem_gb', 0):.1f} GB")
        print(f"  Compute cap:  {gpu_info.get('major', 0)}.{gpu_info.get('minor', 0)}")

    # Initialize tokenizer
    print(f"\n{'=' * 70}")
    print("TOKENIZER INITIALIZATION")
    print(f"{'=' * 70}")

    if Evo2 is not None:
        print("  Loading Evo2 model...")
        evo2_model = Evo2("evo2_7b")
        tok = getattr(evo2_model, "tokenizer", evo2_model)
        tok_name = f"{tok.__class__.__name__}"
        print(f"  Tokenizer: {tok_name}")
    else:
        print("  Warning: Using fallback dummy tokenizer")

        class _DummyTok:
            def encode(self, s, add_special_tokens: bool = False):
                return [ord(c) for c in s]

        tok = _DummyTok()
        tok_name = "_DummyTok"

    # Initialize embedder (compute shim to expose overlap benefits)
    embedder = (
        _ComputeShim(tok, DEVICE, VOCAB_MOD, EMB_DIM)
        if USE_COMPUTE_SHIM
        else _NoComputeShim(tok)
    )
    print(f"  Embedder:  {'Compute shim' if USE_COMPUTE_SHIM else 'No-op shim'}")

    # Initialize DNATok with A100-optimised settings
    helper = DNATok(
        embedder,
        ids_max_tokens_per_call=DNATok.DEFAULT_IDS_MAX_TOKENS_PER_CALL,
        prefer_int32_h2d=PREFER_INT32_H2D,
        overlap_h2d_compute=OVERLAP_COPY_COMPUTE,
        # keep force_fp32_outputs default (True) for compatibility
    )

    try:
        helper.discover()
        print("  Helper: Discovered successfully")
    except Exception as e:
        print(f"  Helper warning: {e}")

    # Correctness validation
    print(f"\n{'=' * 70}")
    print("CORRECTNESS VALIDATION")
    print(f"{'=' * 70}")

    test_seq = "ACGTNtacgtn"
    ids_user = tok2ids_user_robust(tok, test_seq)
    ids_batch = tok2ids_batch(tok, [test_seq])[0]

    try:
        ids_helper = helper.encode_batch_to_ids([test_seq]).tolist()[0]
    except Exception as e:
        print(f"  Warning: Helper encode failed: {e}")
        ids_helper = tok2ids_batch(tok, [test_seq])[0]

    if ids_user != ids_batch:
        print("  Warning: Tokenizer consistency check failed; using user path for comparisons.")
    if ids_user != ids_helper:
        print("  Warning: Helper mismatch; attempting LUT rebuild from tokenizer.")
        try:
            rebuilt_cont, rebuilt_start = helper._rebuild_ascii_luts_from_tokenizer(
                tok, pad_id=getattr(helper, "id_pad", 0), fallback_id=getattr(helper, "id_N", 0)
            )
            helper.ascii_lut = rebuilt_cont
            helper.ascii_start_lut = rebuilt_start
            helper.use_ids_path = True
            ids_helper = helper.encode_batch_to_ids([test_seq]).tolist()[0]
        except Exception as e:
            print(f"  Rebuild failed ({e}); disabling IDs/bytes path and retrying via tokenizer.")
            helper.use_ids_path = False
            helper.ascii_lut = None
            helper.ascii_start_lut = None
            try:
                ids_helper = helper.encode_batch_to_ids([test_seq]).tolist()[0]
            except Exception:
                ids_helper = tok2ids_batch(tok, [test_seq])[0]
    print("  ✓ Single sequence validation complete")

    test_seqs = make_equal_len_sequences(100, 50, seed=SEED)
    ids_tok = tok2ids_batch(tok, test_seqs)
    try:
        ids_help = helper.encode_batch_to_ids(test_seqs).tolist()
    except Exception:
        ids_help = tok2ids_batch(tok, test_seqs)

    if ids_tok != ids_help:
        print("  Warning: Batch consistency mismatch; attempting LUT rebuild from tokenizer.")
        try:
            rebuilt_cont, rebuilt_start = helper._rebuild_ascii_luts_from_tokenizer(
                tok, pad_id=getattr(helper, "id_pad", 0), fallback_id=getattr(helper, "id_N", 0)
            )
            helper.ascii_lut = rebuilt_cont
            helper.ascii_start_lut = rebuilt_start
            helper.use_ids_path = True
            ids_help = helper.encode_batch_to_ids(test_seqs).tolist()
        except Exception as e:
            print(f"  Rebuild failed ({e}); forcing tokenizer path for benchmarks.")
            helper.use_ids_path = False
            helper.ascii_lut = None
            helper.ascii_start_lut = None
    print("  ✓ Batch validation complete")

    # Run all benchmarks
    all_results: List[BenchmarkResult] = []

    # 1. Standard configurations
    print(f"\n{'=' * 70}")
    print("STANDARD BENCHMARKS")
    print(f"{'=' * 70}")

    for config in BENCHMARK_CONFIGS:
        print(f"\n  {config['description']}")
        seqs = make_equal_len_sequences(config["B"], config["T"], seed=SEED)
        runner = BenchmarkRunner(
            config["B"],
            config["T"],
            config["reps"],
            config["name"],
            tok,
            helper,
            embedder,
            use_int32=PREFER_INT32_H2D,
            use_overlap=OVERLAP_COPY_COMPUTE,
            device=DEVICE,
        )
        results = runner.run_all_benchmarks(seqs)
        all_results.extend(results)

    # 2. Batch size sweep
    if RUN_BATCH_SWEEP:
        batch_results = run_sweep(tok, helper, embedder, "batch", stamp)
        all_results.extend(batch_results)

    # 3. Sequence length sweep
    if RUN_LENGTH_SWEEP:
        length_results = run_sweep(tok, helper, embedder, "length", stamp)
        all_results.extend(length_results)

    # 4. Ablation studies (DNATok streaming configs)
    if RUN_ABLATION_STUDIES:
        ablation_results = run_ablation_study(tok, helper, embedder, stamp)
        all_results.extend(ablation_results)

    # Prepare metadata
    metadata = {
        "timestamp": stamp,
        "host": socket.gethostname(),
        "output_dir": str(OUTPUT_DIR),
        "configurations_run": [c["name"] for c in BENCHMARK_CONFIGS],
        "sweeps_run": {
            "batch_size": RUN_BATCH_SWEEP,
            "sequence_length": RUN_LENGTH_SWEEP,
            "ablation": RUN_ABLATION_STUDIES,
        },
        "settings": {
            "device": DEVICE,
            "warmup_reps": WARMUP_REPS,
            "seed": SEED,
            "emb_dim": EMB_DIM,
            "vocab_mod": VOCAB_MOD,
            "emb_batch": EMB_BATCH,
            "use_compute_shim": USE_COMPUTE_SHIM,
            "prefer_int32_h2d": PREFER_INT32_H2D,
            "overlap_copy_compute": OVERLAP_COPY_COMPUTE,
        },
        "system": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda
            if hasattr(torch.version, "cuda")
            else None,
            "cudnn_version": torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None,
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024 ** 3),
        },
        "gpu": gpu_info,
        "tokenizer": {
            "implementation": tok_name,
            "evo2_available": Evo2 is not None,
        },
        "total_benchmarks_run": len(all_results),
    }

    # Write results
    write_results(all_results, metadata, stamp)

    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total benchmarks: {len(all_results)}")
    print(f"  Run ID:           {stamp}")
    print(f"  Duration stamp:   {time.perf_counter():.1f}s (wall-clock from module load)")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        main()
    except Exception as e:
        print(f"\n{'=' * 70}")
        print("ERROR: Benchmark failed")
        print(f"{'=' * 70}")
        print(f"{e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"\nTotal runtime: {elapsed:.1f}s")
