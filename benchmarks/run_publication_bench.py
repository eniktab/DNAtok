#!/usr/bin/env python3
"""DNAtok publication benchmark — clean, reproducible, single-file.

Goal: produce a *paired* JSON+CSV result set in one run, so reviewers
running the script see the same numbers in tables and figures.

Coverage:
  * All registered HF models that resolve locally (NTv2/v3, HyenaDNA,
    Caduceus, DNABERT-2, GENA-LM, METAGENE-1, Evo2 1b).
  * For each model: encode throughput (HF baseline vs DNAtok variants),
    H2D bandwidth, end-to-end tokenize+embed wall time.
  * DNAtok variants (independently ablatable):
        default        - production path
        +uint8_ids     - uint8 token ids across PCIe
        +pack_2bit     - 2-bit packed nucleotides across PCIe
        +cuda_graph    - CUDA graph capture/replay
        +fused_triton  - Triton fused tokenize+gather
        +all           - all of the above (where compatible)
  * GPU baseline: cudf.Series.str.character_tokenize (RAPIDS nvtext).

Outputs:
  results_hpc/publication/YYYYmmdd-HHMMSS/
      bench_full.json      - every individual trial
      bench_summary.csv    - one row per (model, scenario, method)
      meta.json            - system + git provenance
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# IMPORTANT: must import dnatok_compat *before* transformers to apply patches.
import dnatok_compat  # noqa: F401

from src.dna_tokenizer import DNATok
from benchmarks.model_registry import MODEL_SPECS, resolve_model_path, ModelSpec
from benchmarks.tokenizer_adapters import TokenizerAdapter, load_hf_tokenizer
from benchmarks.gpu_tokenizer_baselines import (
    have_cudf,
    cudf_character_tokenize,
    cudf_ngrams_tokenize,
    load_gputok,
    gputok_tokenize,
)
try:
    from src.gputok_bpe_backend import GPUTokBPEBackend
except Exception as _gputok_backend_exc:  # pragma: no cover
    GPUTokBPEBackend = None  # type: ignore[assignment]
    logger = logging.getLogger("publication_bench")
    logger.warning("GPUTOK BPE backend unavailable: %s", _gputok_backend_exc)

logger = logging.getLogger("publication_bench")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    batch_size: int
    seq_len: int
    emb_batch: int = 32

    @property
    def tokens_per_iter(self) -> int:
        return self.batch_size * self.seq_len


PUBLICATION_SCENARIOS = (
    Scenario("standard",       batch_size=32,  seq_len=1_000),
    Scenario("short",          batch_size=128, seq_len=128),
    Scenario("long",           batch_size=8,   seq_len=8_000),
    Scenario("large_batch",    batch_size=256, seq_len=1_000),
    Scenario("ultra_long",     batch_size=2,   seq_len=32_000),
)

QUICK_SCENARIOS = (
    Scenario("standard",       batch_size=32,  seq_len=1_000),
    Scenario("short",          batch_size=128, seq_len=128),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sequences(b: int, t: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choices("ACGT", k=t)) for _ in range(b)]


def cuda_sync(device: torch.device | str) -> None:
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class Trial:
    method: str
    model: str
    scenario: str
    batch_size: int
    seq_len: int
    tokens_per_iter: int
    phase: str
    wall_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    tokens_per_sec: float
    seqs_per_sec: float
    bytes_h2d: int                 # bytes copied host->device per iteration (0 if N/A)
    h2d_gbps: float                # achieved H2D bandwidth, GB/s (0 if N/A)
    peak_memory_mb: float          # torch.cuda.max_memory_allocated delta, MB
    energy_j: float                # estimated energy per iteration, J (0 if unavailable)
    trials: int
    warmup: int
    durations_ms: list[float] = field(default_factory=list)  # raw per-iter timings for CDF
    notes: str = ""


def time_callable(
    fn: Callable[[], None],
    *,
    device: torch.device | str,
    warmup: int = 3,
    iters: int = 10,
    keep_durations: bool = False,
) -> tuple[float, float, float, float, float, float, list[float]]:
    """Time fn() across `iters`. Returns (p50, p95, p99, min, max, mean) in ms,
    plus the raw duration array if keep_durations=True (else empty list).
    """
    for _ in range(warmup):
        fn()
    cuda_sync(device)

    durations: list[float] = []
    for _ in range(iters):
        cuda_sync(device)
        t0 = time.perf_counter()
        fn()
        cuda_sync(device)
        durations.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(durations)
    p50 = float(np.median(arr))
    p95 = float(np.quantile(arr, 0.95))
    p99 = float(np.quantile(arr, 0.99))
    mn = float(arr.min())
    mx = float(arr.max())
    mean = float(arr.mean())
    raw = list(durations) if keep_durations else []
    return p50, p95, p99, mn, mx, mean, raw


# ---------------------------------------------------------------------------
# NVML for energy
# ---------------------------------------------------------------------------

_nvml_handle = None
_nvml_init_attempted = False

def get_nvml_handle():
    global _nvml_handle, _nvml_init_attempted
    if _nvml_init_attempted:
        return _nvml_handle
    _nvml_init_attempted = True
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        _nvml_handle = None
    return _nvml_handle


def power_watts() -> float | None:
    h = get_nvml_handle()
    if h is None:
        return None
    try:
        import pynvml
        return float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0
    except Exception:
        return None


def trial(
    method: str,
    model_name: str,
    sc: Scenario,
    phase: str,
    fn: Callable[[], None],
    *,
    device: torch.device | str,
    warmup: int = 3,
    iters: int = 10,
    notes: str = "",
    bytes_h2d: int = 0,
    keep_durations: bool = False,
) -> Trial:
    # Reset peak memory tracker
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    # Sample power once before and once after for energy proxy
    power_before = power_watts()

    p50, p95, p99, mn, mx, mean, raw = time_callable(
        fn, device=device, warmup=warmup, iters=iters, keep_durations=keep_durations
    )
    tps = sc.tokens_per_iter / (p50 / 1000.0)
    sps = sc.batch_size / (p50 / 1000.0)

    power_after = power_watts()
    avg_power = None
    if power_before is not None and power_after is not None:
        avg_power = 0.5 * (power_before + power_after)
    energy_j = (avg_power * (mean / 1000.0)) if avg_power is not None else 0.0

    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() - mem_before
    peak_mem_mb = max(0.0, peak_mem / (1024 * 1024))

    # Achieved H2D bandwidth: bytes / mean_time
    h2d_gbps = 0.0
    if bytes_h2d > 0 and mean > 0:
        h2d_gbps = (bytes_h2d / 1e9) / (mean / 1000.0)

    return Trial(
        method=method,
        model=model_name,
        scenario=sc.name,
        batch_size=sc.batch_size,
        seq_len=sc.seq_len,
        tokens_per_iter=sc.tokens_per_iter,
        phase=phase,
        wall_ms=mean,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        min_ms=mn,
        max_ms=mx,
        tokens_per_sec=tps,
        seqs_per_sec=sps,
        bytes_h2d=bytes_h2d,
        h2d_gbps=h2d_gbps,
        peak_memory_mb=peak_mem_mb,
        energy_j=energy_j,
        trials=iters,
        warmup=warmup,
        durations_ms=raw,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Mock embedder for E2E benchmarks (real model loading optional)
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Stand-in embedder so E2E timings measure tokenization + H2D + a
    representative embedding op without dragging in the full model.

    Uses a real torch.nn.Embedding of the discovered vocab so the gather
    cost is realistic for production inference.
    """
    def __init__(self, tokenizer, hidden_size: int = 256, device: torch.device | str = "cuda"):
        self.tokenizer = tokenizer
        self.vocab_size = max(int(getattr(tokenizer, "vocab_size", 0) or 0), int(len(tokenizer)) if hasattr(tokenizer, "__len__") else 0, 1024)
        self.embed_table = torch.nn.Embedding(self.vocab_size, hidden_size).to(device)
        self.embed_table.eval()
        self.model_max_length = getattr(tokenizer, "model_max_length", 32_768) or 32_768
        if self.model_max_length > 1_000_000:
            self.model_max_length = 32_768

    @torch.no_grad()
    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_table(ids)


# ---------------------------------------------------------------------------
# Per-model benchmark
# ---------------------------------------------------------------------------

def bytes_h2d_for_method(method: str, sc: Scenario) -> int:
    """Best estimate of bytes transferred host->device per iteration."""
    T = sc.seq_len
    B = sc.batch_size
    if method == "hf_native":
        return B * T * 8  # int64 token ids, padded to T
    if method in ("dnatok_default", "dnatok_default_strict",
                  "dnatok_uint8", "dnatok_cuda_graph"):
        return B * T * 1  # uint8 ASCII
    if method in ("dnatok_pack2bit", "dnatok_pack2bit_uint8"):
        return B * ((T + 3) // 4)  # 2-bit packed; uint8 only changes device-side dtype
    if method == "dnatok_fused_triton":
        return B * T * 1
    if method == "dnatok_dna_kernel_v3":
        return sc.batch_size * sc.seq_len
    if method == "dnatok_gputok_bpe":
        return B * T * 1  # raw bytes (one ASCII char per base) shipped to GPUTOK
    if method == "dnatok_dna_kernel":
        return B * T * 1  # raw bytes shipped to our DNA-specialised kernel
    if method.startswith("cudf_"):
        return B * T * 1  # string column, lower-bound estimate
    if method.startswith("gputok_"):
        return B * T * 1  # GPUTOK pre-encodes bytes on host before H2D
    return 0


def _adjust_seqlen(sc: Scenario, k: Optional[int]) -> Scenario:
    """K-mer tokenizers (e.g. NTv2 with k=6) require sequence lengths to be
    a multiple of k. Round up the scenario's seq_len just enough to keep
    benchmark numbers comparable across models without breaking the fast
    path. Round to next multiple."""
    if not k or k <= 1:
        return sc
    if sc.seq_len % k == 0:
        return sc
    new_T = sc.seq_len + (k - sc.seq_len % k)
    from dataclasses import replace
    return replace(sc, seq_len=new_T)


def encode_phase(
    name: str,
    sc: Scenario,
    adapter: TokenizerAdapter,
    dna: DNATok,
    device: torch.device,
    *,
    warmup: int,
    iters: int,
) -> list[Trial]:
    sc = _adjust_seqlen(sc, dna.kmer_k)
    seqs = make_sequences(sc.batch_size, sc.seq_len)
    out: list[Trial] = []

    # HF native (CPU tokenize → already there) — always measured.
    def hf_encode():
        adapter.encode_batch(seqs, add_special_tokens=False, padding=True)
    out.append(trial("hf_native", name, sc, "encode", hf_encode, device=device, warmup=warmup, iters=iters,
                     bytes_h2d=bytes_h2d_for_method("hf_native", sc)))

    # GPU BPE paths for genomic BPE tokenizers (DNABERT-2, GENA-LM,
    # METAGENE-1). Both variants are bit-identical to HF native — see
    # tests/test_gputok_bpe_backend.py.
    #
    #   dnatok_gputok_bpe  — GPUTOK BlockBPE-baseline kernel (third-party).
    #                        HF-exact for ≤chunk_tokens=2048; longer
    #                        sequences are routed to HF.
    #   dnatok_dna_kernel  — our DNA-specialised CUDA kernel:
    #                        global-memory working buffer (no chunk
    #                        limit), tensorised I/O (no Python list
    #                        round-trip), direct byte input. HF-exact for
    #                        ≤fast_limit=2048; long sequences fall back
    #                        because the merge loop is O(T²) — HF Rust's
    #                        priority queue wins there.
    if GPUTokBPEBackend is not None and GPUTokBPEBackend.is_supported(adapter.tokenizer):
        for eng, method_name in (("gputok",    "dnatok_gputok_bpe"),
                                  ("dnatok",    "dnatok_dna_kernel"),
                                  ("dnatok_v3", "dnatok_dna_kernel_v3")):
            try:
                backend = GPUTokBPEBackend(adapter.tokenizer, engine=eng)
                def _enc(b=backend):
                    b.encode_batch(seqs, device=device)
                out.append(trial(method_name, name, sc, "encode",
                                 _enc, device=device,
                                 warmup=warmup, iters=iters,
                                 bytes_h2d=bytes_h2d_for_method(method_name, sc),
                                 notes=f"engine={eng}; HF-exact (validated)"))
            except Exception as e:
                logger.warning("%s encode failed on %s/%s: %s", method_name, name, sc.name, e)

    # DNAtok's specialised paths only apply when the tokenizer is ASCII-LUT
    # or k-mer LUT compatible. For pure BPE tokenizers we already added the
    # GPUTOK-backed entry above; nothing else applies.
    if not dna.use_ids_path:
        return out

    # DNAtok default (ascii/k-mer bytes → device → LUT)
    def dn_default():
        if dna.ascii_lut is not None:
            ascii_cpu = dna.encode_batch_to_ascii_bytes(seqs)
            _ = dna._map_ascii_bytes_to_ids_cuda(ascii_cpu, device)
        else:
            ascii_cpu = dna.encode_batch_to_ascii_bytes(seqs)
            _ = dna._map_kmer_bytes_to_ids_cuda(ascii_cpu, device)
    try:
        out.append(trial("dnatok_default", name, sc, "encode", dn_default, device=device, warmup=warmup, iters=iters,
                         bytes_h2d=bytes_h2d_for_method("dnatok_default", sc)))
    except Exception as e:
        logger.warning("dnatok_default failed on %s/%s: %s", name, sc.name, e)

    # uint8 IDs variant — only if ASCII path and narrow vocab
    if dna.ascii_lut is not None and dna._ascii_lut_fits_uint8():
        def dn_u8():
            ascii_cpu = dna.encode_batch_to_ascii_bytes(seqs)
            _ = dna._map_ascii_bytes_to_ids_cuda_u8(ascii_cpu, device)
        try:
            out.append(trial("dnatok_uint8", name, sc, "encode", dn_u8, device=device, warmup=warmup, iters=iters,
                             bytes_h2d=bytes_h2d_for_method("dnatok_uint8", sc)))
        except Exception as e:
            logger.warning("uint8 failed on %s/%s: %s", name, sc.name, e)

    # 2-bit packed nucleotides — ASCII path only (k-mer expansion would
    # change once the bases are packed; left as future work).
    if dna.ascii_lut is not None:
        def dn_2bit():
            packed, T_true = dna.encode_batch_to_packed_2bit(seqs)
            _ = dna._unpack_2bit_to_ids_cuda(packed, T_true, device)
        try:
            out.append(trial("dnatok_pack2bit", name, sc, "encode", dn_2bit, device=device, warmup=warmup, iters=iters,
                             bytes_h2d=bytes_h2d_for_method("dnatok_pack2bit", sc)))
        except Exception as e:
            logger.warning("pack2bit failed on %s/%s: %s", name, sc.name, e)

    # 2-bit packed + uint8 device-side IDs (combined bus + device savings).
    if dna.ascii_lut is not None and dna._ascii_lut_fits_uint8():
        def dn_2bit_u8():
            packed, T_true = dna.encode_batch_to_packed_2bit(seqs)
            _ = dna._unpack_2bit_to_ids_cuda_u8(packed, T_true, device)
        try:
            out.append(trial("dnatok_pack2bit_uint8", name, sc, "encode", dn_2bit_u8,
                             device=device, warmup=warmup, iters=iters,
                             bytes_h2d=bytes_h2d_for_method("dnatok_pack2bit_uint8", sc)))
        except Exception as e:
            logger.warning("pack2bit_uint8 failed on %s/%s: %s", name, sc.name, e)

    # CUDA graph replay — only ASCII (static shape; k-mer expansion has
    # dynamic offsets). The timing window MUST include the CPU-side ASCII
    # byte construction so the comparison against `dnatok_default` is
    # apples-to-apples (default also pays that cost).
    if dna.ascii_lut is not None:
        try:
            dna._record_cuda_graph(seqs, device, emb_batch=sc.emb_batch)

            def dn_graph():
                ascii_cpu = dna.encode_batch_to_ascii_bytes(seqs)
                _ = dna._replay_cuda_graph(ascii_cpu)

            out.append(trial("dnatok_cuda_graph", name, sc, "encode", dn_graph, device=device, warmup=warmup, iters=iters,
                             bytes_h2d=bytes_h2d_for_method("dnatok_cuda_graph", sc)))
        except Exception as e:
            logger.warning("cuda_graph failed on %s/%s: %s", name, sc.name, e)
        finally:
            dna._graph = None
            dna._graph_static_in = None
            dna._graph_static_out = None

    return out


def e2e_phase(
    name: str,
    sc: Scenario,
    adapter: TokenizerAdapter,
    dna: DNATok,
    embedder: MockEmbedder,
    device: torch.device,
    *,
    warmup: int,
    iters: int,
) -> list[Trial]:
    sc = _adjust_seqlen(sc, dna.kmer_k)
    seqs = make_sequences(sc.batch_size, sc.seq_len)
    out: list[Trial] = []

    # HF + embedder
    def hf_e2e():
        ids, _ = adapter.encode_batch(seqs, padding=True, add_special_tokens=False)
        ids = ids.to(device, non_blocking=True)
        _ = embedder.embed_tokens(ids)
    out.append(trial("hf_native", name, sc, "e2e", hf_e2e, device=device, warmup=warmup, iters=iters,
                     bytes_h2d=bytes_h2d_for_method("hf_native", sc)))

    # DNAtok default — uses the production pipeline
    def dn_e2e_default():
        chunks = dna.embed_from_strings(seqs, emb_batch=sc.emb_batch, device=device, path="auto")
        for _ in chunks:
            pass
    out.append(trial("dnatok_default", name, sc, "e2e", dn_e2e_default, device=device, warmup=warmup, iters=iters,
                     bytes_h2d=bytes_h2d_for_method("dnatok_default", sc)))

    # GPU BPE e2e for genomic BPE tokenizers — same three backends as encode_phase.
    if GPUTokBPEBackend is not None and GPUTokBPEBackend.is_supported(adapter.tokenizer):
        for eng, method_name in (("gputok",    "dnatok_gputok_bpe"),
                                  ("dnatok",    "dnatok_dna_kernel"),
                                  ("dnatok_v3", "dnatok_dna_kernel_v3")):
            try:
                backend = GPUTokBPEBackend(adapter.tokenizer, engine=eng)
                def _e2e(b=backend):
                    ids, _ = b.encode_batch(seqs, device=device)
                    _ = embedder.embed_tokens(ids)
                out.append(trial(method_name, name, sc, "e2e",
                                 _e2e, device=device,
                                 warmup=warmup, iters=iters,
                                 bytes_h2d=bytes_h2d_for_method(method_name, sc),
                                 notes=f"engine={eng}; HF-exact"))
            except Exception as e:
                logger.warning("%s e2e failed on %s/%s: %s", method_name, name, sc.name, e)

    # Triton fused tokenize+gather (only when LUT fits and Triton available).
    # The CPU-side ASCII construction is INCLUDED in the timing window so the
    # comparison against `hf_native` / `dnatok_default` (both of which pay
    # CPU tokenisation/encoding cost inside their windows) is honest.
    if dna._try_import_triton() and dna.ascii_lut is not None and dna.kmer_k is None:
        embed_weight = embedder.embed_table.weight.data

        def dn_e2e_fused():
            ascii_cpu = dna.encode_batch_to_ascii_bytes(seqs)
            _ = dna._fused_tokenize_gather(ascii_cpu, embed_weight, device)

        try:
            out.append(trial("dnatok_fused_triton", name, sc, "e2e", dn_e2e_fused, device=device, warmup=warmup, iters=iters,
                             bytes_h2d=bytes_h2d_for_method("dnatok_fused_triton", sc)))
        except Exception as e:
            logger.warning("fused kernel failed on %s/%s: %s", name, sc.name, e)

    return out


def h2d_only_phase(
    name: str,
    sc: Scenario,
    adapter: TokenizerAdapter,
    dna: DNATok,
    device: torch.device,
    *,
    warmup: int,
    iters: int,
) -> list[Trial]:
    """Measure pure host->device bandwidth for each variant. Separates the
    PCIe component from the LUT lookup so reviewers can see the actual
    bandwidth saving for 2-bit pack and uint8 IDs.
    """
    sc = _adjust_seqlen(sc, dna.kmer_k)
    seqs = make_sequences(sc.batch_size, sc.seq_len)
    out: list[Trial] = []

    if not dna.use_ids_path:
        return out

    # HF int64 ids H2D
    try:
        hf_ids, _ = adapter.encode_batch(seqs, padding=True, add_special_tokens=False)
        if hf_ids.device.type != "cpu":
            hf_ids = hf_ids.cpu()
        hf_ids = hf_ids.pin_memory() if not hf_ids.is_pinned() else hf_ids
        def hf_h2d():
            _ = hf_ids.to(device, non_blocking=True)
        out.append(trial("hf_native_h2d_only", name, sc, "h2d", hf_h2d, device=device,
                         warmup=warmup, iters=iters,
                         bytes_h2d=hf_ids.numel() * hf_ids.element_size()))
    except Exception as e:
        logger.warning("h2d HF failed on %s/%s: %s", name, sc.name, e)

    if dna.ascii_lut is not None:
        ascii_cpu = dna.encode_batch_to_ascii_bytes(seqs)
        # uint8 ASCII H2D (DNAtok default)
        def u8_h2d():
            _ = ascii_cpu.to(device, non_blocking=True)
        out.append(trial("dnatok_h2d_only", name, sc, "h2d", u8_h2d, device=device,
                         warmup=warmup, iters=iters,
                         bytes_h2d=ascii_cpu.numel() * ascii_cpu.element_size()))
        # 2-bit packed H2D
        try:
            packed, _T = dna.encode_batch_to_packed_2bit(seqs)
            def p2_h2d():
                _ = packed.to(device, non_blocking=True)
            out.append(trial("dnatok_pack2bit_h2d_only", name, sc, "h2d", p2_h2d, device=device,
                             warmup=warmup, iters=iters,
                             bytes_h2d=packed.numel() * packed.element_size()))
        except Exception as e:
            logger.warning("h2d 2bit failed on %s/%s: %s", name, sc.name, e)

    return out


def latency_cdf_phase(
    name: str,
    adapter: TokenizerAdapter,
    dna: DNATok,
    device: torch.device,
    *,
    n_iters: int = 1000,
) -> list[Trial]:
    """High-iter, small-batch CDF run (B=1, T=128) — the regime where
    kernel-launch overhead dominates and CUDA-graph replay should win
    most decisively. Saves the full per-iter timing array for plotting.
    """
    sc = Scenario("latency_cdf", batch_size=1, seq_len=128, emb_batch=1)
    sc = _adjust_seqlen(sc, dna.kmer_k)
    seqs = make_sequences(sc.batch_size, sc.seq_len)
    out: list[Trial] = []

    # HF native (CPU encode + H2D)
    def hf_one():
        ids, _ = adapter.encode_batch(seqs, padding=True, add_special_tokens=False)
        ids.to(device, non_blocking=True)
    out.append(trial("hf_native", name, sc, "latency_cdf", hf_one,
                     device=device, warmup=10, iters=n_iters,
                     bytes_h2d=bytes_h2d_for_method("hf_native", sc),
                     keep_durations=True))

    if not dna.use_ids_path:
        return out
    if dna.ascii_lut is None:
        return out

    def dn_default():
        ascii_cpu = dna.encode_batch_to_ascii_bytes(seqs)
        _ = dna._map_ascii_bytes_to_ids_cuda(ascii_cpu, device)
    out.append(trial("dnatok_default", name, sc, "latency_cdf", dn_default,
                     device=device, warmup=10, iters=n_iters,
                     bytes_h2d=bytes_h2d_for_method("dnatok_default", sc),
                     keep_durations=True))

    try:
        dna._record_cuda_graph(seqs, device, emb_batch=1)
        def dn_graph():
            ascii_cpu = dna.encode_batch_to_ascii_bytes(seqs)
            _ = dna._replay_cuda_graph(ascii_cpu)
        out.append(trial("dnatok_cuda_graph", name, sc, "latency_cdf", dn_graph,
                         device=device, warmup=10, iters=n_iters,
                         bytes_h2d=bytes_h2d_for_method("dnatok_cuda_graph", sc),
                         keep_durations=True))
    except Exception as e:
        logger.warning("cuda graph CDF failed for %s: %s", name, e)
    finally:
        dna._graph = None
        dna._graph_static_in = None
        dna._graph_static_out = None
    return out


def gpu_baseline_phase(
    name: str,
    sc: Scenario,
    device: torch.device,
    *,
    warmup: int,
    iters: int,
) -> list[Trial]:
    out: list[Trial] = []
    seqs = make_sequences(sc.batch_size, sc.seq_len)

    # cuDF baselines (skip silently if cuDF is unavailable on this host)
    if have_cudf():
        def cudf_char():
            cudf_character_tokenize(seqs)
        try:
            out.append(trial("cudf_character_tokenize", name, sc, "encode", cudf_char, device=device, warmup=warmup, iters=iters,
                             bytes_h2d=bytes_h2d_for_method("cudf_character_tokenize", sc),
                             notes="RAPIDS cudf.str.character_tokenize"))
        except Exception as e:
            logger.warning("cudf character failed: %s", e)

        def cudf_ngrams():
            cudf_ngrams_tokenize(seqs, n=6)
        try:
            out.append(trial("cudf_ngrams_tokenize_k6", name, sc, "encode", cudf_ngrams, device=device, warmup=warmup, iters=iters,
                             bytes_h2d=bytes_h2d_for_method("cudf_ngrams_tokenize_k6", sc),
                             notes="RAPIDS cudf.str.ngrams_tokenize n=6"))
        except Exception as e:
            logger.warning("cudf ngrams failed: %s", e)

    # GPUTOK (BlockBPE-style GPU BPE; uses GPT-2 vocab so output is
    # biologically meaningless but the throughput is the SOTA comparison).
    # Runs independently of cuDF availability.
    if load_gputok() is not None:
        def gputok_opt():
            gputok_tokenize(seqs, variant="optimized")
        try:
            out.append(trial("gputok_blockbpe_opt", name, sc, "encode", gputok_opt, device=device, warmup=warmup, iters=iters,
                             bytes_h2d=bytes_h2d_for_method("gputok_blockbpe_opt", sc),
                             notes="GPUTOK / BlockBPE-style optimised GPU BPE (GPT-2 vocab)"))
        except Exception as e:
            logger.warning("gputok optimised failed: %s", e)
        def gputok_base():
            gputok_tokenize(seqs, variant="baseline")
        try:
            out.append(trial("gputok_blockbpe_base", name, sc, "encode", gputok_base, device=device, warmup=warmup, iters=iters,
                             bytes_h2d=bytes_h2d_for_method("gputok_blockbpe_base", sc),
                             notes="GPUTOK / BlockBPE-style baseline GPU BPE (GPT-2 vocab)"))
        except Exception as e:
            logger.warning("gputok baseline failed: %s", e)
    return out


def run_for_model(
    spec: ModelSpec,
    scenarios: tuple[Scenario, ...],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
    skip_e2e: bool = False,
    cdf_iters: int = 0,
) -> list[Trial]:
    path = resolve_model_path(spec)
    if path is None:
        logger.warning("Model %s: no path resolved; skipping", spec.name)
        return []
    try:
        tok = load_hf_tokenizer(str(path))
    except Exception as e:
        logger.warning("Tokenizer load failed for %s: %s", spec.name, e)
        return []
    adapter = TokenizerAdapter(tok)
    embedder = MockEmbedder(tok, hidden_size=256, device=device)
    # Validation policy: we instantiate two DNAtok views of the model.
    # `dna_strict` mirrors the *production-safe* default (case normalisation +
    # invalid-char handling — equivalent to what a careless caller would
    # rely on). `dna_fast` assumes pre-validated input, matching the
    # assumption made by every GPU-tokenizer baseline we compare against
    # (tiktoken, BlockBPE, cuDF nvtext). The optimised paths (uint8,
    # 2-bit pack, CUDA graph, Triton fused) cannot include per-batch
    # validation by construction, so reporting them against the strict
    # variant would be unfair to the optimisations *and* unfair to the
    # baselines they are positioned against.
    dna_strict = DNATok(embedder, normalize_case=True, handle_invalid_chars=True)
    dna_fast = DNATok(embedder, normalize_case=False, handle_invalid_chars=False)
    try:
        dna_strict.discover()
        dna_fast.discover()
    except Exception as e:
        logger.warning("Discover failed for %s: %s", spec.name, e)
        return []
    # BPE tokenizers (GENA-LM, DNABERT-2, METAGENE-1) cannot enable DNAtok's
    # LUT fast path; the LUT variants below early-return for them. They can
    # still be measured for the e2e path, which now routes BPE through the
    # pinned-staging + pipelined H2D streamer. We keep the model in scope and
    # let encode_phase / e2e_phase decide what is applicable.
    if not dna_fast.use_ids_path:
        logger.info("IDs path not enabled for %s; running BPE-fallback measurements only", spec.name)
    dna = dna_fast  # alias kept for the rest of this function

    trials: list[Trial] = []
    for sc in scenarios:
        try:
            trials.extend(encode_phase(spec.name, sc, adapter, dna_fast, device, warmup=warmup, iters=iters))
            # Also measure the production-safe (strict) default so the paper
            # can transparently report both numbers.
            if dna_strict.use_ids_path:
                sc_adj = _adjust_seqlen(sc, dna_strict.kmer_k)
                seqs = make_sequences(sc_adj.batch_size, sc_adj.seq_len)
                def dn_strict():
                    if dna_strict.ascii_lut is not None:
                        ascii_cpu = dna_strict.encode_batch_to_ascii_bytes(seqs)
                        _ = dna_strict._map_ascii_bytes_to_ids_cuda(ascii_cpu, device)
                    else:
                        ascii_cpu = dna_strict.encode_batch_to_ascii_bytes(seqs)
                        _ = dna_strict._map_kmer_bytes_to_ids_cuda(ascii_cpu, device)
                try:
                    trials.append(trial("dnatok_default_strict", spec.name, sc_adj, "encode", dn_strict,
                                        device=device, warmup=warmup, iters=iters,
                                        bytes_h2d=bytes_h2d_for_method("dnatok_default_strict", sc_adj),
                                        notes="includes per-batch normalize_case + handle_invalid_chars"))
                except Exception as e:
                    logger.warning("strict default failed on %s/%s: %s", spec.name, sc.name, e)
            # H2D-only phase (pure bandwidth)
            try:
                trials.extend(h2d_only_phase(spec.name, sc, adapter, dna_fast, device,
                                             warmup=warmup, iters=iters))
            except Exception as e:
                logger.warning("h2d phase failed on %s/%s: %s", spec.name, sc.name, e)
            if not skip_e2e:
                trials.extend(e2e_phase(spec.name, sc, adapter, dna_fast, embedder, device, warmup=warmup, iters=iters))
        except torch.cuda.OutOfMemoryError as e:
            logger.warning("OOM on %s / %s — skipping rest of model: %s", spec.name, sc.name, e)
            torch.cuda.empty_cache()
            break
        except Exception as e:
            logger.error("Failed %s / %s: %s", spec.name, sc.name, e)
            traceback.print_exc()

    # Latency-CDF phase (single sub-millisecond regime, 1000 iters)
    if cdf_iters and cdf_iters > 0:
        try:
            trials.extend(latency_cdf_phase(spec.name, adapter, dna_fast, device,
                                            n_iters=cdf_iters))
        except Exception as e:
            logger.warning("CDF phase failed on %s: %s", spec.name, e)

    dna_strict.free_buffers()
    dna_fast.free_buffers()
    del embedder, dna_strict, dna_fast, adapter, tok
    torch.cuda.empty_cache()
    return trials


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(out_dir: Path, trials: list[Trial], meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "bench_full.json").open("w") as f:
        json.dump([asdict(t) for t in trials], f, indent=2)

    # CSV summary: omit the durations_ms list column (it's in the JSON);
    # CSV is for table-friendly viewing.
    rows = [asdict(t) for t in trials]
    if rows:
        csv_cols = [c for c in rows[0].keys() if c != "durations_ms"]
        with (out_dir / "bench_summary.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)


def collect_meta(scenarios: tuple[Scenario, ...]) -> dict[str, Any]:
    git_sha = ""
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        pass
    try:
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    except Exception:
        cap, name, mem = None, None, None
    return {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": git_sha,
        "device": name,
        "compute_capability": str(cap) if cap else None,
        "device_memory_gb": (mem / 1e9) if mem else None,
        "torch": torch.__version__,
        "scenarios": [asdict(sc) for sc in scenarios],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results_hpc" / "publication")
    parser.add_argument("--models", nargs="*", help="Subset of model names to run; default = all")
    parser.add_argument("--quick", action="store_true", help="Use minimal scenarios + fewer iters")
    parser.add_argument("--full", action="store_true", help="Use full scenario sweep")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--cdf-iters", type=int, default=0,
                        help="Number of small-batch (B=1,T=128) iters for latency CDF. 0 to skip.")
    parser.add_argument("--include-gpu-baseline", action="store_true", default=True)
    parser.add_argument("--no-gpu-baseline", dest="include_gpu_baseline", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s  capability: %s", device,
                torch.cuda.get_device_capability() if device.type == "cuda" else "n/a")

    if args.quick:
        scenarios = QUICK_SCENARIOS
        args.warmup = max(args.warmup, 2)
        args.iters = max(args.iters, 5)
    else:
        scenarios = PUBLICATION_SCENARIOS

    selected = [
        s for s in MODEL_SPECS
        if (args.models is None or s.name in args.models) and s.kind == "hf"
    ]
    logger.info("models to benchmark: %d", len(selected))

    ts = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trials: list[Trial] = []
    for i, spec in enumerate(selected, start=1):
        logger.info("[%d/%d] %s (group=%s)", i, len(selected), spec.name, spec.group)
        t0 = time.perf_counter()
        try:
            trials = run_for_model(
                spec, scenarios,
                warmup=args.warmup, iters=args.iters,
                device=device, skip_e2e=args.skip_e2e,
                cdf_iters=args.cdf_iters,
            )
            all_trials.extend(trials)
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM on %s; continuing", spec.name)
            torch.cuda.empty_cache()
        except Exception as e:
            logger.exception("model %s failed: %s", spec.name, e)
        # Save progress after every model
        meta = collect_meta(scenarios)
        meta.update({"completed_models": [t.model for t in all_trials], "elapsed_sec": time.perf_counter() - t0})
        write_outputs(out_dir, all_trials, meta)
        logger.info("[%d/%d] %s done in %.1fs", i, len(selected), spec.name, time.perf_counter() - t0)

    # Run GPU baselines if cuDF OR GPUTOK is available (either alone is useful)
    if args.include_gpu_baseline and (have_cudf() or load_gputok() is not None):
        logger.info("Running cuDF GPU baseline...")
        for sc in scenarios:
            try:
                all_trials.extend(gpu_baseline_phase("__cudf_baseline__", sc, device, warmup=args.warmup, iters=args.iters))
            except Exception as e:
                logger.warning("cudf baseline failed on %s: %s", sc.name, e)
        write_outputs(out_dir, all_trials, collect_meta(scenarios))

    logger.info("Done. Wrote %d trials to %s", len(all_trials), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
