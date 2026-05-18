"""DNA-specialised GPU BPE kernel — Python loader + wrapper.

Provides a `DnatokBpeKernel` class that builds the CUDA extension on first
use, then exposes a `tokenize_batch(seqs) -> (ids_cuda, lengths_cuda)` API.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_THIS_DIR = Path(__file__).resolve().parent
_KERNEL_SRC = str(_THIS_DIR / "dnatok_bpe.cu")

# We reuse the cuCollections + cccl headers that already ship with the
# gpu-tokenizer repo (Apache 2.0) — no need to vendor them a second time.
# Resolution order:
#   1. $GPUTOK_DIR env var (preferred; works on any host).
#   2. /tmp/gpu-tokenizer (common dev location).
def _externals_for(root: str) -> list[str]:
    base = f"{root.rstrip('/')}/externals"
    return [
        f"{base}/cuCollections/include",
        f"{base}/cccl/cub",
        f"{base}/cccl/thrust",
        f"{base}/cccl/libcudacxx/include",
    ]


_DEFAULT_EXTERNALS: list[str] = []
_GPUTOK_DIR_ENV = os.environ.get("GPUTOK_DIR")
if _GPUTOK_DIR_ENV:
    _DEFAULT_EXTERNALS.extend(_externals_for(_GPUTOK_DIR_ENV))
_DEFAULT_EXTERNALS.extend(_externals_for("/tmp/gpu-tokenizer"))


_extension: Optional[Any] = None
# Cache instances by (merges_path, max_iters). cuCollections instances share
# a process-wide allocator; holding many alive at once has triggered
# device-side asserts in pytest runs. Reusing one instance per unique
# (merges, max_iters) tuple side-steps the issue and is strictly faster
# (we skip the merge-map build).
_INSTANCES: Dict[Tuple[str, int], Any] = {}


def _resolve_externals() -> list[str]:
    """Return the subset of include paths that actually exist on this host."""
    return [p for p in _DEFAULT_EXTERNALS if os.path.isdir(p)]


def _load_extension():
    """JIT-compile (or reuse cached) the dnatok_bpe CUDA extension."""
    global _extension
    if _extension is not None:
        return _extension

    externals = _resolve_externals()
    if not externals:
        raise RuntimeError(
            "cuCollections / cccl headers not found. Set $GPUTOK_DIR to the "
            "root of a checked-out https://github.com/gpu-tokenizer repo, or "
            "clone it to /tmp/gpu-tokenizer."
        )

    from torch.utils.cpp_extension import load as load_extension

    _extension = load_extension(
        name="dnatok_bpe",
        sources=[_KERNEL_SRC],
        extra_include_paths=externals,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-extended-lambda"],
        verbose=False,
    )
    return _extension


class DnatokBpeKernel:
    """Build-on-first-use wrapper around the DNA-specialised CUDA BPE kernel.

    Use a separate instance per HF tokenizer (each has its own merges file
    and therefore its own internal vocab). The constructor builds the
    device merge map once; subsequent ``tokenize_batch`` calls are pure
    GPU work.

    The returned id tensor uses this kernel's INTERNAL vocab (byte symbols
    0..255 then merge results in rank order). Callers translate to HF ids
    via a remap LUT — see ``GPUTokBPEBackend`` for the same pattern.
    """

    def __init__(self, merges_path: str, max_iters: int = 1024) -> None:
        ext = _load_extension()
        key = (str(merges_path), int(max_iters))
        impl = _INSTANCES.get(key)
        if impl is None:
            byte_lut = list(range(256))
            impl = ext.DNATokBPE(merges_path, byte_lut, max_iters)
            _INSTANCES[key] = impl
        self._impl = impl

    def vocab_size(self) -> int:
        return int(self._impl.vocab_size())

    def id_to_token(self, idx: int) -> str:
        return self._impl.id_to_token(int(idx))

    def tokenize_batch(self, texts: List[str],
                        *, clone: bool = False,
                        version: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of strings.

        Args:
            texts: list of input strings (raw byte sequences).
            clone: if True, return independent tensors (not views into the
                workspace cache).
            version: which kernel to dispatch to.
                * 1 = Phase 2 rank-batched (default — bit-identical to HF,
                  fast on inputs ≤ ~2 kbp).
                * 3 = Phase 3 entry-pool bucket scheduling (bit-identical
                  to HF, intended for long inputs).
                Version 2 was the buggy first Phase-3 attempt; it is
                wired up but should never be selected.

        Returns:
            ids: int32 CUDA tensor [B, T_max], padded to 0 in positions
                 ≥ lengths[b].
            lengths: int32 CUDA tensor [B]. lengths[b] = -1 if the input
                 exceeded T_max; lengths[b] = -2 if v3's entry pool
                 overflowed (kernel-internal safety — caller should
                 increase DNATOK_V3_ENTRY_FACTOR or fall back to v1).

        IMPORTANT — caching contract:
            Unless ``clone=True``, the returned tensors are NARROW VIEWS
            into the kernel's persistent workspace cache. The next call to
            ``tokenize_batch`` on this instance will OVERWRITE the
            underlying storage.
        """
        if version == 1:
            ids, lens = self._impl.tokenize_batch(list(texts))
        elif version == 3:
            ids, lens = self._impl.tokenize_batch_v3(list(texts))
        elif version == 2:
            ids, lens = self._impl.tokenize_batch_v2(list(texts))
        else:
            raise ValueError(f"DnatokBpeKernel: unsupported version={version}")
        if clone:
            ids = ids.clone()
            lens = lens.clone()
        return ids, lens
