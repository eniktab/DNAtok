"""GPU BPE backend with HF-compatible output for genomic foundation models.

Wraps two underlying CUDA kernels behind a single HF-compatible interface
for the BPE tokenizers used in DNABERT-2, GENA-LM, METAGENE-1:

    backend = GPUTokBPEBackend(hf_tokenizer, engine="gputok"|"dnatok")
    ids, mask = backend.encode_batch(seqs)
    # ids: int64 cuda [B, T_max] (HF-vocab ids, pad=hf.pad_token_id)
    # mask: uint8 cuda [B, T_max] (1 = valid, 0 = pad)

Engines
-------
* **engine="gputok"** — stock third-party BlockBPE-baseline kernel
  (gpu-tokenizer/gputok_binding.cu, Apache 2.0). HF-exact for inputs
  whose normalised length fits in one chunk (chunk_tokens=2048);
  longer sequences fall back to HF on the CPU side. Algorithm-1 with
  the working buffer in shared memory.

* **engine="dnatok"** — our entry-pool bucket-scheduling kernel
  (src/dnatok_bpe_kernel/). HF Algorithm-1 semantics, no per-input
  length cap, O(T log T) merge schedule. Wins on every measured BPE
  tokeniser at standard / short / large-batch workloads (1.4-7.9×
  vs HF native on GB10). Has a shape-aware fallback that routes long
  inputs to HF when the batch is small enough that the one-block-per-
  sequence design under-utilises the GPU; see ``encode_batch`` for the
  heuristic.

Both engines produce id streams that are bit-identical to HF native —
asserted across 256 random sequences, N-content, mixed case, 4 kbp
and 8 kbp inputs in tests/test_gputok_bpe_backend.py.

Three things that make HF compatibility work
--------------------------------------------
1. **Merges extraction.** GPT-2-style ``merges.txt``: one ranked pair
   per line with an optional ``#version`` header. HF stores merges in
   ``tokenizer.json`` either as space-separated strings or ``[a, b]``
   lists. We materialise the file once per model under
   ``$DNATOK_GPUTOK_CACHE`` (defaults to ``~/.cache/dnatok-gputok``)
   keyed by SHA-256 of the merge list.

2. **Vocab id remap.** Both kernels seed their internal vocabs with 256
   byte symbols at ids 0-255, then assign sequential ids to merge
   results. HF uses the model's own vocab order. After the kernel runs
   we translate via a precomputed (kernel_id → hf_id) LUT.

3. **Normaliser application.** ``_compile_normalizer`` translates the
   HF tokenizer's normaliser pipeline (Strip / Replace-Regex /
   Replace-String / Lowercase / NFC variants / Sequence-of-the-above)
   into a Python callable applied to each input before the kernel sees
   it. CPU-fallback sequences are handed to HF unchanged so HF's
   pipeline applies once, not twice (the original code double-applied
   METAGENE-1's ``_`` prefix; the fix is a strict gate test now).

GPUTOK has an "optimized" kernel too but it uses merge-all semantics
that diverge from HF on ~80% of inputs. The backend never calls it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re as _re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# GPUTOK source / extension loader
# ---------------------------------------------------------------------------

_GPUTOK_EXT = None
_GPUTOK_INSTANCES: Dict[str, Any] = {}  # merges_path -> GpuTokenizer C++ instance


def _find_gpu_tokenizer_dir() -> Optional[str]:
    """Locate the gpu-tokenizer source tree (Apache 2.0 external repo).

    Returns the directory containing ``gputok_binding.cu`` or None.
    """
    candidates = [
        os.environ.get("GPUTOK_DIR"),
        "/tmp/gpu-tokenizer",
        os.path.expanduser("~/gpu-tokenizer"),
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "gputok_binding.cu")):
            return c
    return None


def _load_gputok_extension():
    """Build (once) and cache the GPUTOK PyTorch extension."""
    global _GPUTOK_EXT
    if _GPUTOK_EXT is not None:
        return _GPUTOK_EXT
    gtok_dir = _find_gpu_tokenizer_dir()
    if gtok_dir is None:
        raise RuntimeError(
            "GPUTOK source not found. Set GPUTOK_DIR or place the "
            "gpu-tokenizer repo at one of the standard locations."
        )
    from torch.utils.cpp_extension import load as load_extension

    prev_cwd = os.getcwd()
    os.chdir(gtok_dir)
    try:
        _GPUTOK_EXT = load_extension(
            name="gputok_gpu",
            sources=["gputok_binding.cu"],
            extra_include_paths=[
                "externals/cuCollections/include",
                "externals/cccl/cub",
                "externals/cccl/thrust",
                "externals/cccl/libcudacxx/include",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-extended-lambda"],
            verbose=False,
        )
    finally:
        os.chdir(prev_cwd)
    return _GPUTOK_EXT


# ---------------------------------------------------------------------------
# Helpers: tokenizer.json extraction
# ---------------------------------------------------------------------------

def _read_tokenizer_json(hf_tokenizer: Any) -> Optional[Dict[str, Any]]:
    """Locate and read the tokenizer.json that backs this HF tokenizer."""
    # Standard path: name_or_path attribute points at the model dir
    paths_to_try: List[str] = []
    for attr in ("name_or_path", "vocab_file"):
        v = getattr(hf_tokenizer, attr, None)
        if isinstance(v, str) and v:
            paths_to_try.append(v)
    for base in paths_to_try:
        for cand in (os.path.join(base, "tokenizer.json"), base):
            if os.path.isfile(cand) and cand.endswith("tokenizer.json"):
                with open(cand) as f:
                    return json.load(f)
    return None


def _compile_normalizer_step(step: Dict[str, Any]):
    """Translate one HF normalizer step into a Python callable str→str.

    Returns the callable, or None if the step is unsupported.
    """
    if not isinstance(step, dict):
        return None
    t = step.get("type")
    if t == "Strip":
        sl = bool(step.get("strip_left", True))
        sr = bool(step.get("strip_right", True))
        def _strip(s: str) -> str:
            if sl and sr:
                return s.strip()
            if sl:
                return s.lstrip()
            if sr:
                return s.rstrip()
            return s
        return _strip
    if t == "Replace":
        pattern = step.get("pattern")
        content = str(step.get("content", ""))
        if isinstance(pattern, dict):
            if "String" in pattern:
                literal = pattern["String"]
                return lambda s: s.replace(literal, content)
            if "Regex" in pattern:
                rx = _re.compile(pattern["Regex"])
                return lambda s: rx.sub(content, s)
        return None
    if t == "Prepend":
        # HF's Prepend normaliser inserts a fixed string at the start of
        # every input (semantically equivalent to METAGENE-1's Replace(^,X)
        # but more direct). Common on byte-level BPE tokenizers.
        prefix = str(step.get("prepend", ""))
        return lambda s: prefix + s
    if t == "StripAccents":
        # No-op for ASCII inputs (DNA alphabet). Accept as identity.
        return lambda s: s
    if t in (None, "NFC", "NFKC", "NFD", "NFKD"):
        # Unicode normalisation: identity for ASCII DNA inputs. Accept.
        return lambda s: s
    if t == "Lowercase":
        return lambda s: s.lower()
    if t == "Sequence":
        sub_fns = []
        for sub in step.get("normalizers", []):
            f = _compile_normalizer_step(sub)
            if f is None:
                return None
            sub_fns.append(f)
        def _seq(s: str) -> str:
            for f in sub_fns:
                s = f(s)
            return s
        return _seq
    return None


def _compile_normalizer(cfg: Dict[str, Any]):
    """Return a callable applying the tokenizer's normalizer, or None if
    no normalizer / identity. Raises ValueError on unsupported config."""
    norm = cfg.get("normalizer")
    if norm is None:
        return None
    fn = _compile_normalizer_step(norm)
    if fn is None:
        raise ValueError(f"Unsupported normalizer config: {norm.get('type')!r}")
    return fn


def _extract_merges(cfg: Dict[str, Any]) -> List[str]:
    """Return merges as a list of ``"a b"`` strings, in rank order."""
    merges = cfg.get("model", {}).get("merges", [])
    out: List[str] = []
    for m in merges:
        if isinstance(m, str):
            out.append(m)
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            out.append(f"{m[0]} {m[1]}")
        else:
            raise ValueError(f"Unrecognised merge entry: {m!r}")
    return out


def _supported_tokenizer(cfg: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Decide if a tokenizer.json describes a model we can accelerate.

    Returns (ok, reason). Currently supports:
      * BPE model
      * Whitespace / Split / WhitespaceSplit / null pre_tokenizer (no-op
        for the whitespace-free DNA inputs we target — empirically
        verified against HF native on all three BPE genomic tokenizers).
      * Any normalizer whose steps reduce to Strip / Replace / Sequence
        of those (Lowercase / Unicode-NFC are also accepted as identity
        for ASCII inputs).
    """
    if cfg is None:
        return False, "tokenizer.json not found"
    model = cfg.get("model") or {}
    if model.get("type") != "BPE":
        return False, f"model.type={model.get('type')!r} (only BPE supported)"
    pre = cfg.get("pre_tokenizer")
    if pre is not None:
        pt = pre.get("type") if isinstance(pre, dict) else None
        if pt not in (None, "Whitespace", "Split", "WhitespaceSplit"):
            return False, f"pre_tokenizer={pt!r}"
    norm = cfg.get("normalizer")
    if norm is not None:
        try:
            fn = _compile_normalizer_step(norm)
        except Exception as e:
            return False, f"normalizer compile failed: {e}"
        if fn is None:
            return False, f"unsupported normalizer={norm.get('type')!r}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    p = Path(os.environ.get("DNATOK_GPUTOK_CACHE",
                            str(Path.home() / ".cache" / "dnatok-gputok")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _merges_cache_path(merges: List[str]) -> Path:
    """Stable path keyed by merges hash (so multiple tokenizers don't collide
    and we avoid rebuilding the GpuTokenizer when nothing changed)."""
    h = hashlib.sha256()
    for m in merges:
        h.update(m.encode("utf-8"))
        h.update(b"\n")
    return _cache_dir() / f"merges_{h.hexdigest()[:16]}.txt"


# GPUTOK chunks input into pieces of chunk_tokens; each chunk is BPE-merged
# independently. Because BPE is rank-driven, chunk boundaries break merges
# that span them and produce wrong output. The default chunk_tokens=2048
# is bounded by GPUTOK's static shared-memory budget (~48 KB at 13 bytes
# per token). For HF-exact output we must NOT chunk: every input must fit
# in a single chunk. We pick 2048 as the safe ceiling and fall back to
# HF for any input that exceeds it (see encode_batch).
_GPUTOK_SAFE_CHUNK = 2048

# Our DNA-specialised kernel does NOT chunk (no shared-memory limit), and
# its rank-batched merge loop is faster than the naive single-merge
# scheme (3-5× fewer outer iterations on random DNA). It still has a
# per-iter O(T) rank scan, so the late iterations — which only batch one
# or two merges — push the worst-case toward O(T²). HF Rust uses a heap
# and is O(T log T). Empirically the kernel wins below ~2 kbp; above it,
# HF wins. We route long sequences to HF until a heap-on-GPU pass closes
# the gap. The threshold is set to 2048 so it coincides with the gputok
# engine's chunk_tokens limit, keeping behavioural comparisons direct.
_DNATOK_FAST_LIMIT = 2048


def _build_gputok_instance(merges_path: str, chunk_tokens: int = _GPUTOK_SAFE_CHUNK,
                            max_iters: int = 200):
    """Get-or-build cached GpuTokenizer for these merges.

    max_iters bumped from GPUTOK's default 50 → 200 because deep merge
    chains are common on long DNA where every initial pair is mergeable.
    The kernel terminates as soon as no further merge applies, so larger
    max_iters has no cost on shorter inputs.
    """
    cache_key = f"{merges_path}|{chunk_tokens}|{max_iters}"
    if cache_key in _GPUTOK_INSTANCES:
        return _GPUTOK_INSTANCES[cache_key]
    ext = _load_gputok_extension()
    inst = ext.GpuTokenizer(merges_path, chunk_tokens, max_iters)
    _GPUTOK_INSTANCES[cache_key] = inst
    return inst


def _reconstruct_gputok_vocab(merges: List[str]) -> List[str]:
    """Reproduce GPUTOK's exact id→string assignment without touching the C++.

    GPUTOK's ``build_vocab_and_merges`` (gputok_binding.cu) does:
      1. Seed ids 0..255 with the GPT-2 byte-encoder symbols. For printable
         ASCII (which covers the DNA alphabet ACGTNacgtn) the symbol is the
         character itself, so chr(b) suffices.
      2. For each merge line ``a b`` in rank order:
            * if ``a`` not in vocab → add it (next id)
            * if ``b`` not in vocab → add it (next id)
            * add the merge result ``a + b`` (next id)

    We must replicate steps 2a/2b precisely; if we skip them, the produced
    vocab indices drift from GPUTOK's and the remap LUT is silently wrong.
    """
    vocab: List[str] = [chr(b) for b in range(256)]
    seen: Dict[str, int] = {tok: i for i, tok in enumerate(vocab)}

    def _add(tok: str) -> int:
        idx = seen.get(tok)
        if idx is not None:
            return idx
        idx = len(vocab)
        vocab.append(tok)
        seen[tok] = idx
        return idx

    for line in merges:
        parts = line.split()
        if len(parts) != 2:
            continue
        a, b = parts
        _add(a)
        _add(b)
        _add(a + b)
    return vocab


class GPUTokBPEBackend:
    """HF-compatible GPU BPE encoder for genomic foundation models.

    See the module docstring for the architecture overview. In short:
      * ``engine="gputok"`` — third-party BlockBPE-baseline kernel.
      * ``engine="dnatok"`` — our entry-pool bucket-scheduling kernel
                              (recommended default).

    The ``gputok`` engine routes inputs longer than 2 kbp to HF.
    ``dnatok`` has no algorithmic length cap and falls back to HF only
    when the batch is small enough that the one-block-per-sequence GPU
    design under-utilises the device — the routing heuristic is
    documented inline in ``encode_batch``. Both paths are HF-exact
    (validated by the strict bit-identical gate).
    """

    def __init__(self, hf_tokenizer: Any, *,
                 engine: str = "dnatok") -> None:
        if engine not in ("gputok", "dnatok"):
            raise ValueError(f"engine must be 'gputok' or 'dnatok', got {engine!r}")
        self.tokenizer = hf_tokenizer
        self.engine = engine

        cfg = _read_tokenizer_json(hf_tokenizer)
        ok, reason = _supported_tokenizer(cfg)
        if not ok:
            raise RuntimeError(f"GPUTokBPEBackend cannot accelerate this tokenizer: {reason}")
        assert cfg is not None  # guaranteed by _supported_tokenizer

        # Compiled normalizer (None if no normalizer / identity).
        self._normalize = _compile_normalizer(cfg)
        self._merges: List[str] = _extract_merges(cfg)
        if not self._merges:
            raise RuntimeError("Tokenizer has no merges; nothing to accelerate.")

        # Materialise merges.txt (cached by hash) and build the GpuTokenizer.
        path = _merges_cache_path(self._merges)
        if not path.exists():
            with open(path, "w") as f:
                f.write("#version: 0.2\n")
                for m in self._merges:
                    f.write(m + "\n")
        self._merges_path = str(path)
        if self.engine == "gputok":
            self._gputok = _build_gputok_instance(self._merges_path)
            self._dnatok = None
        else:  # "dnatok"
            from dnatok_bpe_kernel import DnatokBpeKernel  # local import: build on first use
            self._gputok = None
            self._dnatok = DnatokBpeKernel(self._merges_path, max_iters=1024)

        # Build the (gputok_id -> hf_id) remap LUT. Lookups via HF's
        # convert_tokens_to_ids (or vocab dict) work for every token GPUTOK
        # can produce — IF the merges came from the HF tokenizer in the
        # first place (verified for DNABERT-2, GENA-LM, METAGENE-1).
        hf_vocab = hf_tokenizer.get_vocab() if hasattr(hf_tokenizer, "get_vocab") else {}
        gputok_vocab = _reconstruct_gputok_vocab(self._merges)
        unk_id = int(getattr(hf_tokenizer, "unk_token_id", 0) or 0)
        remap = torch.full((len(gputok_vocab),), unk_id, dtype=torch.long)
        unmapped: List[int] = []
        for gid, tok_str in enumerate(gputok_vocab):
            hf_id = hf_vocab.get(tok_str)
            if isinstance(hf_id, int):
                remap[gid] = hf_id
            else:
                # Byte-symbol slot that the HF vocab does not include — this
                # is only ever reached for non-DNA input bytes, which is the
                # caller's contract (we tokenize DNA strings). Mapping to
                # UNK keeps downstream embed safe if it ever fires.
                unmapped.append(gid)
        self._remap_cpu = remap                       # int64 [V_gputok]
        self._remap_cuda: Optional[torch.Tensor] = None  # lazy-promoted
        self.vocab_size = int(getattr(hf_tokenizer, "vocab_size", 0) or len(hf_vocab))
        self.pad_id = int(getattr(hf_tokenizer, "pad_token_id", 0) or 0)
        self.unk_id = unk_id
        # Track for diagnostics; never raise — DNA encoding doesn't hit
        # these byte slots in practice and we want the backend usable on
        # arbitrary BPE genomic tokenizers without per-model patches.
        self._n_unmapped = len(unmapped)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def is_supported(hf_tokenizer: Any) -> bool:
        """Cheap probe: would the backend accept this tokenizer?"""
        try:
            cfg = _read_tokenizer_json(hf_tokenizer)
            ok, _ = _supported_tokenizer(cfg)
            return ok
        except Exception:
            return False

    def encode_batch(self, seqs: List[str], *, device: object = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of DNA strings.

        Returns:
            (input_ids, attention_mask) on ``device``:
              * input_ids:      int64 tensor [B, T_max]
              * attention_mask: uint8 tensor [B, T_max] (1 = valid, 0 = pad)

        Padding positions in ``input_ids`` are ``hf_tokenizer.pad_token_id``.
        The non-pad ids are bit-identical to what
        ``hf_tokenizer(seqs, padding=True, add_special_tokens=False)['input_ids']``
        would produce — see tests/test_gputok_bpe_backend.py.
        """
        if not seqs:
            raise ValueError("No sequences provided.")
        # Keep references to the originals so the CPU fallback can hand
        # them to HF unchanged (HF will re-apply its own normalizer; if
        # we normalize first, the HF call double-applies and produces
        # wrong tokens — found by the strict gate on METAGENE-1).
        original_seqs = seqs
        # Normalized strings are what GPUTOK actually consumes.
        if self._normalize is not None:
            norm_seqs = [self._normalize(s) for s in seqs]
        else:
            norm_seqs = list(seqs)

        # ---------------- chunk-boundary correctness gate ----------------
        # GPUTOK chunks each input into pieces of chunk_tokens. Each chunk
        # is BPE-merged independently, which is wrong: rank-based BPE can
        # have merges that span the boundary. Any input long enough to
        # chunk will silently produce non-HF-exact output, so we route
        # those individual sequences through HF native and merge the
        # results back into the batched tensor. Only sequences that fit
        # in one chunk go through the GPU kernel.
        #
        # IMPORTANT: gate on the NORMALIZED length, because the kernel
        # operates on that stream. A METAGENE input of 2048 bytes becomes
        # 2049 once the leading "_" is prepended, which crosses the
        # default chunk_tokens=2048 boundary.
        #
        # The DNA-specialised kernel does NOT chunk — global-memory
        # working buffer has no chunk limit — so every sequence goes to
        # the GPU there regardless of length.
        # Length-based routing. Both engines use the same threshold for
        # different reasons (see the constants above); the gating logic
        # is therefore the same.
        # v3 (entry-pool) has no length pathology algorithmically — it
        # processes long sequences in O(T log T) on the GPU. But the
        # one-block-per-sequence design under-utilises the GPU when B is
        # small: at B=2 on a 144-SM device, only 2 SMs are doing work.
        # In that regime HF CPU parallelism wins on tokenisers with a
        # non-trivial merge table.
        #
        # Routing heuristic uses (B, num_merges) — both are known at this
        # point:
        #   * Small merge table (< 2k merges, e.g. METAGENE-1): per-iter
        #     cost on the kernel is low; it stays competitive even at
        #     small B and moderate T. Only fall back when both B and T
        #     are extreme.
        #   * Large merge table (>= 2k merges, e.g. DNABERT-2 / GENA-LM):
        #     per-iter scan cost dominates at small B. Fall back to HF
        #     at 2 kbp.
        if self.engine == "dnatok":
            n_merges = len(self._merges)
            B_eff = len(norm_seqs)
            if B_eff < 16 and n_merges >= 2000:
                threshold = _DNATOK_FAST_LIMIT  # 2 kbp
            elif B_eff < 4 and n_merges < 2000:
                threshold = 16384               # ultra-long only
            else:
                threshold = float("inf")
        else:  # gputok
            threshold = _GPUTOK_SAFE_CHUNK
        fits = [len(s) <= threshold for s in norm_seqs]
        gpu_idx = [i for i, ok in enumerate(fits) if ok]
        cpu_idx = [i for i, ok in enumerate(fits) if not ok]

        B = len(seqs)
        dev = torch.device(device) if not isinstance(device, torch.device) else device

        # ------------------------------------------------------------------
        # Stay-on-GPU fast path: dnatok engine + every sequence fits.
        # ------------------------------------------------------------------
        # The kernel returns int32 [B, T_kernel] ids + int32 [B] lengths,
        # both already on the GPU. We can build the (input_ids, attention_mask)
        # output without ever touching the CPU side. This skips the
        # tolist() / pad / re-upload round-trip that dominates the slow path
        # on large batches (~3-10 ms on B=256 standard scenarios).
        if self.engine == "dnatok" and not cpu_idx and gpu_idx:
            ids_dev, lens_dev = self._dnatok.tokenize_batch(norm_seqs)
            if self._remap_cuda is None or self._remap_cuda.device != dev:
                self._remap_cuda = self._remap_cpu.to(dev)
            ids_hf_dev = self._remap_cuda[ids_dev.long()]  # int64 [B, T_kernel]
            # Trim to actual max length (one sync). The kernel zero-fills
            # the tail of ids_dev so remap-indexing never read garbage.
            max_len_t = lens_dev.max()
            max_len = int(max_len_t.item())
            if max_len == 0:
                return (
                    torch.empty((B, 0), dtype=torch.long, device=dev),
                    torch.empty((B, 0), dtype=torch.uint8, device=dev),
                )
            ids_hf_trim = ids_hf_dev[:, :max_len]
            # mask[b, j] = (j < lens_dev[b]) — built entirely on device.
            pos = torch.arange(max_len, dtype=torch.int32, device=dev).unsqueeze(0)
            mask_bool = pos < lens_dev.unsqueeze(1)
            mask = mask_bool.to(torch.uint8)
            out_ids = torch.where(
                mask_bool,
                ids_hf_trim,
                torch.full_like(ids_hf_trim, self.pad_id),
            )
            return out_ids, mask

        # --- GPU path for sequences that fit in the engine's GPU budget ---
        # gputok engine: BlockBPE-baseline kernel (Algorithm 1, leftmost
        # merge of the globally lowest-rank pair, one merge per iter).
        # dnatok engine: our entry-pool bucket-scheduling kernel.
        # Both produce internal vocab ids; we apply the (kernel_id → hf_id)
        # remap LUT before writing into per_seq_hf_ids.
        per_seq_hf_ids: List[List[int]] = [[] for _ in range(B)]

        if gpu_idx:
            sub = [norm_seqs[i] for i in gpu_idx]
            if self.engine == "gputok":
                gputok_ids_list, _kernel_ms = self._gputok.tokenize_batch_blockbpe_base(sub)
                remap_cpu_np = self._remap_cpu.numpy()
                for local, src_i in enumerate(gpu_idx):
                    row = gputok_ids_list[local]
                    if row:
                        per_seq_hf_ids[src_i] = remap_cpu_np[row].tolist()
                    else:
                        per_seq_hf_ids[src_i] = []
            else:  # engine == "dnatok"
                # The DNA kernel returns tensors directly on the GPU. Remap
                # on the device too — no CPU-side fanout, no Python lists.
                # Kernel zero-fills the output tensor's tail (positions
                # beyond each sequence's length) so remap-LUT indexing
                # never reads stale workspace data.
                ids_dev, lens_dev = self._dnatok.tokenize_batch(sub)
                if self._remap_cuda is None or self._remap_cuda.device != dev:
                    self._remap_cuda = self._remap_cpu.to(dev)
                ids_hf_dev = self._remap_cuda[ids_dev.long()]
                lens_cpu = lens_dev.cpu().tolist()
                ids_hf_cpu = ids_hf_dev.cpu()
                for local, src_i in enumerate(gpu_idx):
                    L = int(lens_cpu[local])
                    if L > 0:
                        per_seq_hf_ids[src_i] = ids_hf_cpu[local, :L].tolist()
                    else:
                        per_seq_hf_ids[src_i] = []

        # --- CPU fallback for sequences that would have chunked ---
        # We route these through the HF tokenizer to preserve correctness.
        # Hand HF the *original* (un-normalized) string — HF re-applies
        # the full normalizer + pre-tokenizer pipeline itself; passing
        # our already-normalized version would double-apply (e.g.
        # METAGENE would get "__seq" instead of "_seq").
        #
        # Batch the fallback into ONE HF call rather than one call per
        # sequence — the fast tokenizers parallelise across input
        # strings internally, and we save the per-call Python overhead.
        if cpu_idx:
            fallback_seqs = [original_seqs[i] for i in cpu_idx]
            ids_batch = self.tokenizer(
                fallback_seqs,
                add_special_tokens=False,
            )["input_ids"]
            for src_i, ids in zip(cpu_idx, ids_batch):
                per_seq_hf_ids[src_i] = [int(x) for x in ids]

        # Pad the per-sequence id lists into a [B, T_max] tensor.
        max_len = max((len(r) for r in per_seq_hf_ids), default=0)
        out_ids_cpu = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        mask_cpu = torch.zeros((B, max_len), dtype=torch.uint8)
        for i, row in enumerate(per_seq_hf_ids):
            L = len(row)
            if L:
                out_ids_cpu[i, :L] = torch.as_tensor(row, dtype=torch.long)
                mask_cpu[i, :L] = 1

        # Move to device and return. Remap was already applied per-element
        # in the GPU branch; the CPU branch produced HF ids directly.
        out_ids = out_ids_cpu.to(device=dev, non_blocking=True)
        mask_dev = mask_cpu.to(device=dev, non_blocking=True)
        return out_ids, mask_dev

    def encode_batch_ids_only(self, seqs: List[str], *, device: object = "cuda") -> torch.Tensor:
        """Convenience: return just the int64 ids tensor."""
        ids, _ = self.encode_batch(seqs, device=device)
        return ids
