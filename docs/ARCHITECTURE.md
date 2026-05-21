# DNAtok architecture

How DNAtok routes a Hugging Face tokenizer through GPU fast paths
while keeping the public API identical to the upstream library.

## The drop-in surface

```python
from dna_tokenizer import DNATok
dn = DNATok(model_or_embedder)
dn.discover()                       # auto-probe + select a fast path
ids = dn.encode_batch_to_ids(seqs)  # CPU pinned int64 [B, T_max]
```

The output `ids` is bit-identical to:

```python
torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(hf_tok(s, add_special_tokens=False)["input_ids"]) for s in seqs],
    batch_first=True, padding_value=hf_tok.pad_token_id or 0,
)
```

(after accounting for left/right padding orientation).

## Discover-then-execute

`DNATok.discover()` probes the upstream tokenizer with a fixed pool of
DNA test sequences and selects the fastest correct path:

```
                                   ┌─────────────┐
                                   │ HF tokenizer│
                                   └──────┬──────┘
                                          │
                ┌─────────────────────────┴─────────────────────────┐
                │              DNATok.discover()                    │
                └─────────────────────────┬─────────────────────────┘
                                          │
       ┌──────────────┬──────────────┬────┴─────────────┬─────────────┐
       ▼              ▼              ▼                  ▼             ▼
  ┌─────────┐    ┌─────────┐    ┌──────────┐      ┌──────────┐   ┌─────────┐
  │ ASCII   │    │ k-mer   │    │ GPU BPE  │      │ Cached   │   │ HF      │
  │ LUT     │    │ LUT     │    │ kernel   │      │ LMM      │   │ fallback│
  └────┬────┘    └────┬────┘    └────┬─────┘      └────┬─────┘   └────┬────┘
       │              │              │                  │              │
       │              │              │                  │              │
     char/byte    k ∈ {2..12}   genomic BPE       non-SP BPE     anything
     tokenizer    tokenizer     (DNABERT-2,       streaming-cache  unsupported
     (NTv3,       (NTv2)        GENA-LM,          fallback         by the
     HyenaDNA,                  METAGENE-1)                        kernel
     Evo2-byte,
     Caduceus)
```

A correctness gate then encodes `ACGTNacgtn` through both the selected
fast path and the upstream tokenizer; the path is enabled only if
output is bit-identical.

## BPE: dispatch order matters

For BPE tokenisers `DNATok.encode_batch_to_ids` dispatches in this order
(verified by `gputok_bpe_backend` being present and built):

1. **GPU BPE kernel** (`gputok_bpe_backend`) — primary. Bit-identical
   to HF, ~2-4× faster on Hopper-class memory bandwidth, falls back
   on exception.
2. **Cached safe-margin lookahead encoder** (`CachedLMMBPE`) — secondary.
   Useful for streaming workloads with high cache hit rates; the
   complementary panel (Fig. 3 in the paper).
3. **Upstream HF tokeniser** — last resort.

A common silent failure mode (which we hit while writing the paper):
if the GPU kernel doesn't build (e.g. `GPUTOK_DIR` not set, or the
tokenizer.json lookup misses the HF cache), the dispatch silently
falls to (2) or (3), and benchmark timings get attributed to the GPU
kernel when in fact the slow path ran. To prevent this, the Dockerfile
sanity check now runs `dnatok info` at build time, which prints which
path `discover()` selected — if `has_gpu_bpe_backend` is False for a
BPE model, the build aborts.

## Fast paths in detail

### ASCII LUT (single-character tokenisers)

256-entry uint16 table mapping ASCII byte → token id. Built once at
`discover()` time, copied to the GPU as a constant memory bank.
Encoding is a single coalesced gather kernel.

- Tokenisers: NTv3, HyenaDNA, Caduceus, Evo2-byte
- Complexity: O(B × T) global memory reads, no branches
- Throughput on H200: 300–400 Mbp/s (Pipeline 05 pretokenize)

### k-mer LUT (k-mer tokenisers)

Precomputed table of size 4^k mapping each k-character window to its
token id. Variable batch lengths are handled by per-length-group
dispatch with a partial-tail extension that pads the final window with
the k-mer's pad token id.

- Tokenisers: NTv2 (k=6)
- Vocabulary check: every k-mer of length k must map to exactly one id
- Throughput on H200: 50–135 Mbp/s

### GPU BPE kernel (genomic BPE tokenisers)

Replicates Hugging Face's BPE algorithm bit-identically on the GPU.
Each pre-token is represented as a doubly-linked list of Symbol
records; adjacent pairs are stored in a bucket-indexed entry pool
ordered by merge rank. The CUDA kernel uses a bit-array bucket
summary plus `__ffs` lookup to find the next non-empty rank in
constant time, CUB BlockMergeSort to order candidate merges within a
block, and atomic linked-list edits to apply merges.

- Tokenisers: DNABERT-2, GENA-LM, METAGENE-1
- Built on first use via `torch.utils.cpp_extension.load`
- Headers from gpu-tokenizer (`$GPUTOK_DIR`)
- Throughput on H200: 8–16 Mbp/s; on A100: 6–10 Mbp/s (bandwidth-bound)

### Cached safe-margin lookahead encoder

For non-SentencePiece BPE tokenisers, DNAtok additionally builds a
streaming-cache layer (CachedLMMBPE). It processes input in
K-character sliding windows where K = 2·max_vocab_token_len + 2 (74
for DNABERT-2, 130 for GENA-LM). For each window, the upstream HF
tokeniser is called once, then only the leading tokens whose cumulative
character end-position is at most K − safety (with safety equal to the
longest vocabulary token) are emitted. Cache writes are keyed by the
verbatim K-character string.

The cache is **bit-identical** to HF on diverse held-out data
(verified across 2,878 chr21 sequences for DNABERT-2 + GENA-LM) but
its speedup is workload-dependent: speedup grows as cache hit rate
grows, so it shines for streaming pre-tokenization pipelines that
re-encode the same regions repeatedly (lg-asm assembly, large-cohort
variant prioritization).

## V1 stack: staging + H2D overlap

DNAtok's `encode_batch_to_ids_staging` returns a persistent pinned CPU
tensor (int32 by default) that's reused across calls. Combined with
`iter_embed_tokens_pipelined`, this:

1. Pinned-memory CPU buffer eliminates per-call alloc + pin.
2. int32 H2D halves the PCIe payload vs int64.
3. CUDA-stream ping-pong overlaps H2D with the previous batch's
   forward.
4. Device ping-pong buffers eliminate device alloc per batch.

The four optimizations contribute 1.4–2.1× on top of the kernel's
base speedup (Fig. 5b ablation).
