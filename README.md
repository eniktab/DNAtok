# DNAtok

**GPU-native tokenization for genomic foundation models. Bit-identical to Hugging Face. 2–100× faster.**

<p align="center">
  <a href="#installation"><img src="https://img.shields.io/badge/install-pip-blue" alt="pip"></a>
  <a href="#docker"><img src="https://img.shields.io/badge/docker-ready-2496ED?logo=docker" alt="docker"></a>
  <a href="#supported-models"><img src="https://img.shields.io/badge/models-21%20variants-success" alt="models"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="license"></a>
  <a href="#benchmarks"><img src="https://img.shields.io/badge/benchmark-V100%20%7C%20A100%20%7C%20H200-orange" alt="benchmark"></a>
</p>

Modern genomic foundation models — Evo2 (Nature 2026), NTv3 (2025), DNABERT-2, HyenaDNA, Caduceus, GENA-LM, METAGENE-1, and the rest — have moved tokenization to **single-nucleotide resolution** to capture biology at base granularity. At single-base resolution, on high-throughput workloads, the CPU-bound HF tokenizer can no longer keep up — it becomes the rate-limiting step that **starves the GPU**, dropping end-to-end throughput well below what the hardware could deliver. DNAtok removes that bottleneck by tokenizing on the GPU and produces token IDs bit-identical to Hugging Face's reference implementation on every supported model.

## 🚀 Try it in 30 seconds — no installation

The Docker image is fully self-contained: NGC PyTorch 25.11 base + DNAtok + all dependencies pinned. Give it a Hugging Face model ID and it tokenizes, validates bit-identity vs HF, and benchmarks throughput — no other setup.

```bash
# Pull the image (one-time, ~10 GB compressed)
docker pull ghcr.io/anu-dnatok/dnatok:latest

# Run an end-to-end demo on any of the 7 supported families
docker run --rm --gpus all \
    -v ~/.cache/huggingface:/work/.hf-cache \
    ghcr.io/anu-dnatok/dnatok:latest \
    dnatok demo --model zhihan1996/DNABERT-2-117M
```

Output (DNABERT-2 on H200):

```
[1/4] tokenizer info: path=gpu-bpe-kernel, vocab=4096, device=cuda
[2/4] encode a sample sequence: 12 tokens
[3/4] bit-identity vs HF (n=50, random ACGT, win≤256bp): PASS
[4/4] tiny benchmark (n=100, win=1024, chunk=32, HF threads=4):
        hf       30.0 ms   ( 3.4 Mbp/s)
        dnatok    7.2 ms   (14.2 Mbp/s)
        speedup    4.2x
```

Swap `zhihan1996/DNABERT-2-117M` for any supported HF model (`dnatok list-models`) — no rebuild needed; weights download to the mounted cache on first use.

For HPC clusters that disallow Docker, use the Apptainer/Singularity `.sif`:

```bash
apptainer pull dnatok.sif docker://ghcr.io/anu-dnatok/dnatok:latest
apptainer exec --nv -B ~/.cache/huggingface:/work/.hf-cache dnatok.sif \
    dnatok demo --model arcinstitute/evo2_1b_base
```

## Contents

- [Why DNAtok](#why-dnatok)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Supported models](#supported-models)
- [Usage by model family](#usage-by-model-family)
- [Docker](#docker)
- [Benchmarks](#benchmarks)
- [How it works](#how-it-works)
- [Reproducing the paper](#reproducing-the-paper)
- [Citation](#citation)
- [License](#license)

## Why DNAtok

| Tokenization era | Example models | Tokens / base | Typical throughput pain |
|---|---|---|---|
| 6-mer (2023) | NTv2 | 0.17 | Mild — short reads only |
| BPE (2023–24) | DNABERT-2, GENA-LM, METAGENE-1 | ~0.25 | Moderate — long reads |
| **Single-base (2025–26)** | **NTv3, Evo2** | **1.0** | **Severe — every workload** |

At single-base resolution there are **~6× more tokens per base** than at 6-mer (1 token/base vs ~0.17 token/base) and contexts have grown to **1 Mbp**. The GPU model forward pass is still the largest single cost of inference, but for high-throughput pipelines the CPU-bound tokenizer can no longer produce token batches fast enough to keep the GPU busy — the GPU **starves**. DNAtok removes that starvation by tokenizing on the GPU directly. Same `AutoTokenizer` API, same outputs — much faster end-to-end.

## Installation

### From PyPI _(once published)_

```bash
pip install dnatok
```

### From source

```bash
git clone https://github.com/[org]/DNAtok.git
cd DNAtok
pip install -e .
```

### Requirements

| | Version |
|---|---|
| CUDA | 11.8+ (12.x recommended) |
| GPU compute capability | sm_70 (V100) through sm_120 (Blackwell) |
| PyTorch | 2.1+ |
| Python | 3.10+ |
| Transformers | 4.46+ |

The CUDA kernel JIT-compiles on first use for your specific GPU architecture. No precompiled wheels needed.

## Getting models

DNAtok is a drop-in replacement for the tokenization step. The models themselves live on Hugging Face Hub — you download them once and DNAtok loads them from the local cache. Three equivalent ways to fetch a model:

### 1. Implicit download via `from_pretrained` (most common)

```python
from transformers import AutoTokenizer, AutoModel
# This downloads weights + tokenizer files to ~/.cache/huggingface/hub
# the first time, then loads from cache on subsequent runs.
tok = AutoTokenizer.from_pretrained("InstaDeepAI/NTv3_650M_post_131kb",
                                     trust_remote_code=True)
model = AutoModel.from_pretrained("InstaDeepAI/NTv3_650M_post_131kb",
                                   trust_remote_code=True).cuda()
```

### 2. Explicit download via `huggingface-cli` (recommended for clusters)

```bash
pip install -U "huggingface_hub[cli]"

# Pre-download the tokenizer + weights into a chosen cache:
export HF_HOME=/path/to/your/hf-cache
huggingface-cli download InstaDeepAI/NTv3_650M_post_131kb
huggingface-cli download arcinstitute/evo2_1b_base
huggingface-cli download zhihan1996/DNABERT-2-117M
```

Then in Python:

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"            # optional: enforce offline
os.environ["HF_HOME"] = "/path/to/your/hf-cache"
```

### 3. Bulk download via a one-liner

```python
# Pull the canonical 14-model set into the current HF cache.
from huggingface_hub import snapshot_download
for repo in [
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "InstaDeepAI/NTv3_8M_pre", "InstaDeepAI/NTv3_100M_pre",
    "InstaDeepAI/NTv3_100M_post", "InstaDeepAI/NTv3_650M_post",
    "InstaDeepAI/NTv3_650M_post_131kb",
    "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "LongSafari/hyenadna-small-32k-seqlen-hf",
    "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3",
    "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
    "zhihan1996/DNABERT-2-117M",
    "AIRI-Institute/gena-lm-bert-base-t2t",
    "metagene-ai/METAGENE-1",
    "arcinstitute/evo2_1b_base",
]:
    snapshot_download(repo)
```

`snapshot_download` is idempotent — it skips already-cached files, so running it twice is a no-op.

### Models with custom tokenizers (Evo2 + NTv3)

Some published genomic FMs ship a `config.json` without a `model_type` field (Evo2) or require `trust_remote_code` for a custom tokenizer class (NTv3). The project ships a single loader that handles both:

```python
from benchmarks.tokenizer_adapters import load_hf_tokenizer
hf_tok = load_hf_tokenizer("arcinstitute/evo2_1b_base")  # works for any supported family
```

Use `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` for ordinary HF models, and `load_hf_tokenizer` when you want the same code path to work transparently across Evo2, NTv3 and the rest.

## Quick start

Bit-identical to Hugging Face. The only change is wrapping `AutoTokenizer` with `DNATok`.

```python
import torch
from transformers import AutoTokenizer, AutoModel
from dna_tokenizer import DNATok

# Load any supported genomic foundation model
hf_tok = AutoTokenizer.from_pretrained(
    "InstaDeepAI/NTv3_650M_post_131kb", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "InstaDeepAI/NTv3_650M_post_131kb", trust_remote_code=True).cuda()

# Wrap the tokenizer — DNAtok auto-discovers the right fast path
dnatok = DNATok(model)
dnatok.discover()

# Tokenize and embed a batch
seqs = ["ACGT" * 2048 for _ in range(64)]
ids = dnatok.encode_batch_to_ids(seqs)         # [B, T_max] padded
with torch.no_grad():
    out = model(ids.cuda()).last_hidden_state  # [B, T_max, D]
```

The output `ids` tensor matches `hf_tok(seqs)["input_ids"]` exactly (after padding alignment). For verification, see [the correctness gate](#how-it-works).

## Supported models

DNAtok supports 21 published variants of the seven major genomic FM families:

| Family | Model variants | Tokenization | Context | Year |
|---|---|---|---|---|
| **Evo2** | `arcinstitute/evo2_1b_base` (+ 7B / 40B) | Single-nucleotide | 1 Mbp | 2026 |
| **NTv3** | `InstaDeepAI/NTv3_{8M,100M,650M}_{pre,post}`, `..._131kb` | Single-nucleotide | up to 1 Mbp | 2025 |
| **HyenaDNA** | `LongSafari/hyenadna-{tiny-1k,small-32k,medium-160k,medium-450k,large-1m}-seqlen-hf` | Char-level LUT | 1k–1Mbp | 2023 |
| **Caduceus** | `kuleshov-group/caduceus-ph_seqlen-{1k,131k}_...`, `..._ps_...` | Char-level LUT | 1k–131k | 2024 |
| **NTv2** | `InstaDeepAI/nucleotide-transformer-v2-{50M,500M}-multi-species` | 6-mer LUT | 1k | 2023 |
| **DNABERT-2** | `zhihan1996/DNABERT-2-117M` | BPE | 4k | 2023 |
| **GENA-LM** | `AIRI-Institute/gena-lm-bert-base-t2t` | BPE | ~36k | 2023 |
| **METAGENE-1** | `metagene-ai/METAGENE-1` | BPE | 32k | 2024 |

DNAtok's auto-discovery probes any Hugging Face tokenizer at `discover()` time and routes it to the appropriate GPU fast path:

```python
dnatok = DNATok(model)
dnatok.discover()  # → routes to k-mer, BPE, or single-base path automatically
```

## Usage by model family

| Family | Runnable example | What it shows |
|---|---|---|
| **Generic** | [`examples/quickstart.py`](examples/quickstart.py) | Wrap any HF tokenizer; pick a model with `--model`. |
| **Evo2** | [`examples/evo2_variant_effect.py`](examples/evo2_variant_effect.py) | Tokenize a 4 kbp ref + alt SNV window. |
| **NTv3** | [`examples/ntv3_long_context.py`](examples/ntv3_long_context.py) | Tokenize 32 kbp regulatory-scan windows. |
| **All published genomic FMs** | [`bio_examples/`](bio_examples/) | Case-study pipelines (ClinVar / chr21 / ENCODE). |

### NTv3 (single-nucleotide, long-context)

```python
hf_tok = AutoTokenizer.from_pretrained(
    "InstaDeepAI/NTv3_650M_post_131kb", trust_remote_code=True)
dnatok = DNATok(model); dnatok.discover()

# Long-context regulatory scan
chr21_windows = [genome.fetch("chr21", i, i + 131_072)
                 for i in range(0, len(chr21), 1000)]
ids = dnatok.encode_batch_to_ids(chr21_windows)  # ~25× faster than HF
```

### Evo2 (single-nucleotide, variant effect)

```python
# Evo2 ships a byte-level tokenizer; use the project loader
from benchmarks.tokenizer_adapters import load_hf_tokenizer
hf_tok = load_hf_tokenizer("arcinstitute/evo2_1b_base")
dnatok = DNATok(model); dnatok.discover()

# Score a ClinVar variant
ref_window = genome.fetch("chr1", pos - 2048, pos + 2048)
alt_window = ref_window[:2048] + alt + ref_window[2049:]
ids = dnatok.encode_batch_to_ids([ref_window, alt_window])
log_lik = model(ids.cuda()).logits.log_softmax(-1)  # variant effect: Δ log_lik
```

### DNABERT-2 (BPE, sub-word units)

```python
hf_tok = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
dnatok = DNATok(model); dnatok.discover()
ids = dnatok.encode_batch_to_ids(seqs)  # BPE on GPU
```

### NTv2 (6-mer)

```python
hf_tok = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
dnatok = DNATok(model); dnatok.discover()
ids = dnatok.encode_batch_to_ids(seqs)  # k-mer LUT path
```

### HyenaDNA / Caduceus / GENA-LM / METAGENE-1

Same wrapping pattern — `DNATok(model).discover()`, then `encode_batch_to_ids(seqs)`. The auto-discovery picks the correct kernel path for each family.

## Docker

The Docker image runs unchanged from a consumer Blackwell GB10 to an HPC H200. Same image, same `.sif`, validated on sm_80 (A100), sm_90 (H200) and sm_120 (GB10).

```bash
# Pull (once published)
docker pull ghcr.io/anu-dnatok/dnatok:latest

# Or build locally (~15-20 min, see docker/BUILD_AND_TEST.md)
docker build -t dnatok:dev -f docker/Dockerfile .

# CLI: any sub-command takes --model <HF id>
docker run --rm --gpus all -v ~/.cache/huggingface:/work/.hf-cache \
    dnatok:dev dnatok info     --model zhihan1996/DNABERT-2-117M
docker run --rm --gpus all -v ~/.cache/huggingface:/work/.hf-cache \
    dnatok:dev dnatok encode   --model zhihan1996/DNABERT-2-117M --seq ACGTACGT
docker run --rm --gpus all -v ~/.cache/huggingface:/work/.hf-cache \
    dnatok:dev dnatok validate --model zhihan1996/DNABERT-2-117M --n 500
docker run --rm --gpus all -v ~/.cache/huggingface:/work/.hf-cache \
    dnatok:dev dnatok bench    --model zhihan1996/DNABERT-2-117M --n 1000 --window 4096
docker run --rm --gpus all -v ~/.cache/huggingface:/work/.hf-cache \
    dnatok:dev dnatok demo     --model zhihan1996/DNABERT-2-117M  # all-in-one
docker run --rm dnatok:dev dnatok list-models                     # registry

# Run an interactive Python session inside the image
docker run --rm --gpus all -it -v ~/.cache/huggingface:/work/.hf-cache \
    dnatok:dev bash
```

### Singularity / Apptainer (HPC)

```bash
# Convert from Docker
apptainer build dnatok.sif docker-daemon://dnatok:dev

# Or pull a published image
apptainer pull dnatok.sif docker://ghcr.io/anu-dnatok/dnatok:latest

# Use exactly like Docker — the `dnatok` CLI is on $PATH
apptainer exec --nv -B ~/.cache/huggingface:/work/.hf-cache dnatok.sif \
    dnatok demo --model arcinstitute/evo2_1b_base
```

The kernel JIT-compiles for the host's CUDA architecture on first use. We have validated **sm_80 (A100), sm_90 (H200), and sm_120 (GB10)** as part of the paper. V100 (sm_70) is excluded — NGC PyTorch 25.10/25.11 dropped sm_70 under CUDA 13.

### Built-in smoke test

`docker/smoke_test.sh` iterates over all 7 supported families inside the image and validates `info → encode → validate(n=100) → bench` for each. Exit code 0 ⇔ all 7 pass. See `docker/BUILD_AND_TEST.md` for the full Docker → Apptainer → Gadi workflow.

## Benchmarks

End-to-end tokenisation speedup vs HF native, measured across all 19 supported model variants × 8 realistic DNA workloads (Illumina/PacBio/Nanopore/gene-models/clinical-mix/GC-20/GC-65/poly-A) × 2 batch sizes on three HPC GPUs:

| Model class | Examples | H200 (median) | A100 (median) | V100 (median) | V100 (max) |
|---|---|---|---|---|---|
| **LUT-char** (single-base & char-level) | Evo2, NTv3, HyenaDNA, Caduceus | **182×** | **212×** | **309×** | **794×** |
| **LUT-kmer** | NTv2-50M, NTv2-500M | 44× | 59× | 53× | 122× |
| **BPE** | DNABERT-2, GENA-LM, METAGENE-1 | **2.3–4.4×** | 1.05× | 1.01× | 1.29× |

Headline pattern matches the paper's thesis: speedup grows monotonically as the field moves from BPE → k-mer → single-base tokenisation, and as the CPU baseline gets slower (V100 wins biggest in *relative* terms because its HF baseline is most starved). LUT-kmer ties HF on truly variable-length workloads (illumina_short, nanopore_long) and wins ~100× on fixed-length scenarios.

Raw timing tables: [`results_hpc/realistic_{h200,a100,v100}/`](results_hpc/). Per-class summary: [`results_hpc/summaries/`](results_hpc/summaries/).

**Correctness:** 2,242 adversarial inputs (homopolymers up to 32 kbp, all-N, mixed case, CGTT counter-example, Unicode-edge, partial-tail k-mer) + 400 variable-length k-mer inputs + 8 variable-length single-base inputs — **bit-identical to Hugging Face on every supported model**.

Correctness: 2,242 adversarial inputs (homopolymers up to 32 kbp, all-N, mixed case, CGTT counter-example, Unicode-edge, partial-tail k-mer) + 400 variable-length k-mer inputs — **bit-identical to Hugging Face on every supported model**.

## How it works

DNAtok routes each tokenizer through one of three GPU fast paths:

1. **Single-base / character LUT** (Evo2, NTv3, HyenaDNA, Caduceus).
   ASCII byte → token ID via a 256-entry lookup table; vectorized over the whole batch.

2. **K-mer LUT** (NTv2). A precomputed table maps every `k`-character window to its token ID. Variable batch lengths are handled by per-length-group dispatch with the partial-tail extension.

3. **BPE Algorithm-1** (DNABERT-2, GENA-LM, METAGENE-1). An entry-pool bucket scheduler runs Hugging Face's reference Algorithm-1 on the GPU using a doubly-linked-list of live positions, CUB BlockMergeSort within each rank, and an optional speculative multi-rank inner loop (see Methods in the paper) for additional speedup on long reads. This GPU BPE kernel is the primary BPE path and runs 2.3–4.4× faster than Hugging Face Rust threaded on H200 with bit-identical output. A complementary cached safe-margin lookahead encoder is built alongside for non-SentencePiece BPE tokenisers (DNABERT-2, GENA-LM) and serves as a streaming-cache fallback for workloads with high corpus-level sequence repetition.

All three paths produce token IDs **bit-identical** to Hugging Face's tokenizers reference (after stripping `id_pad`); the BPE path is validated against an adversarial 2,242-input gate.

## Reproducing the paper

Each main figure has a single-command reproduction:

```bash
# Figure 2a — correctness gate
docker run --rm --gpus all -v $HOME/.cache/huggingface:/work/.hf-cache dnatok:dev \
    python3 -m pytest tests/test_gputok_bpe_backend.py -q

# Figure 3 — cross-platform throughput
docker run --rm --gpus all -v $HOME/.cache/huggingface:/work/.hf-cache \
    -v $(pwd)/results:/work/results dnatok:dev \
    bash benchmarks/run_realistic_benchmark.py --warmup 2 --iters 20

# Figure 4 — biological case studies (Evo2 ClinVar, NTv3 chr21, DNABERT-2 ENCODE)
for case in 01_evo2_clinvar_variants 02_ntv3_chr21_regulatory 03_dnabert2_encode_ctcf; do
    docker run --rm --gpus all -v $HOME/.cache/huggingface:/work/.hf-cache \
        -v $(pwd)/results:/work/results dnatok:dev \
        python3 bio_examples/${case}/run.py
done
```

Sample correctness output (each case study):

```
============================================================
Pipeline validation: NTv3-650M-131kb (chr21 long-context tiling)
============================================================
  inputs: n=20  lens(min/med/max)=(16384, 16384, 16384)
  HF tokenize:        112.47 ms
  DNAtok tokenize:      4.19 ms
  speedup:             26.84x
  correctness:     BIT-IDENTICAL
============================================================
```

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/CLI.md`](docs/CLI.md) | Full `dnatok` CLI reference — every sub-command + env vars |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How discover-then-execute routes each tokenizer; BPE dispatch details |
| [`docker/BUILD_AND_TEST.md`](docker/BUILD_AND_TEST.md) | Docker → Apptainer .sif → Gadi end-to-end build/test workflow |
| [`docker/smoke_test.sh`](docker/smoke_test.sh) | In-container test for all 7 supported families |
| [`paper_content/`](paper_content/) | Paper drafts + supplementary notes (gitignored on public repo) |

## Citation

If you use DNAtok, please cite:

```bibtex
@article{dnatok2026,
  title  = {DNAtok: a GPU-native tokenizer for genomic foundation models},
  author = {Niktab, M. and {DNAtok consortium}},
  journal= {Nature Methods},
  year   = {2026},
  note   = {In submission}
}
```

The Docker image and Apptainer `.sif` are mirrored to Zenodo for
permanence (DOI to be added at publication).

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgements

DNAtok's GPU BPE kernel re-uses the cuCollections and CCCL (CUB / Thrust / libcudacxx) header libraries vendored from the [gpu-tokenizer](https://github.com/gpu-tokenizer/gpu-tokenizer) project, both Apache 2.0. The cross-platform benchmarks ran on the Australian National Computational Infrastructure (NCI) Gadi cluster under project `te53`.
