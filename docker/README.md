# DNAtok Docker image

Plug-and-play GPU tokenisation for all 21 supported genomic FM variants.

## Quick start

```bash
# Build (one-time)
docker build -t dnatok:dev -f docker/Dockerfile .

# Or pull (when we publish)
docker pull ghcr.io/[org]/dnatok:v1.0

# Run the cross-model correctness gate
# NOTE: mount HF cache as read-write (default) so transformers can
# extract dynamic modules for tokenisers like Caduceus that ship
# trust_remote_code Python files.
docker run --rm --gpus all \
    -v $(pwd)/.hf-cache:/work/.hf-cache \
    dnatok:dev bash -c "
        python3 tests/test_gputok_bpe_backend.py
"

# Run a biological case study
docker run --rm --gpus all \
    -v $(pwd)/.hf-cache:/work/.hf-cache \
    -v $(pwd)/results:/work/results \
    dnatok:dev bash bio_examples/01_evo2_clinvar_variants/run.sh
```

## What's inside

- NVIDIA NGC PyTorch 25.10 base (CUDA 12.6, cuDNN, NCCL, Triton).
- DNAtok source under `/work` with editable install.
- gpu-tokenizer headers (cuCollections + CCCL) vendored under
  `/opt/gpu-tokenizer` and exposed via `GPUTOK_DIR`.
- transformers + tokenizers + huggingface_hub.
- biopython + pysam for the biological case studies (FASTA, VCF,
  BED, ChIP-seq narrowPeak).

## What's NOT inside

- Model weights — these download to `/work/.hf-cache` on first use.
  Mount that path from the host so the cache persists between runs
  and so multiple containers share it.
- A specific CUDA arch — the kernel JIT-compiles for whatever arch
  the host GPU exposes on first use. We've tested sm_70 / sm_80 /
  sm_90 / sm_121.

## Singularity / Apptainer

```bash
singularity build dnatok.sif docker-daemon://dnatok:dev
# or, from a published image:
singularity pull docker://ghcr.io/[org]/dnatok:v1.0

apptainer run --nv -B $HF_HOME:/work/.hf-cache dnatok.sif \
    python3 tests/test_gputok_bpe_backend.py
```

## Reproducing the paper figures

```bash
# Figure 2a — bit-identical correctness gate
docker run --rm --gpus all -v $HF_HOME:/work/.hf-cache \
    dnatok:dev python3 tests/test_gputok_bpe_backend.py

# Figure 3 — cross-platform throughput
docker run --rm --gpus all -v $HF_HOME:/work/.hf-cache \
    -v $(pwd)/results:/work/results dnatok:dev \
    bash benchmarks/run_realistic_benchmark.py --warmup 5 --iters 30

# Figure 4 — case studies
for case in 01_evo2_clinvar_variants 02_ntv3_chr21_regulatory 03_dnabert2_encode_ctcf; do
    docker run --rm --gpus all -v $HF_HOME:/work/.hf-cache \
        -v $(pwd)/results:/work/results dnatok:dev \
        bash bio_examples/${case}/run.sh
done
```

## TODO before submission

- Decide on registry (GHCR vs Docker Hub).
- Add multi-arch manifest (x86_64 only for now; ARM64 if any biologist
  hosts on Grace / Hopper Grace).
- Confirm we don't ship anything non-Apache (gpu-tokenizer is
  Apache-2.0 ✓; cuCollections is Apache-2.0 ✓; CCCL is Apache-2.0 ✓).
- Confirm NGC base image redistribution is permitted (NVIDIA EULA).
