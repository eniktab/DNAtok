# DNAtok — plug-and-play GPU tokenisation for genomic foundation models.
#
# Base: NVIDIA's official PyTorch container. NGC PyTorch 25.11 matches
# the validated pinned environment in `env_detail/audit/reference.json`
# (torch 2.10.0a0+...nv25.11, triton 3.7.0, CUDA 13). The same image
# works unchanged on H200 (sm_90), A100 (sm_80), and consumer Blackwell
# GB10 (sm_120). V100 (sm_70) is excluded — NGC 25.10/25.11 dropped
# sm_70 support under CUDA 13.
#
# Image size: ~20 GB on disk (base is ~19.5 GB; we add ~400 MB of
# headers + source + Python deps + evo2 CUDA extensions).
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="DNAtok"
LABEL org.opencontainers.image.description="GPU-native tokenisation for genomic foundation models (Evo2, NTv3, DNABERT-2, GENA-LM, METAGENE-1, NTv2, HyenaDNA). Bit-identical to Hugging Face."
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/anu-dnatok/dnatok"

WORKDIR /work

# ---- Python deps (exact pins to reference.json) ---------------------
# Two-stage install so the Evo2 stack (mamba_ssm / causal_conv1d /
# evo2) can use the parent env's torch — those wheels' upstream
# setup.py imports torch at build time, which fails inside pip's
# default PEP 517 isolated build env. We bypass isolation via
# `--no-build-isolation` for that subset only.
COPY docker/requirements.txt        /tmp/requirements.txt
COPY docker/requirements-evo2.txt   /tmp/requirements-evo2.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir --no-build-isolation -r /tmp/requirements-evo2.txt

# ---- DNAtok source ---------------------------------------------------
# Copy the package + bench + bio examples. We do NOT bake model weights
# into the image — they live in the HF cache (mountable from host).
COPY src/                 /work/src/
COPY benchmarks/          /work/benchmarks/
COPY bio_examples/        /work/bio_examples/
COPY examples/            /work/examples/
COPY tests/               /work/tests/
COPY setup.py             /work/setup.py
COPY README.md            /work/README.md
COPY LICENSE              /work/LICENSE

# ---- gpu-tokenizer headers ------------------------------------------
# DNAtok's BPE kernel needs cuCollections + cccl headers from the
# gpu-tokenizer repo. The headers total ~100 MB so we don't vendor
# them in the DNAtok repo; instead we clone the upstream public repo
# (Apache-2.0) at a known-good commit at image-build time. The image
# is still self-contained at runtime — no network access needed
# during inference; only at build.
ARG GPU_TOKENIZER_REPO=https://github.com/gpu-tokenizer/gpu-tokenizer
ARG GPU_TOKENIZER_REF=main
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    git clone --recurse-submodules --depth 1 --branch "${GPU_TOKENIZER_REF}" \
        "${GPU_TOKENIZER_REPO}" /opt/gpu-tokenizer && \
    rm -rf /opt/gpu-tokenizer/.git /opt/gpu-tokenizer/externals/*/.git
ENV GPUTOK_DIR=/opt/gpu-tokenizer

# ---- Editable install of DNAtok + CLI entrypoint --------------------
# `pip install -e .` registers the `dnatok` console_script (see setup.py)
# so `dnatok demo --model <hf-id>` works on the PATH.
RUN pip install --no-cache-dir -e /work

# ---- Triton libcuda link path (Apptainer-only override) -------------
# Under Docker `--gpus all`, NGC's default LD_LIBRARY_PATH already finds
# both libcuda (host-bound by the NVIDIA container toolkit) and libcudart
# (in /usr/local/cuda/lib64). Adding the compat paths up-front triggers
# CUDA_ERROR_SYSTEM_DRIVER_MISMATCH (the compat libs require a specific
# driver minor that may not match the host).
#
# Apptainer/Singularity does NOT do the same driver binding by default;
# the host driver lives at /.singularity.d/libs/. The Gadi runbook §10b
# describes a wrapper that adds those paths AT RUN TIME (sif_smoke_*.pbs).
# We keep the Triton-specific env minimal here so it does not interfere
# with Docker.
ENV TRITON_LIBCUDA_PATH="/.singularity.d/libs:/usr/local/cuda/compat/lib.real:/usr/lib/aarch64-linux-gnu:/usr/lib/x86_64-linux-gnu"

# ---- Cache + output locations ---------------------------------------
ENV HF_HOME=/work/.hf-cache
ENV TORCH_EXTENSIONS_DIR=/work/.torch-extensions
ENV TRANSFORMERS_VERBOSITY=warning
ENV PYTHONUNBUFFERED=1
RUN mkdir -p ${HF_HOME} ${TORCH_EXTENSIONS_DIR} /work/results

# ---- Sanity check at build time -------------------------------------
# Verifies imports succeed inside the image. Does NOT touch GPU (build
# environments are typically CPU-only).
RUN python3 -c "import torch, transformers, dnatok_compat; from dna_tokenizer import DNATok; print('[dnatok] build sanity import OK')" && \
    dnatok --help > /dev/null && echo "[dnatok] CLI entry point registered"

# ---- Entrypoint ------------------------------------------------------
# Default command runs the CLI; override at runtime to drop into a shell
# or run a custom Python script.
#
#   docker run --gpus all -v $HF_HOME:/work/.hf-cache <image> \
#       dnatok demo --model zhihan1996/DNABERT-2-117M
#
#   docker run --gpus all -it <image> bash
ENV PYTHONPATH=/work:/work/src
WORKDIR /work
ENTRYPOINT []
CMD ["dnatok", "--help"]
