# DNAtok Docker → Apptainer → Gadi test workflow

End-to-end workflow for building the plug-and-play image **locally**,
converting to a Singularity/Apptainer `.sif`, copying to Gadi, and
running the smoke test on **H200 (sm_90)** and **A100 (sm_80)** to
prove the image is self-contained and cross-architecture.

```
local            local            Gadi             Gadi (H200 + A100)
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────────┐
│ docker    │──→ │ apptainer │──→ │  rsync    │──→ │ submit smoke  │
│ build     │    │ build sif │    │  to gadi  │    │ test PBS x2   │
└───────────┘    └───────────┘    └───────────┘    └───────────────┘
   ~15min          ~5min             1min            ~15min × 2 archs
```

## Step 0 — prerequisites (one-time)

- Docker installed locally (`docker --version` returns 24+).
- Apptainer installed locally OR available on the target cluster
  (most HPC sites including NCI Gadi ship apptainer ≥1.3 as a module).
- Free disk: ~25 GB during build, ~10 GB for the final .sif.

### CRITICAL: architecture must match

The Docker image is **architecture-specific**. NCI Gadi's H200 and A100
nodes are **x86_64**. If you build on an aarch64 host (e.g. a GB10 /
GH200), the resulting image will NOT run on Gadi.

| Host where you build | Resulting image arch | Runs on Gadi? |
|---|---|---|
| Intel/AMD laptop, x86 cloud VM | x86_64 | ✅ |
| GB10 / GH200 / Apple Silicon | aarch64 | ❌ |
| `docker buildx --platform=linux/amd64` on aarch64 host | x86_64 (emulated) | ✅ — but CUDA wheels may fail under emulation; expect 2-3× slower build |
| GitHub Actions `ubuntu-latest` runner | x86_64 (native) | ✅ — recommended for paper release |

For the paper release the recommended path is **GitHub Actions CI**
(see `.github/workflows/docker-build.yml`) — it builds natively on
x86_64 Ubuntu runners and pushes to GHCR. Users on Gadi then run
`apptainer pull dnatok.sif docker://ghcr.io/anu-dnatok/dnatok:latest`
from the **login node** (compute nodes lack internet on Gadi).

## Step 1 — build Docker locally

From the DNAtok repo root:

```bash
docker build -t dnatok:dev -f docker/Dockerfile .
```

Expected: ~15-20 min on first build (pulls NGC base ~10GB + installs all
deps + builds Evo2 / mamba_ssm / causal_conv1d CUDA extensions).

The Dockerfile runs a build-time sanity check (`dnatok --help`); if that
fails the build aborts with a clear message.

## Step 2 — smoke-test the Docker image (local GPU)

```bash
# Mount a host HF cache so weights don't re-download on every run.
mkdir -p ~/.hf-cache
docker run --rm --gpus all \
    -v ~/.hf-cache:/work/.hf-cache \
    dnatok:dev bash /work/docker/smoke_test.sh
```

The smoke test iterates over all 7 model families (DNABERT-2, GENA-LM,
METAGENE-1, NTv3, NTv2, HyenaDNA, Evo2-1B), running `info` →
`encode` → `validate` (n=100, must be 100% match) → `bench` for each.

Exit code 0 = all 7 pass.

## Step 3 — convert to Apptainer `.sif`

```bash
apptainer build dnatok.sif docker-daemon://dnatok:dev
```

Expected: ~5 min. The `.sif` is a single immutable file (~10 GB) you
can scp around; it requires no Docker on the receiving machine.

## Step 4 — copy `.sif` (or docker tarball) to Gadi

### Option A — direct `.sif` upload (recommended if you have apptainer)

```bash
rsync -av --progress dnatok.sif \
    gadi:/g/data/te53/<account>/data/containers/dnatok.sif
```

### Option B — docker tarball (no local apptainer)

If you only have Docker locally and apptainer is only on Gadi:

```bash
# Local
docker save dnatok:dev | gzip > dnatok.tar.gz       # ~10 GB
rsync -av --progress dnatok.tar.gz \
    gadi:/g/data/te53/<account>/data/containers/

# On Gadi (compute node — apptainer build from docker-archive does NOT
# require fakeroot or internet, only file read access)
ssh gadi
module load apptainer
apptainer build /g/data/te53/<account>/data/containers/dnatok.sif \
    docker-archive:///g/data/te53/<account>/data/containers/dnatok.tar.gz
```

### Option C — GHCR pull (post-paper-release path)

If the image is published to GHCR (via the CI workflow):

```bash
# On Gadi LOGIN NODE (compute nodes lack internet)
ssh gadi
# apptainer is on compute nodes only, so we cache via docker:// → docker-archive:
# ... or use GHCR's docker-content-api directly with curl into a tarball
# (See `scripts/pull_ghcr_dnatok.sh` for the curl-based approach.)
```

Expected upload: ~10 minutes for a 10 GB image at typical 15 MB/s.

## Step 5 — submit PBS smoke tests on H200 + A100

Two PBS scripts live under `benchmarks/`:

```bash
# H200 (sm_90, Hopper) — pulls the latest dispatch + libcuda fixes
ssh gadi 'source /etc/profile && qsub /g/data/te53/<account>/workspace/sync/ANU/DNAtok/benchmarks/sif_smoke_h200.pbs'

# A100 (sm_80, Ampere) — same image, different CUDA arch
ssh gadi 'source /etc/profile && qsub /g/data/te53/<account>/workspace/sync/ANU/DNAtok/benchmarks/sif_smoke_a100.pbs'
```

Each PBS:
1. Loads apptainer module.
2. Runs `apptainer exec --nv dnatok.sif bash /work/docker/smoke_test.sh`.
3. Writes results to `results_hpc/sif_smoke_<arch>_<jobid>/`.
4. Exit code 0 ⇔ all 7 model families pass on that architecture.

ETA per arch: ~15 min wall-clock (mostly HF weight download on first run;
cached after that).

## Step 6 — interpret results

A passing run looks like (per architecture):

```
============================================================
Summary
============================================================
  passed:  7 / 7
  failed:  0 / 7
  ALL CLEAR — DNAtok image is good to ship.
```

If any of the 7 families fails, the smoke test exits non-zero and the
PBS log includes the failing model, the captured stderr, and which
sub-command (info / encode / validate / bench) failed.

## Step 7 — publish

### Option A — automated via GitHub Actions (recommended)

The workflow at `.github/workflows/docker-build.yml` builds the image
on every push to `main` and on every `v*` tag, pushing to GHCR
(`ghcr.io/<owner>/dnatok`). It uses the repo's built-in
`GITHUB_TOKEN` — no extra secrets to configure.

One-time setup:
1. Make sure GHCR is enabled for your account or org: https://github.com/settings/packages → "Improved container support" toggle.
2. Under repo `Settings` → `Actions` → `General`, set `Workflow permissions` to "Read and write permissions".
3. Push a tag to trigger the release build:
   ```bash
   git tag v0.1.0
   git push --tags
   ```
4. After the workflow completes, the image is available at
   `ghcr.io/<owner>/dnatok:v0.1.0` and `ghcr.io/<owner>/dnatok:latest`.
5. Mark the package public under https://github.com/users/<owner>/packages → dnatok → Package settings → Change visibility.

### Option B — manual local build + push

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <username> --password-stdin
docker tag dnatok:dev ghcr.io/<owner>/dnatok:0.1.0
docker push ghcr.io/<owner>/dnatok:0.1.0
docker tag dnatok:dev ghcr.io/<owner>/dnatok:latest
docker push ghcr.io/<owner>/dnatok:latest
```

### Mirror the `.sif` to Zenodo (DOI for the paper)

- Visit https://zenodo.org/uploads → New upload.
- Metadata: title `DNAtok v0.1.0 Apptainer image`, license `Apache-2.0`,
  cite the DNAtok paper.
- Upload `dnatok.sif`.
- Publish → Zenodo returns a DOI like `10.5281/zenodo.XXXXXXX`.
- Cite that DOI in the paper's "Code availability" section.

### Gadi user instructions (post-publish)

```bash
# On Gadi LOGIN NODE (has internet)
curl -L -o /g/data/te53/<account>/data/containers/dnatok.sif \
    "https://zenodo.org/record/<DOI-numeric>/files/dnatok.sif"

# OR pull via apptainer if you have it on the login node
module load apptainer
apptainer pull /g/data/te53/<account>/data/containers/dnatok.sif \
    docker://ghcr.io/<owner>/dnatok:latest

# On a compute node — no internet needed
module load apptainer
apptainer exec --nv \
    -B /g/data/te53/<account>/data/scratch/hf-cache:/work/.hf-cache \
    -B /g/data/te53/<account>/your-data:/data \
    /g/data/te53/<account>/data/containers/dnatok.sif \
    dnatok demo --model zhihan1996/DNABERT-2-117M
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker build` fails on `pip install mamba_ssm` | nvcc not on PATH during build | Re-run with `--build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3` explicitly |
| `dnatok --help` fails at build time | `setup.py` entry point not picked up | Ensure `pip install --no-cache-dir -e /work` runs before the sanity check |
| Apptainer `--nv` doesn't see the GPU | Driver mismatch host vs container | Update host driver or rebuild .sif with matching NGC version |
| `validate` reports n<100 / 100 match | Tokeniser-specific edge case (e.g. SentencePiece prefix) | Open an issue; the GPU BPE kernel handles SP marker variants, but new ones may need a fix |
| Build hits disk-space error | NGC base is huge | Mount a fresh volume; `docker system prune -a` to reclaim old layers |

## Files in this workflow

```
docker/
├── Dockerfile                  the recipe
├── requirements.txt            exact pinned dependencies
├── smoke_test.sh               in-container test of all 7 families
├── BUILD_AND_TEST.md           this file
├── README.md                   user-facing docs
└── gpu-tokenizer-headers/      vendored cuCollections/CCCL

benchmarks/
├── sif_smoke_h200.pbs          Gadi H200 smoke test
└── sif_smoke_a100.pbs          Gadi A100 smoke test

src/
├── dnatok_cli.py               the `dnatok` CLI entry point
└── ...

examples/
├── 01_hello_world.py           12-line minimal demo
├── 02_any_model.py             auto-discover any HF model
└── ...
```
