# DNAtok biological case studies

Three end-to-end pipelines that exercise DNAtok on published genomic
foundation models against public ground-truth. Each reproduces a
published analysis at minutes-scale instead of hours-scale, with
token IDs bit-identical to Hugging Face's reference implementation.

**We make no new biology claims.** We make existing biology
practical to run.

| Pipeline | Model | Tokenisation | Public ground truth | Role in the paper |
|---|---|---|---|---|
| `01_evo2_clinvar_variants` | Evo2-1b (Arc Inst., Nature 2026) | Single-nucleotide | ClinVar pathogenic/benign SNVs | **Headline:** modern base-level FM where the bottleneck is severest |
| `02_ntv3_chr21_regulatory` | NTv3-650M-131kb (InstaDeepAI, Dec 2025) | Single-nucleotide | Chr21 regulatory annotation @ 131 kbp context | **Headline:** modern base-level FM at long context |
| `03_dnabert2_encode_ctcf` | DNABERT-2 (Zhihan Zhou et al., 2023) | BPE | ENCODE GM12878 CTCF ChIP-seq | **Compatibility:** the same Docker image, same one-line invocation, works on a BERT-era BPE model |

## The paper's argument

Genomic foundation model tokenisation has evolved across three
generations:

1. **K-mer (2023):** NTv2 — 6-mer tokens, vocab ~4096, low
   throughput need.
2. **BPE (2023–24):** DNABERT-2, GENA-LM, METAGENE-1 — sub-word
   tokens, vocab ~30 k, moderate throughput need.
3. **Single-base (2025–26):** NTv3, Evo2 — one token per nucleotide,
   vocab ~11, 1 Mbp contexts, maximum throughput need.

The newest models move to single-base tokenisation because biology
lives at base resolution — variant effects, splice sites, modifications,
regulatory motifs. But single-base tokenisation is also the
worst case for throughput: more tokens per base, more contexts to
process, more CPU work. DNAtok addresses this.

The three case studies cover all three generations:
- **Evo2 (1)** and **NTv3 (2)** demonstrate the modern single-base
  regime — the headline numbers.
- **DNABERT-2 (3)** demonstrates the BERT-era BPE regime — proof that
  DNAtok is plug-and-play and not a single-model trick.

## Common design principles

- **No new biology.** We score existing labelled data; we do not claim
  new findings.
- **No new training.** All three models are loaded from their official
  Hugging Face checkpoints; weights are unchanged.
- **Bit-identical correctness check** at the start of each pipeline:
  for the first 1,000 inputs, assert `DNAtok IDs == HF IDs`, and
  assert downstream model logits match HF to floating-point tolerance.
- **Public data only.** Each script downloads its inputs if absent.
- **One-command reproduction** inside the Docker image.

## Running

From inside the Docker image (see `../docker/`):

```bash
# Pull the image
docker pull ghcr.io/[org]/dnatok:v1.0

# Run any pipeline
docker run --gpus all -v $(pwd)/results:/work/results dnatok:v1.0 \
    bash bio_examples/01_evo2_clinvar_variants/run.sh
```

Each pipeline writes:
- `results/<pipeline>/hf_timing.json` — HF reference timing.
- `results/<pipeline>/dnatok_timing.json` — DNAtok timing.
- `results/<pipeline>/correctness.json` — bit-identical IDs +
  downstream-logit FP-tolerance check.
- `results/<pipeline>/output.{npz,bed,tsv}` — biological output (per
  pipeline).

## On hardware

Each pipeline runs on the same Docker image, on:
- NVIDIA GB10 (consumer; Blackwell sm_121).
- NVIDIA V100 (legacy HPC; Volta sm_70).
- NVIDIA A100 80GB (current HPC; Ampere sm_80).
- NVIDIA H200 (latest HPC; Hopper sm_90).

The kernel JIT-compiles for the host architecture on first use.

## Validation gate (separate from the three case studies)

`../tests/test_gputok_bpe_backend.py` runs a 2,242-input adversarial
correctness gate (homopolymers, all-N, Unicode-edge, CGTT counter-
example, partial-tail k-mer) plus a 400-input variable-length k-mer
gate. Both gates run inside the Docker image as part of CI.
