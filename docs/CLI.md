# `dnatok` command-line reference

The `dnatok` CLI is the plug-and-play entry point shipped with the
Docker / Apptainer image. Every sub-command takes a Hugging Face
model id (`--model`) and works out of the box without a Python
script.

```
dnatok <subcommand> --model <hf-id> [flags]
```

| Sub-command | Purpose |
|---|---|
| `info` | Print tokenizer family + fast path selected |
| `encode` | Encode a single DNA sequence |
| `validate` | Bit-identity check vs HF on N random sequences |
| `bench` | Pre-tokenization benchmark vs HF Rust threaded |
| `demo` | Run info + encode + validate + small bench end-to-end |
| `list-models` | Print the validated model registry |

All sub-commands accept `--device cuda|cpu` (defaults to auto-detect)
and `--json` (machine-readable output).

## `dnatok info`

Inspect which fast path DNAtok auto-discovers for a given model.

```bash
dnatok info --model zhihan1996/DNABERT-2-117M
```

Output:

```
  model                  zhihan1996/DNABERT-2-117M
  tokenizer_class        PreTrainedTokenizerFast
  vocab_size             4096
  selected_path          gpu-bpe-kernel
  device                 cuda
  kmer_k                 None
  has_ascii_lut          False
  has_gpu_bpe_backend    True
  has_cached_lmm         True
  padding_side           right
  id_pad                 0
```

`selected_path` values:
- `ascii-lut` — single-character / byte-level tokenizer (NTv3, HyenaDNA, Evo2)
- `kmer-K` — fixed-K k-mer tokenizer (NTv2 at k=6)
- `gpu-bpe-kernel` — genomic BPE via the CUDA kernel (DNABERT-2, GENA-LM, METAGENE-1)
- `cached-lmm` — fallback to the cached safe-margin lookahead encoder
- `hf-fallback` — upstream tokenizer used unchanged

## `dnatok encode`

Encode a single DNA string. Returns the token IDs (with padding stripped).

```bash
dnatok encode --model zhihan1996/DNABERT-2-117M --seq ACGTACGTACGTACGT
```

For machine-readable output:

```bash
dnatok encode --model zhihan1996/DNABERT-2-117M --seq ACGTACGTACGT --json
```

## `dnatok validate`

Bit-identity check vs Hugging Face on N random ACGT sequences.

```bash
dnatok validate --model zhihan1996/DNABERT-2-117M --n 500 --window 1024
```

Output (machine-readable with `--json`):

```json
{
  "model": "zhihan1996/DNABERT-2-117M",
  "n_sequences": 500,
  "n_match": 500,
  "n_mismatch": 0,
  "match_rate": 1.0,
  "selected_path": "gpu-bpe-kernel",
  "first_mismatch": null
}
```

Exits 0 ⇔ 100% bit-identical; non-zero otherwise.

## `dnatok bench`

Pre-tokenization throughput vs Hugging Face Rust threaded baseline.

```bash
dnatok bench --model zhihan1996/DNABERT-2-117M \
    --n 1000 --window 4096 --chunk 32 --hf-threads 8
```

Output (JSON):

```json
{
  "model": "zhihan1996/DNABERT-2-117M",
  "n_sequences": 1000, "window_bp": 4096, "chunk": 32,
  "total_bp": 4096000,
  "hf_threads": 8,
  "hf_time_s": 0.62,    "hf_mbp_per_s": 6.60,
  "dnatok_time_s": 0.21, "dnatok_mbp_per_s": 19.50,
  "speedup_dnatok_vs_hf": 2.95,
  "selected_path": "gpu-bpe-kernel",
  "device": "cuda"
}
```

**Note**: bench numbers depend on hardware (memory bandwidth most),
neighbouring CPU workloads, and the random seed used for sequence
generation. For reproducible numbers, run n=3 with `--seed`
fixed and report mean ± SD.

## `dnatok demo`

Runs `info` → `encode` → `validate` (n=50) → `bench` (small) on a
single model. The fastest end-to-end sanity check that the image works:

```bash
dnatok demo --model arcinstitute/evo2_1b_base
```

Exits 0 ⇔ image is fully functional for that model family.

## `dnatok list-models`

Prints the registry of model families tested end-to-end:

```bash
dnatok list-models
```

```
  family         hf_id                                                            path           tested_on        notes
  -------------  ---------------------------------------------------------------- -------------- ---------------- -------------------------------------------------
  DNABERT-2      zhihan1996/DNABERT-2-117M                                        gpu_bpe        H200/A100/GB10   GPU BPE kernel + cached safe-margin layer
  GENA-LM        AIRI-Institute/gena-lm-bert-base-t2t                             gpu_bpe        H200/A100/GB10   GPU BPE kernel + cached safe-margin layer
  METAGENE-1     metagene-ai/METAGENE-1                                           gpu_bpe        H200/A100/GB10   GPU BPE kernel (SentencePiece-flavoured)
  NTv3-8M        InstaDeepAI/NTv3_8M_pre                                          ascii_lut      H200/A100/GB10   ASCII byte lookup table (4-base alphabet)
  NTv2-50M       InstaDeepAI/nucleotide-transformer-v2-50m-multi-species          kmer_lut       H200/A100/GB10   6-mer lookup table with variable-length dispatch
  HyenaDNA-tiny  LongSafari/hyenadna-tiny-1k-seqlen-hf                            ascii_lut      H200/A100/GB10   Char-level ASCII LUT
  Evo2-1B        arcinstitute/evo2_1b_base                                        ascii_lut      H200/A100/GB10   Byte-level (requires the evo2 package)
```

DNAtok accepts ANY HF model id, not just the ones in this list. The
list captures models we've tested end-to-end with bit-identity gates.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | Where model weights download / cache |
| `GPUTOK_DIR` | (set in Dockerfile) | gpu-tokenizer source for the CUDA kernel JIT compile |
| `TRITON_LIBCUDA_PATH` | (set in Dockerfile) | Triton's libcuda.so.1 lookup path |
| `RAYON_NUM_THREADS` | 1 | HF Rust BPE threading |
| `TOKENIZERS_PARALLELISM` | `false` | HF Rust BPE parallelism |
| `DNATOK_DEBUG` | unset | If set, prints full tracebacks on error |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | `validate` found mismatches (non-zero `n_mismatch`) |
| 2 | Other error — re-run with `DNATOK_DEBUG=1` for full traceback |
| 130 | Interrupted (Ctrl-C) |
