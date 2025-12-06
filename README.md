# DNATok

**High-performance GPU-accelerated DNA sequence tokenization for genomic foundation models**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

DNATok provides **34-244x speedup** over standard tokenization for DNA foundation models through intelligent runtime optimization, vectorized operations, and GPU acceleration.

---

## Table of Contents

- [Features](#features)
- [Performance](#performance)
- [Installation](#installation)
- [Examples](#examples)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Supported Models](#supported-models)
- [Benchmarks](#benchmarks)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Features

### Core Capabilities

- **GPU-Accelerated**: Device-side lookup tables (LUTs) eliminate CPU→GPU tokenization bottleneck
- **K-mer Fast Path**: Automatic detection and optimization for k-mer tokenizers (k=3-6)
- **Runtime Discovery**: Builds ASCII/k-mer LUTs from any tokenizer at runtime—no manual configuration
- **Pipelined Execution**: Overlaps H2D copies with embedding compute using CUDA streams
- **Robust Fallback**: Gracefully handles edge cases by falling back to tokenizer when needed
- **Auto-Tuning**: Dynamically adjusts micro-batch sizes to avoid 32-bit index overflow and OOM

### Encoding Paths

DNATok automatically selects the fastest path:

| Path | Description | Speedup | Use Case |
|------|-------------|---------|----------|
| **Bytes Path** | GPU LUT mapping | **68-244x** | Character-level tokenizers on GPU |
| **K-mer Path** | Optimized k-mer encoding | **50-150x** | K-mer tokenizers (NT, others) |
| **IDs Path** | CPU staging with overlap | **34-43x** | Fallback for complex tokenizers |
| **Tokenizer Fallback** | Native tokenizer | **1x Baseline** | Unsupported edge cases |

---

## Performance

### Real-World Benchmarks

Tested on NVIDIA A100 with production genomic foundation models:

| Model | Scenario | Batch Size | Seq Length | Speedup | Tokens/sec |
|-------|----------|------------|------------|---------|------------|
| **Nucleotide Transformer** | Throughput | 32 | 1024 | **37.3x** | 2.1B tok/s |
| **HyenaDNA** | Throughput | 64 | 1024 | **100.1x** | 3.5B tok/s |
| **Evo2** | Latency | 1 | 1024 | **24.2x** | 56M tok/s |
| **MockTokenizer** | Large Batch | 8192 | 512 | **200.8x** | 2.1B tok/s |

> **Note**: Speedups measured as end-to-end tokenization+embedding vs. baseline tokenizer approach.

### Scaling Characteristics

- **Batch Size**: Larger batches → higher speedup (up to 200x at B=8192)
- **Sequence Length**: Scales linearly; bytes path particularly efficient for long sequences
- **K-mer Models**: 50-150x speedup with automatic k-mer structure detection

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- CUDA 11.7+ (for GPU acceleration)

### From Source

```bash
git clone https://github.com/yourusername/DNATok.git
cd DNATok
pip install -e .
```

### Dependencies

Core dependencies (installed automatically):
```
torch>=2.0.0
numpy>=1.20.0
```

Optional dependencies for examples:
```
transformers>=4.30.0  # For Hugging Face models
evo2                  # For Evo2 model support
```

---

## Examples

- `examples/nt_transformer_demo.py`: Run Nucleotide Transformer end-to-end and benchmark DNATok vs. the baseline tokenizer
- `examples/hyena_demo.py`: Demonstrate bytes-path acceleration for HyenaDNA character-level tokenization
- `examples/evo2_demo.py`: Use the k-mer path with Evo2 and validate against the reference tokenizer

See the `examples/` directory for usage and configuration details.

---

## Quick Start

### Nucleotide Transformer Example

```bash
# Install dependencies
pip install torch transformers

# Set model path (or use HF hub)
export NT_MODEL_PATH=/path/to/nucleotide-transformer-2.5b-1000g

# Run demonstration
python examples/nt_transformer_demo.py
```

The demo will:
1. Load the Nucleotide Transformer model
2. Run DNATok discovery to build LUTs
3. Verify correctness against baseline tokenizer
4. Benchmark and report speedup

### Minimal Code Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dna_tokenizer import DNATok

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    trust_remote_code=True
)
model = AutoModelForMaskedLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    trust_remote_code=True
).eval().to("cuda")

# Create adapter for DNATok
class ModelAdapter(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self._embed = model.get_input_embeddings()
    
    def embed_tokens(self, input_ids):
        return self._embed(input_ids)

adapter = ModelAdapter(model, tokenizer)

# Initialize DNATok and discover tokenizer structure
dna_tok = DNATok(adapter)
dna_tok.discover()

# Encode sequences (all must be equal length)
sequences = ["ACGTACGTACGT" * 85] * 32  # 32 sequences of length 1024
embeddings_iter = dna_tok.embed_from_strings(
    sequences,
    emb_batch=32,
    device="cuda",
    path="auto"  # Automatically select fastest path
)

# Get embeddings
embeddings = torch.cat(list(embeddings_iter), dim=0)
print(f"Embeddings shape: {embeddings.shape}")  # [32, 1024, hidden_dim]
```

---

## Usage

### Advanced Configuration

```python
from dna_tokenizer import DNATok

# Initialize with custom parameters
dna_tok = DNATok(
    embedder=adapter,
    ids_max_tokens_per_call=4_194_304,  # Max tokens per embedding call
    prefer_int32_h2d=True,              # Use int32 H2D for bandwidth savings
    overlap_h2d_compute=True,           # Pipeline H2D and compute
    force_fp32_outputs=False,           # Allow fp16 outputs for speed
    normalize_case=True,                # Force uppercase normalization
    handle_invalid_chars=True           # Map invalid chars to 'N'
)

# Discover tokenizer structure
dna_tok.discover()

# Check discovered path
if dna_tok.kmer_k:
    print(f"K-mer tokenizer detected (k={dna_tok.kmer_k})")
elif dna_tok.ascii_lut is not None:
    print("Character-level tokenizer detected")
```

### Path Selection

```python
# Auto (recommended): Automatically select fastest available path
embeddings = dna_tok.embed_from_strings(seqs, emb_batch=32, path="auto")

# Bytes path: Force GPU-side LUT mapping (fastest for char-level)
embeddings = dna_tok.embed_from_strings(seqs, emb_batch=32, path="bytes")

# IDs path: Force CPU staging with optional overlap
embeddings = dna_tok.embed_from_strings(seqs, emb_batch=32, path="ids")
```

### Handling Edge Cases

```python
# Normalize case for mixed-case inputs
dna_tok = DNATok(adapter, normalize_case=True)

# Handle invalid characters gracefully
dna_tok = DNATok(adapter, handle_invalid_chars=True)

# Both together for maximum robustness
dna_tok = DNATok(adapter, normalize_case=True, handle_invalid_chars=True)
dna_tok.discover()

# Now accepts: "AcGtXnNn" → "ACGTNNN" (X mapped to N)
```

---

## Supported Models

DNATok has been tested and validated with:

### Character-Level Tokenizers
- **HyenaDNA** (all sizes: tiny-1k to large-1m)
- **Custom DNA vocabularies** (ACGTN)

### K-mer Tokenizers
- **Nucleotide Transformer** (all variants: 500M-2.5B, k=6)
- **Evo2** (7B parameter model, k=6)
- **Custom k-mer models** (k=3,4,5,6)

### Compatibility

DNATok automatically adapts to:
- Hugging Face `transformers` models
- Custom tokenizers with `encode()` or `tokenize()` methods
- Models with `embed_tokens()` or `get_input_embeddings()`

---

## Benchmarks

### Running Benchmarks

#### Simple Benchmark (Mock Tokenizer)
```bash
python benchmarks/benchmark_ids_path_vs_tokenizer.py --kmer 1 --reps 3
```

Output includes baseline comparison and speedup:
```
Config: standard (4096 x 512)
  Baseline (HF)                   :    9,435,641 tok/s
  DNATok (bytes path)             :  646,845,743 tok/s (68.55x speedup)
```

#### Real Model Benchmarks
```bash
# Configure model paths
export NT_MODEL_PATH=/path/to/nucleotide-transformer
export HYENA_MODEL_PATH=/path/to/hyenadna

# Run comprehensive benchmark
python tests/benchmark_real_models.py
```

### Benchmark Scenarios

- **Latency**: Single sequence (B=1), typical context length
- **Throughput**: Large batch sizes (B=16-64)
- **Long Sequence**: Extended contexts (T=4096-8192)

Results saved to `results/` directory as CSV and JSON.

---

## API Reference

### DNATok Class

```python
class DNATok:
    """GPU-accelerated DNA tokenization with automatic optimization."""
    
    def __init__(
        self,
        embedder: object,
        ids_max_tokens_per_call: int = 4_194_304,
        prefer_int32_h2d: bool = True,
        overlap_h2d_compute: bool = True,
        force_fp32_outputs: bool = True,
        normalize_case: bool = False,
        handle_invalid_chars: bool = False,
        strict_lut_check: bool = True,
        logger: Optional[logging.Logger] = None
    )
```

#### Key Methods

##### `discover() -> None`
Analyzes tokenizer and builds optimization structures (LUTs, k-mer tables).

##### `embed_from_strings(seqs, emb_batch, device, path="auto") -> Iterator[Tensor]`
End-to-end tokenization and embedding.

**Parameters:**
- `seqs`: List of equal-length DNA sequences
- `emb_batch`: Micro-batch size for embedding calls
- `device`: Target device (`"cuda"`, `"cuda:0"`, `"cpu"`)
- `path`: Encoding path (`"auto"`, `"bytes"`, `"ids"`)

**Returns:** Iterator of embedding tensors (allows streaming large batches)

##### `encode_batch_to_ids(seqs) -> Tensor`
Tokenize sequences to token IDs (CPU).

**Parameters:**
- `seqs`: List of equal-length strings

**Returns:** `[B, T]` tensor of token IDs

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Test Coverage

- **Correctness**: DNATok vs baseline tokenizer equivalence
- **Path Consistency**: Bytes path matches IDs path
- **Fallback Behavior**: Graceful degradation on errors
- **Edge Cases**: Mixed case, invalid chars, empty strings
- **K-mer Handling**: K-mer detection and unsupported k-mer fallback

### Specific Test Suites

```bash
# Benchmark correctness validation
pytest tests/test_benchmark_correctness.py -v

# IDs path equivalence
pytest tests/test_ids_path_equivalence.py -v

# Fallback behavior
pytest tests/test_bytes_path_fallback.py -v
pytest tests/test_kmer_bytes_fallback.py -v
```

---

## Contributing

We welcome contributions! Please follow these guidelines:

### Code Standards

- **Style**: Follow PEP 8 (use `black` for formatting)
- **Testing**: Add tests for new features
- **Documentation**: Update README and docstrings
- **Type Hints**: Use type annotations where possible

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run tests: `pytest tests/ -v`
5. Format code: `black src/ tests/`
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## Citation

If you use DNATok in your research, please cite:

```bibtex
@software{dnatokens2024,
  title = {DNATok: High-Performance GPU-Accelerated DNA Tokenization},
  author = {[Your Name/Team]},
  year = {2024},
  url = {https://github.com/yourusername/DNATok},
  note = {Software for accelerated DNA sequence tokenization in genomic foundation models}
}
```

### Related Work

DNATok builds upon and supports research in genomic foundation models:

- **Nucleotide Transformer**: Dalla-Torre et al. (2023) - [Paper](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1)
- **HyenaDNA**: Nguyen et al. (2023) - [Paper](https://arxiv.org/abs/2306.15794)
- **Evo**: Nguyen et al. (2024) - [Paper](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v1)

---

## Troubleshooting

### Common Issues

#### "Sequence length not divisible by k"
K-mer tokenizers require sequence length divisible by k. Pad or truncate sequences:
```python
k = dna_tok.kmer_k
seq_len = (len(seq) // k) * k
seq = seq[:seq_len]
```

#### "All sequences must have equal length"
DNATok requires batch sequences to be equal length. Pre-pad to max length:
```python
max_len = max(len(s) for s in seqs)
seqs = [s + 'N' * (max_len - len(s)) for s in seqs]
```

#### CUDA Out of Memory
Reduce `emb_batch` size:
```python
# Instead of emb_batch=64
embeddings = dna_tok.embed_from_strings(seqs, emb_batch=16, device="cuda")
```

---

## Roadmap

- [ ] PyPI package distribution
- [ ] Support for BPE tokenizers
- [ ] Multi-GPU scaling
- [ ] INT8/FP16 embedding optimization
- [ ] Rust/C++ backend for CPU path
- [ ] Support for variable-length batches

---



---

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/DNATok/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/DNATok/discussions)
- **Email**: eli.niktab@anu.edu
