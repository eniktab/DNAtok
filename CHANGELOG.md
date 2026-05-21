# Changelog

All notable changes to DNATok will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README with installation, usage, and benchmarks
- Setup.py for pip installation
- Contributing guidelines (CONTRIBUTING.md)
- Requirements.txt for dependency management
- Apache 2.0 License

## [0.1.0] - 2024-12-06

### Added
- Initial release of DNATok
- GPU-accelerated DNA sequence tokenization
- Automatic k-mer structure detection (k=3-6)
- Runtime ASCII/k-mer LUT discovery
- Multiple encoding paths: bytes, k-mer, IDs, fallback
- Pipelined H2D/compute overlap with CUDA streams
- Auto-tuning micro-batch sizes
- Support for Nucleotide Transformer, HyenaDNA, Evo2
- Comprehensive test suite (14 tests)
- Real-world model benchmarks
- Example scripts for NT, Hyena, Evo2

### Performance
- 34-244x speedup over baseline tokenization
- Bytes path: 68-244x speedup (char-level tokenizers)
- K-mer path: 50-150x speedup (k-mer tokenizers)
- IDs path: 34-43x speedup (fallback path)

### Robustness
- Case normalization support (`normalize_case`)
- Invalid character handling (`handle_invalid_chars`)
- Graceful fallback to tokenizer on edge cases
- Automatic LUT rebuild on tokenizer mismatch

[Unreleased]: https://github.com/[org]/DNAtok/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/[org]/DNAtok/releases/tag/v0.1.0
