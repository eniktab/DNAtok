# Contributing to DNATok

Thank you for your interest in contributing to DNATok! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in your interactions.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, PyTorch version, GPU model)
- **Minimal code example** that reproduces the issue

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description** of the enhancement
- **Use case** explaining why this would be useful
- **Proposed solution** if you have one in mind
- **Alternatives considered**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Run the test suite**:
   ```bash
   pytest tests/ -v
   ```
6. **Format your code**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```
7. **Update documentation** if needed (README, docstrings)
8. **Commit with clear messages** describing what and why
9. **Push to your fork** and submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/DNATok.git
cd DNATok

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install optional dependencies for examples
pip install -e ".[examples]"
```

## Coding Standards

### Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use `black` for code formatting (line length: 100)
- Use `flake8` for linting
- Maximum line length: 100 characters
- Use descriptive variable names

### Type Hints

- Add type hints to all function signatures
- Use `typing` module for complex types
- Example:
  ```python
  from typing import List, Optional, Iterator
  import torch
  
  def process_sequences(seqs: List[str], batch_size: int = 32) -> Iterator[torch.Tensor]:
      ...
  ```

### Documentation

- Write docstrings for all public functions and classes
- Use Google-style docstrings
- Include examples in docstrings when helpful
- Example:
  ```python
  def encode_batch(self, sequences: List[str]) -> torch.Tensor:
      """Encode a batch of DNA sequences to token IDs.
      
      Args:
          sequences: List of equal-length DNA sequences (ACGTN alphabet).
          
      Returns:
          Token ID tensor of shape [B, T] where B is batch size and T is
          sequence length.
          
      Raises:
          ValueError: If sequences have different lengths.
          
      Example:
          >>> seqs = ["ACGT", "TGCA"]
          >>> ids = tokenizer.encode_batch(seqs)
          >>> ids.shape
          torch.Size([2, 4])
      """
  ```

### Testing

- Write tests for all new features
- Aim for >80% code coverage
- Use `pytest` for testing
- Name test files: `test_*.py`
- Name test functions: `test_*`
- Group related tests in classes
- Example:
  ```python
  def test_encode_batch_returns_correct_shape():
      """Test that encode_batch returns tensor with correct shape."""
      tokenizer = DNATok(embedder)
      seqs = ["ACGT"] * 10
      ids = tokenizer.encode_batch(seqs)
      assert ids.shape == (10, 4)
  ```

## Project Structure

```
DNATok/
├── src/
│   └── dna_tokenizer.py       # Core implementation
├── tests/
│   ├── test_*.py              # Unit tests
│   └── benchmark_*.py         # Performance benchmarks
├── benchmarks/
│   └── *.py                   # Benchmark scripts
├── examples/
│   └── *.py                   # Usage examples
├── README.md                  # Main documentation
├── LICENSE                    # Apache 2.0 license
├── setup.py                   # Package configuration
└── requirements.txt           # Dependencies
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_benchmark_correctness.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

1. **Correctness Tests**: Verify output matches baseline
2. **Fallback Tests**: Ensure graceful degradation
3. **Edge Case Tests**: Handle unusual inputs
4. **Performance Tests**: Benchmark regressions

## Benchmark Guidelines

When adding benchmarks:

1. **Use realistic data**: Real genomic sequences when possible
2. **Measure end-to-end**: Include H2D transfer, not just compute
3. **Report statistics**: Mean, std dev, min, max
4. **Compare to baseline**: Always include baseline comparison
5. **Document setup**: Note GPU model, batch size, sequence length

## Documentation Updates

When updating documentation:

- Keep README concise and scannable
- Use clear headings and subheadings
- Include code examples
- Add badges for important metrics
- Update table of contents if adding sections

## Release Process

(For maintainers)

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release branch: `git checkout -b release/v0.x.0`
4. Run full test suite: `pytest tests/`
5. Run benchmarks: `python tests/benchmark_real_models.py`
6. Update documentation
7. Merge to main and tag: `git tag v0.x.0`
8. Push tags: `git push --tags`
9. Create GitHub release with notes

## Questions?

- **Issues**: [GitHub Issues](https://github.com/yourusername/DNATok/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/DNATok/discussions)
- **Email**: your.email@institution.edu

Thank you for contributing to DNATok! 🧬
