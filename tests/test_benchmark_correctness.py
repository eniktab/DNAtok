"""
test_benchmark_correctness.py

Verify that benchmarks are testing the correct thing:
- DNATok output matches tokenizer output (correctness)
- Both path="ids" and path="bytes" produce identical results
- Edge cases are handled correctly
"""

from __future__ import annotations

import random
from typing import List

import pytest
import torch
import numpy as np

from src.dna_tokenizer import DNATok


class SimpleCharTokenizer:
    """Deterministic char-level tokenizer for testing."""
    
    def __init__(self):
        self.pad_token_id = 0
        self.padding_side = "left"
        self.vocab = {"<pad>": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
        # Lowercase variants
        for k, v in list(self.vocab.items()):
            if k.isupper() and len(k) == 1:
                self.vocab[k.lower()] = v
    
    def encode(self, text: str, add_special_tokens=False):
        return [self.vocab.get(ch, self.vocab["N"]) for ch in text]
    
    def __call__(self, texts, add_special_tokens=False, padding=False, 
                 truncation=False, max_length=None, return_tensors=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = [self.encode(t, add_special_tokens) for t in texts]
        
        if padding:
            target_len = max(len(seq) for seq in encoded)
            if padding == "max_length" and max_length:
                target_len = max_length
            
            pad = self.pad_token_id
            if self.padding_side == "left":
                encoded = [[pad] * (target_len - len(seq)) + seq for seq in encoded]
            else:
                encoded = [seq + [pad] * (target_len - len(seq)) for seq in encoded]
        
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(encoded, dtype=torch.long)}
        return {"input_ids": encoded}


class RecordingEmbedder(torch.nn.Module):
    """Mock embedder that records what IDs it received."""
    
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.max_position_embeddings = None
        self._embed = torch.nn.Embedding(512, 4)
        self.last_ids: torch.Tensor | None = None
    
    def embed_tokens(self, input_ids, rc_invariant=False):
        self.last_ids = input_ids.detach().clone()
        return self._embed(input_ids)


def make_random_seqs(B: int, T: int, alphabet="ACGTN", seed: int = 0) -> List[str]:
    random.seed(seed)
    return ["".join(random.choice(alphabet) for _ in range(T)) for _ in range(B)]


def test_dnatok_matches_baseline_tokenizer():
    """Verify DNATok produces same IDs as the underlying tokenizer."""
    tok = SimpleCharTokenizer()
    embedder = RecordingEmbedder(tok)
    helper = DNATok(embedder)
    helper.discover()
    
    assert helper.use_ids_path, "IDs path should be enabled"
    
    # Test sequences - must be equal length for DNATok
    seqs = [
        "ACGTACGT",
        "NNNNNNNN",
        "TGCATGCA",
        "ACGTACGT",
    ]
    
    # Get baseline tokenizer output
    baseline_out = tok(seqs, padding=True, return_tensors="pt")
    baseline_ids = baseline_out["input_ids"]
    
    # Get DNATok output
    dnatok_ids = helper.encode_batch_to_ids(seqs)
    
    # Should match exactly
    assert torch.equal(baseline_ids, dnatok_ids), (
        f"DNATok output doesn't match tokenizer!\n"
        f"Baseline: {baseline_ids}\n"
        f"DNATok:   {dnatok_ids}"
    )


def test_ids_path_vs_bytes_path_consistency():
    """Verify path='ids' and path='bytes' produce identical embeddings."""
    tok = SimpleCharTokenizer()
    embedder = RecordingEmbedder(tok)
    helper = DNATok(embedder)
    helper.discover()
    
    seqs = make_random_seqs(4, 16)
    
    # Path: IDs
    helper.last_ids = None
    outputs_ids = list(helper.embed_from_strings(seqs, emb_batch=4, device="cpu", path="ids"))
    ids_from_ids_path = embedder.last_ids.clone() if embedder.last_ids is not None else None
    
    # Path: Bytes
    embedder.last_ids = None
    outputs_bytes = list(helper.embed_from_strings(seqs, emb_batch=4, device="cpu", path="bytes"))
    ids_from_bytes_path = embedder.last_ids.clone() if embedder.last_ids is not None else None
    
    # Both paths should produce the same IDs
    assert ids_from_ids_path is not None and ids_from_bytes_path is not None
    assert torch.equal(ids_from_ids_path, ids_from_bytes_path), (
        f"IDs path and bytes path produced different token IDs!\n"
        f"IDs path:   {ids_from_ids_path}\n"
        f"Bytes path: {ids_from_bytes_path}"
    )
    
    # Outputs should also match
    assert len(outputs_ids) == len(outputs_bytes)
    for i, (out_ids, out_bytes) in enumerate(zip(outputs_ids, outputs_bytes)):
        assert torch.equal(out_ids, out_bytes), f"Embedding output mismatch at batch {i}"


def test_mixed_case_handling():
    """Test that mixed-case sequences are handled correctly with normalize_case."""
    tok = SimpleCharTokenizer()
    embedder = RecordingEmbedder(tok)
    
    # Test WITH normalization
    helper_norm = DNATok(embedder, normalize_case=True)
    helper_norm.discover()
    
    seqs = ["AcGt", "acgt", "ACGT"]
    ids_norm = helper_norm.encode_batch_to_ids(seqs)
    
    # All three should produce the same IDs when normalized
    assert torch.equal(ids_norm[0], ids_norm[1]), "AcGt != acgt with normalization"
    assert torch.equal(ids_norm[0], ids_norm[2]), "AcGt != ACGT with normalization"
    
    # Test WITHOUT normalization (default)
    helper_no_norm = DNATok(embedder, normalize_case=False)
    helper_no_norm.discover()
    
    ids_no_norm = helper_no_norm.encode_batch_to_ids(seqs)
    
    # With SimpleCharTokenizer, upper and lower map to same IDs, so should still match
    # But this tests the code path at least
    assert ids_no_norm.shape == (3, 4)


def test_invalid_char_handling():
    """Test that invalid characters are mapped to N when handle_invalid_chars=True."""
    tok = SimpleCharTokenizer()
    embedder = RecordingEmbedder(tok)
    
    # With invalid char handling
    helper_handle = DNATok(embedder, handle_invalid_chars=True)
    helper_handle.discover()
    
    seqs_with_invalid = ["ACXGT", "AC$GT"]
    ids_handled = helper_handle.encode_batch_to_ids(seqs_with_invalid)
    
    # Get expected output (X and $ should map to N)
    seqs_expected = ["ACNGT", "ACNGT"]
    ids_expected = helper_handle.encode_batch_to_ids(seqs_expected)
    
    assert torch.equal(ids_handled, ids_expected), (
        f"Invalid char handling failed!\n"
        f"Got:      {ids_handled}\n"
        f"Expected: {ids_expected}"
    )
    
    # Without invalid char handling (should work for SimpleCharTokenizer since it maps to N)
    helper_no_handle = DNATok(embedder, handle_invalid_chars=False)
    helper_no_handle.discover()
    # This should still work because SimpleCharTokenizer maps unknown chars to N
    ids_no_handle = helper_no_handle.encode_batch_to_ids(seqs_with_invalid)
    assert ids_no_handle.shape == ids_handled.shape


def test_edge_case_all_same_base():
    """Test sequences with all the same base."""
    tok = SimpleCharTokenizer()
    embedder = RecordingEmbedder(tok)
    helper = DNATok(embedder)
    helper.discover()
    
    for base in "ACGTN":
        seq = base * 16
        ids = helper.encode_batch_to_ids([seq])
        
        # All IDs should be the same
        expected_id = tok.vocab[base]
        assert torch.all(ids == expected_id), f"All-{base} sequence failed"


def test_empty_batch():
    """Test that empty batch raises appropriate error."""
    tok = SimpleCharTokenizer()
    embedder = RecordingEmbedder(tok)
    helper = DNATok(embedder)
    helper.discover()
    
    with pytest.raises(ValueError, match="No sequences"):
        helper.encode_batch_to_ids([])


def test_variable_length_raises_error():
    """Test that variable-length sequences raise an error."""
    tok = SimpleCharTokenizer()
    embedder = RecordingEmbedder(tok)
    helper = DNATok(embedder)
    helper.discover()
    
    seqs = ["ACGT", "ACGTACGT"]  # Different lengths
    with pytest.raises(ValueError, match="equal length"):
        helper.encode_batch_to_ids(seqs)


def test_large_batch_correctness():
    """Test correctness with a larger batch to catch any batching bugs."""
    tok = SimpleCharTokenizer()
    embedder = RecordingEmbedder(tok)
    helper = DNATok(embedder)
    helper.discover()
    
    B, T = 100, 64
    seqs = make_random_seqs(B, T, seed=42)
    
    # Compare DNATok vs baseline
    baseline_ids = tok(seqs, padding=True, return_tensors="pt")["input_ids"]
    dnatok_ids = helper.encode_batch_to_ids(seqs)
    
    assert torch.equal(baseline_ids, dnatok_ids), "Large batch correctness check failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
