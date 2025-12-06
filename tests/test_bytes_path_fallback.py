from __future__ import annotations

import torch

from src.dna_tokenizer import DNATok


class SimpleTokenizer:
    """Char-level tokenizer: 1 token per char, left-pad aware."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.padding_side = "left"

    def encode(self, text, add_special_tokens: bool = False):
        return [ord(ch) % 97 + 1 for ch in text]  # deterministic, >0

    def __call__(
        self,
        texts,
        add_special_tokens: bool = False,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        target_len = max(len(seq) for seq in encoded)
        if padding is True or padding == "max_length":
            if padding == "max_length" and max_length is not None:
                target_len = max_length
            pad = self.pad_token_id
            encoded = [[pad] * (target_len - len(seq)) + seq for seq in encoded]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(encoded, dtype=torch.long)}
        return {"input_ids": encoded}


class RecordingEmbedder(torch.nn.Module):
    def __init__(self, tokenizer: SimpleTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.max_position_embeddings = None
        self._embed = torch.nn.Embedding(512, 4)
        self.last_ids: torch.Tensor | None = None

    def embed_tokens(self, input_ids, rc_invariant: bool = False):
        self.last_ids = input_ids.detach().clone()
        return self._embed(input_ids)


def test_bytes_path_failure_falls_back_to_ids(monkeypatch):
    tok = SimpleTokenizer()
    embedder = RecordingEmbedder(tok)
    helper = DNATok(embedder)
    helper.discover()

    # Force bytes path to fail to verify graceful fallback to ids path.
    def boom(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(helper, "_map_ascii_bytes_to_ids_cuda", boom)

    seqs = ["ACGT", "TGCA"]
    outputs = list(helper.embed_from_strings(seqs, emb_batch=4, device="cpu", path="bytes"))

    expected_ids = helper.encode_batch_to_ids_staging(seqs).to(torch.long)

    assert outputs and outputs[0].shape[0] == len(seqs)
    assert embedder.last_ids is not None
    assert torch.equal(embedder.last_ids.cpu(), expected_ids)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
