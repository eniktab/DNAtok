from __future__ import annotations

import torch

from src.dna_tokenizer import DNATok


class FakeKmerTokenizer:
    """Minimal k-mer tokenizer stub: single-token A/C/G/T kmers, multi-token for 'N'."""

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.pad_token_id = 0
        self.padding_side = "right"
        self._ids: dict[str, int] = {}

    def __len__(self) -> int:
        return 256

    def _encode_one(self, text: str) -> list[int]:
        ids: list[int] = []
        for i in range(0, len(text), self.k):
            chunk = text[i : i + self.k]
            if len(chunk) < self.k:
                continue
            if "N" in chunk:
                # Unsupported kmers emit multiple tokens, so DNATok's LUT entry stays -1.
                ids.extend([97, 98])
            else:
                ids.append(self._ids.setdefault(chunk, len(self._ids) + 1))
        return ids or [self.pad_token_id]

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
        encoded = [self._encode_one(t) for t in texts]

        target_len = max(len(seq) for seq in encoded)
        if padding == "max_length" and max_length:
            target_len = max_length
        if padding is True or padding == "max_length":
            pad = self.pad_token_id
            if self.padding_side == "left":
                encoded = [[pad] * (target_len - len(seq)) + seq for seq in encoded]
            else:
                encoded = [seq + [pad] * (target_len - len(seq)) for seq in encoded]

        if return_tensors == "pt":
            return {"input_ids": torch.tensor(encoded, dtype=torch.long)}
        return {"input_ids": encoded}


class RecordingEmbedder(torch.nn.Module):
    def __init__(self, tokenizer: FakeKmerTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.max_position_embeddings = None
        self.last_ids: torch.Tensor | None = None
        self._embed = torch.nn.Embedding(256, 4)

    def embed_tokens(self, input_ids, rc_invariant: bool = False):
        self.last_ids = input_ids.detach().clone()
        return self._embed(input_ids)


def test_bytes_path_with_unsupported_kmer_falls_back_to_tokenizer_path():
    tok = FakeKmerTokenizer(k=3)
    embedder = RecordingEmbedder(tok)
    helper = DNATok(embedder)
    helper.discover()

    assert helper.kmer_k == 3
    seqs = ["AAANNN"]

    baseline_ids = helper._tokenize_batch_cpu(seqs, dtype=torch.long, pin=False)
    outputs = list(helper.embed_from_strings(seqs, emb_batch=2, device="cpu", path="bytes"))

    assert embedder.last_ids is not None
    assert torch.equal(embedder.last_ids.cpu(), baseline_ids)
    assert outputs and outputs[0].shape[0] == len(seqs)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
