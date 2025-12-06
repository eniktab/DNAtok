#!/usr/bin/env python3
"""
Hyena DNA + DNATok in a few readable steps.
No CLI flags—tweak the constants below like a notebook cell, then run the file.
"""

import os
import pathlib
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from dna_tokenizer import DNATok  # noqa: E402

# --- Demo settings (edit as needed) ---
MODEL_NAME = "hyenadna-small-32k-seqlen-hf"
MODEL_PATH = pathlib.Path(
    os.environ.get("HYENA_MODEL_PATH", ROOT / "hf-cache" / "models" / MODEL_NAME)
).expanduser()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH = 16
SEQ_LEN = 256
EMB_BATCH = 8


def make_random_seqs(b: int, T: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"))
    return ["".join(rng.choice(alphabet, size=T)) for _ in range(b)]


class HyenaAdapter(torch.nn.Module):
    """Expose embed_tokens + tokenizer for DNATok."""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "pad_id", 0)
        inner = getattr(model, "model", model)
        self.max_position_embeddings = getattr(
            getattr(model, "config", None), "max_position_embeddings", None
        ) or getattr(inner, "max_position_embeddings", None)
        self._embed = self._resolve_embed(inner)

    def _resolve_embed(self, inner):
        for attr in ("embed_tokens", "tok_embeddings"):
            fn = getattr(inner, attr, None)
            if callable(fn):
                return fn
        getter = getattr(inner, "get_input_embeddings", None)
        if getter is None:
            getter = getattr(self.model, "get_input_embeddings", None)
        if callable(getter):
            emb = getter()
            if callable(emb):
                return emb
        return None

    def embed_tokens(self, input_ids, rc_invariant=False):
        if self._embed is None:
            raise RuntimeError("HyenaAdapter could not find input embedding layer.")
        return self._embed(input_ids)


def run_demo() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    torch.set_grad_enabled(False)

    print(f"Generating {BATCH} random DNA strings of length {SEQ_LEN}...")
    seqs = make_random_seqs(BATCH, SEQ_LEN)

    model_path = MODEL_PATH.resolve()
    if not model_path.exists():
        raise SystemExit(
            f"Set HYENA_MODEL_PATH to a downloaded checkpoint folder (expected subfolder like '{MODEL_NAME}')."
        )
    print(f"Using Hyena snapshot at: {model_path}")

    print(f"Loading tokenizer + model on {DEVICE} (local files only)...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float16 if torch.cuda.is_available() else None,
    ).to(DEVICE)

    adapter = HyenaAdapter(model, tokenizer).to(DEVICE)
    tok_helper = DNATok(adapter)
    tok_helper.force_fp32_outputs = False  # keep model dtype
    tok_helper.ids_max_tokens_per_call = BATCH * SEQ_LEN
    tok_helper.discover()
    tok_helper.token_len = SEQ_LEN

    print("Verifying DNATok LUT matches Hyena tokenizer...")
    tok_ids_list = [adapter.tokenizer.encode(s, add_special_tokens=False) for s in seqs]
    tok_ids = torch.tensor(tok_ids_list, dtype=torch.long)
    lut_ids = tok_helper.encode_batch_to_ids_staging(seqs, dtype=torch.long).cpu()
    if lut_ids.shape[1] > tok_ids.shape[1]:
        lut_ids = lut_ids[:, -tok_ids.shape[1]:]
    mismatch = (tok_ids != lut_ids).nonzero(as_tuple=False)
    if mismatch.numel():
        i, j = mismatch[0].tolist()
        raise RuntimeError(
            f"Token mismatch at seq {i}, pos {j}: tokenizer={tok_ids[i, j]}, lut={lut_ids[i, j]}, base={seqs[i][j]}"
        )
    print("IDs match")

    print("\nBaseline: tokenize + embed via Hyena tokenizer")
    t0 = time.perf_counter()
    hf_ids = tok_ids.to(DEVICE)
    t1 = time.perf_counter()
    hf_emb = adapter.embed_tokens(hf_ids)
    if hf_emb.device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f"Tokenize time: {(t1 - t0) * 1e3:.1f} ms")
    print(f"Embed time:    {(t2 - t1) * 1e3:.1f} ms")

    print("\nDNATok fast path (bytes on CUDA if available)")
    t3 = time.perf_counter()
    chunks = list(
        tok_helper.embed_from_strings(seqs, emb_batch=EMB_BATCH, device=DEVICE, path="bytes")
    )
    fast_emb = torch.cat(chunks, dim=0)
    if fast_emb.device.type == "cuda":
        torch.cuda.synchronize()
    t4 = time.perf_counter()
    print(f"DNATok time: {(t4 - t3) * 1e3:.1f} ms")

    diff = (fast_emb - hf_emb).abs().max().item()
    print(f"\nMax |embedding diff|: {diff:.3e}")
    print(f"Speedup vs baseline: {(t2 - t0) / (t4 - t3):.2f}x")


if __name__ == "__main__":
    run_demo()
