#!/usr/bin/env python3
"""
Minimal Evo2 + DNATok walkthrough.
Run top-to-bottom: load Evo2, wrap it, and compare DNATok outputs.
Defaults are baked in; tweak the constants below if you want bigger batches/lengths.
"""

import pathlib
import sys
import time

import numpy as np
import torch

try:
    from evo2 import Evo2
except ImportError as exc:
    raise SystemExit("Install the Evo2 package to run this demo.") from exc

# add DNATok to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from dna_tokenizer import DNATok  # noqa: E402


# --- Demo settings (tweak as needed) ---
MODEL_NAME = "evo2_7b"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH = 32
SEQ_LEN = 128
EMB_BATCH = 16


def make_random_seqs(b, T, seed=0):
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"))
    return ["".join(rng.choice(alphabet, size=T)) for _ in range(b)]


class Evo2Adapter(torch.nn.Module):
    """Expose embed_tokens + tokenizer for DNATok without needing layer_names."""

    def __init__(self, evo2_model: Evo2, layer_name: str | None = None):
        super().__init__()
        self.model = evo2_model
        self.tokenizer = evo2_model.tokenizer
        self.pad_token_id = getattr(self.tokenizer, "pad_id", 0)
        inner = getattr(evo2_model, "model", None)
        self.max_position_embeddings = getattr(inner, "max_position_embeddings", None)
        self._embed = self._resolve_embed_fn(evo2_model)
        self._layer_name = layer_name or self._infer_layer_name(inner or evo2_model)

    def _resolve_embed_fn(self, evo2_model):
        for obj in (evo2_model, getattr(evo2_model, "model", None)):
            if obj is None:
                continue
            for attr in ("embed_tokens", "tok_embeddings"):
                fn = getattr(obj, attr, None)
                if callable(fn):
                    return fn
            getter = getattr(obj, "get_input_embeddings", None)
            if callable(getter):
                emb = getter()
                if callable(emb):
                    return emb
            try:
                for name, module in obj.named_modules():
                    if "embed" in name and "token" in name and callable(module):
                        return module
            except Exception:
                pass
        return None

    def _infer_layer_name(self, obj):
        try:
            keys = list(obj.state_dict().keys())
        except Exception:
            return None
        for k in keys:
            if "embed" in k and k.endswith("weight"):
                return k.rsplit(".weight", 1)[0]
        return None

    def embed_tokens(self, input_ids, rc_invariant=False):
        if self._embed is not None:
            return self._embed(input_ids)
        if not self._layer_name:
            raise RuntimeError("Could not find Evo2 embedding layer; specify layer_name.")
        _, embeds = self.model(
            input_ids,
            return_embeddings=True,
            layer_names=[self._layer_name],
        )
        out = embeds[self._layer_name]
        return out.to(input_ids.device) if out.device != input_ids.device else out


def run_demo():
    torch.set_grad_enabled(False)
    print(f"Generating {BATCH} random DNA strings of length {SEQ_LEN}...")
    seqs = make_random_seqs(BATCH, SEQ_LEN)

    print(f"Loading Evo2 model '{MODEL_NAME}' on {DEVICE} (expects local weights).")
    evo2_model = Evo2(MODEL_NAME)
    adapter = Evo2Adapter(evo2_model).to(DEVICE)

    print("Building DNATok helper and LUT...")
    tok_helper = DNATok(adapter)
    tok_helper.discover()
    tok_helper.token_len = SEQ_LEN  # keep padding aligned with the synthetic inputs

    print("Checking DNATok IDs match the Evo2 tokenizer...")
    tok_ids = torch.tensor([adapter.tokenizer.tokenize(s) for s in seqs], dtype=torch.long)
    lut_ids = tok_helper.encode_batch_to_ids_staging(seqs, dtype=torch.long).cpu()
    if lut_ids.shape[1] > tok_ids.shape[1]:
        lut_ids = lut_ids[:, -tok_ids.shape[1]:]  # drop any left pad area
    mismatch = (tok_ids != lut_ids).nonzero(as_tuple=False)
    if mismatch.numel():
        i, j = mismatch[0].tolist()
        raise RuntimeError(
            f"Token mismatch at seq {i}, pos {j}: tokenizer={tok_ids[i, j]}, lut={lut_ids[i, j]}, base={seqs[i][j]}"
        )
    print("IDs match")

    print("\nBaseline: tokenize + embed via Evo2 tokenizer")
    t0 = time.perf_counter()
    hf_ids = tok_ids.to(DEVICE)
    t1 = time.perf_counter()
    hf_emb = adapter.embed_tokens(hf_ids).float()
    if hf_emb.device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f"Tokenize time: {(t1 - t0) * 1e3:.1f} ms")
    print(f"Embed time:    {(t2 - t1) * 1e3:.1f} ms")

    print("\nDNATok fast path (bytes on CUDA if available)")
    t3 = time.perf_counter()
    chunks = list(tok_helper.embed_from_strings(seqs, emb_batch=EMB_BATCH, device=DEVICE, path="auto"))
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
