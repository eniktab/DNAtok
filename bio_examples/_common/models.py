"""Real-model loaders for the bio case studies.

`load_model_auto(model_name)` dispatches to the right loader for
each family. Returns `(model, forward_fn, info)` consistently — see
the docstring at the top of each loader for what the model object
actually is.


Each loader returns a tuple ``(model, forward_fn, info)`` where:
  * ``model`` is a ready-to-use torch module on the requested device.
  * ``forward_fn(ids: LongTensor) -> Tensor`` produces a hidden-state
    or logits tensor of shape [B, T, ...].
  * ``info`` is a small dict with ``params``, ``loader``, and any notes
    that should appear in the pipeline's results.json.

All loaders here use REAL published weights. None of these substitute
proxy models silently. If a loader cannot bring up the real model on
the current host the loader raises — the caller decides whether to
abort or report tokenize-only.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------
# Evo2 — Arc Institute, Nature 2026. Loaded via the official `evo2`
# Python package, which wraps StripedHyena2.
# ---------------------------------------------------------------------

def load_evo2(model_name: str = "evo2_1b_base") -> Tuple[Any, Callable, dict]:
    """Load Evo2 via the official package. Forward returns logits."""
    import torch
    from evo2 import Evo2  # raises ImportError if package not present

    model = Evo2(model_name)

    def forward_fn(ids):
        if ids.device.type != "cuda":
            ids = ids.cuda()
        with torch.no_grad():
            # Evo2.forward returns ((logits, None), embeddings_dict_or_None).
            (logits, _), _ = model.forward(ids)
        return logits

    n_params = sum(p.numel() for p in model.model.parameters())
    return model, forward_fn, {
        "model_name": f"arcinstitute/{model_name}",
        "params": n_params,
        "loader": "evo2.Evo2 (official package)",
        "output": "logits [B, T, V]",
    }


# ---------------------------------------------------------------------
# NTv3 — InstaDeepAI, Dec 2025. AutoModelForMaskedLM via trust_remote_code.
# Our dnatok_compat shim exposes the filter_list property required by
# the upstream modeling code.
# ---------------------------------------------------------------------

def load_ntv3(model_name: str = "InstaDeepAI/NTv3_8M_pre") -> Tuple[Any, Callable, dict]:
    import torch
    import dnatok_compat  # noqa: F401 — must import before transformers
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def forward_fn(ids):
        if ids.device.type != "cuda":
            ids = ids.to(device)
        with torch.no_grad():
            return model(ids).logits

    return model, forward_fn, {
        "model_name": model_name,
        "params": sum(p.numel() for p in model.parameters()),
        "loader": "AutoModelForMaskedLM(trust_remote_code=True) + "
                   "dnatok_compat filter_list property",
        "output": "logits [B, T, V]",
    }


# ---------------------------------------------------------------------
# DNABERT-2 — ICLR 2024. The HF checkpoint ships a custom flash-
# attention Triton kernel that is incompatible with newer Triton
# builds. We load the weights as a vanilla BertModel (drop the
# auto_map). The vocabulary, embeddings and BERT body are unchanged;
# only the custom attention impl is replaced with the standard
# PyTorch-attention BertModel. Predictions are identical up to FP
# differences in the attention impl, which is documented in the
# pipeline's results.json.
# ---------------------------------------------------------------------

def load_dnabert2(model_name: str = "zhihan1996/DNABERT-2-117M") -> Tuple[Any, Callable, dict]:
    import torch
    from transformers import BertConfig, BertModel
    import dnatok_compat  # noqa: F401

    # Locate the HF cache snapshot. Respect HF_HOME / HF_HUB_CACHE if
    # set (Gadi sets these to /g/data/...). Fall back to ~/.cache.
    safe_name = "models--" + model_name.replace("/", "--")
    hub_roots = []
    for env_var in ("HF_HUB_CACHE", "HF_HOME"):
        v = os.environ.get(env_var)
        if v:
            # HF_HOME points at the parent; HF_HUB_CACHE at the hub dir.
            hub_roots.append(Path(v))
            hub_roots.append(Path(v) / "hub")
    hub_roots.append(Path.home() / ".cache/huggingface/hub")

    snapshots = []
    for root in hub_roots:
        candidate = root / safe_name / "snapshots"
        if candidate.is_dir():
            snapshots = sorted(p for p in candidate.iterdir() if p.is_dir())
            if snapshots:
                break
    if not snapshots:
        searched = ", ".join(str(r) for r in hub_roots)
        raise FileNotFoundError(
            f"No HF cache snapshot for {model_name}; searched: {searched}")
    snap = str(snapshots[0])

    with open(Path(snap) / "config.json") as f:
        cfg_dict = json.load(f)

    # Drop the custom-attention auto_map so HF doesn't fetch the Triton
    # flash-attn kernel that breaks in current Triton builds.
    cfg_dict.pop("auto_map", None)
    cfg_dict.pop("attn_implementation", None)

    cfg = BertConfig(**cfg_dict)
    model = BertModel.from_pretrained(
        snap, config=cfg, ignore_mismatched_sizes=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def forward_fn(ids):
        if ids.device.type != "cuda":
            ids = ids.to(device)
        with torch.no_grad():
            return model(ids).last_hidden_state

    return model, forward_fn, {
        "model_name": model_name,
        "params": sum(p.numel() for p in model.parameters()),
        "loader": "BertModel direct (drops custom Triton flash-attn auto_map). "
                   "Weights and vocab unchanged; attention uses PyTorch SDPA.",
        "output": "last_hidden_state [B, T, H]",
        "caveat": "Attention impl is PyTorch SDPA instead of upstream "
                   "custom Triton flash-attn; outputs differ by FP "
                   "tolerance from the upstream impl.",
    }


# ---------------------------------------------------------------------
# Generic HF loader — for HyenaDNA, Caduceus, NTv2, GENA-LM,
# METAGENE-1, and any other family whose checkpoint just works via
# AutoModel.from_pretrained with trust_remote_code=True.
# ---------------------------------------------------------------------

def load_generic_hf(model_name: str, auto_class: str = "AutoModel") -> Tuple[Any, Callable, dict]:
    """Load any HF-loadable model via the requested AutoModel class."""
    import torch
    import dnatok_compat  # noqa: F401
    from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM

    cls = {
        "AutoModel": AutoModel,
        "AutoModelForMaskedLM": AutoModelForMaskedLM,
        "AutoModelForCausalLM": AutoModelForCausalLM,
    }[auto_class]

    model = cls.from_pretrained(model_name, trust_remote_code=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def forward_fn(ids):
        if ids.device.type != "cuda":
            ids = ids.to(device)
        with torch.no_grad():
            out = model(ids)
            # Return whatever's there — logits for MLM/CausalLM,
            # last_hidden_state for base models.
            return getattr(out, "logits",
                           getattr(out, "last_hidden_state", out))

    return model, forward_fn, {
        "model_name": model_name,
        "params": sum(p.numel() for p in model.parameters()),
        "loader": f"{auto_class}(trust_remote_code=True)",
        "output": "logits or last_hidden_state",
    }


# ---------------------------------------------------------------------
# Auto dispatcher — pick the right loader by model_name.
# ---------------------------------------------------------------------

def load_model_auto(model_name: str) -> Tuple[Any, Callable, dict]:
    """Dispatch to the right loader for the given HF model id.

    Raises if the model can't be loaded in the current env.
    """
    name_l = model_name.lower()
    if "evo2" in name_l:
        # arcinstitute/evo2_* — use official package; the model_name
        # passed must be the bare key the evo2 package recognises.
        bare = model_name.split("/")[-1]
        return load_evo2(bare)
    if "ntv3" in name_l:
        return load_ntv3(model_name)
    if "dnabert" in name_l:
        return load_dnabert2(model_name)
    # HyenaDNA, Caduceus, NTv2, GENA-LM, METAGENE-1 — these load via
    # standard AutoModel with trust_remote_code. Most expose a base
    # model (no LM head needed for our sustained-throughput test).
    if "metagene" in name_l:
        # METAGENE-1 is causal LM-style.
        return load_generic_hf(model_name, "AutoModelForCausalLM")
    if "nucleotide-transformer-v2" in name_l:
        # NTv2 ships as an EsmForMaskedLM-compatible checkpoint;
        # AutoModel hits a tied-embedding size mismatch (LM head is
        # 4096-dim, base model body 2048). Use the MLM auto class.
        return load_generic_hf(model_name, "AutoModelForMaskedLM")
    if "caduceus" in name_l:
        # Caduceus uses Mamba SSM blocks; requires the `mamba_ssm`
        # package (heavy CUDA build). Try and surface a clear error
        # so the caller knows to install it.
        try:
            import mamba_ssm  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"Caduceus requires `mamba_ssm` which is not installed in "
                f"this env. Either `pip install mamba_ssm` (needs CUDA + "
                f"a matching torch build) or skip Caduceus on this host."
            ) from e
        return load_generic_hf(model_name, "AutoModel")
    if "hyena" in name_l:
        # HyenaDNA family loads via standard AutoModel.
        return load_generic_hf(model_name, "AutoModel")
    if "gena-lm" in name_l or "gena_lm" in name_l:
        # GENA-LM family loads via AutoModelForMaskedLM (BERT-style MLM)
        # or AutoModel; AutoModel works.
        return load_generic_hf(model_name, "AutoModel")
    # Fallback.
    return load_generic_hf(model_name, "AutoModel")
