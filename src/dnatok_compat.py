import os
import sys
import json
from pathlib import Path
import torch
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

# -----------------------------------------------------------------------------
# 0. Early Patching (MUST BE BEFORE TRANSFORMERS IMPORT)
# -----------------------------------------------------------------------------

def apply_importlib_patch():
    """Pre-pin compatibility shim for environments that ship a newer
    huggingface_hub than transformers' version check accepts.

    On Gadi NGC and our pinned venv (`reference.json`), `transformers==4.56.2`
    + `huggingface_hub==0.35.1` work together cleanly and this patch is a
    NO-OP. It is retained as a safety belt for legacy environments where
    `huggingface_hub` may have been bumped past the version transformers
    expects (e.g. >=1.5.0); in that case the patch clamps the reported
    version back to a known-compatible value.

    To remove permanently: delete this function and its call below.

    NOTE: This shim does NOT force HF_HUB_OFFLINE — for the plug-and-play
    Docker / Apptainer flow, users expect first-run downloads to work.
    The benchmark scripts that NEED offline mode set the env var
    themselves.
    """
    try:
        import importlib.metadata as md
        try:
            actual = md.version("huggingface_hub")
        except Exception:
            actual = None
        # Only patch if the installed version is OUTSIDE the known-good
        # range for transformers 4.56.x (anything <1.0 is fine; 1.5+ trips
        # the upstream version check). Inside the pinned range, leave the
        # real version visible.
        def _needs_clamp(v):
            if not isinstance(v, str):
                return False
            parts = v.split(".")
            try:
                return int(parts[0]) >= 1
            except Exception:
                return False
        if _needs_clamp(actual):
            CLAMP = "0.35.1"
            _ov = md.version
            md.version = lambda n: CLAMP if str(n).replace("_", "-").lower() == "huggingface-hub" else _ov(n)

            _od = md.distribution
            def _pd(n):
                if str(n).replace("_", "-").lower() == "huggingface-hub":
                    class MockDist:
                        def __init__(self, d): self._d = d; self.version = CLAMP
                        @property
                        def metadata(self): return self._d.metadata
                        def __getattr__(self, k): return getattr(self._d, k)
                    return MockDist(_od(n))
                return _od(n)
            md.distribution = _pd
    except Exception:
        pass

apply_importlib_patch()

# -----------------------------------------------------------------------------
# 1. Imports (Safe now that version check is bypassed)
# -----------------------------------------------------------------------------

from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast

# -----------------------------------------------------------------------------
# 2. Local NTv3 Configuration & Tokenizer
# -----------------------------------------------------------------------------

class Ntv3PreTrainedConfig(PretrainedConfig):
    model_type = "ntv3"
    def __init__(
        self,
        alphabet_size: int = 11,
        pad_token_id: int = 1,
        mask_token_id: int = 2,
        num_downsamples: int = 7,
        attention_heads: int = 8,
        key_size: int = 32,
        token_embed_dim: int = 16,
        conv_init_embed_dim: int = 256,
        embed_dim: int = 256,
        ffn_embed_dim: int = 1024,
        num_layers: int = 2,
        layer_norm_eps: float = 1e-5,
        num_hidden_layers_head: int = 0,
        use_skip_connection: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.alphabet_size = alphabet_size
        self.mask_token_id = mask_token_id
        self.num_downsamples = num_downsamples
        self.attention_heads = attention_heads
        self.key_size = key_size
        self.token_embed_dim = token_embed_dim
        self.conv_init_embed_dim = conv_init_embed_dim
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.num_layers = num_layers
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers_head = num_hidden_layers_head
        self.use_skip_connection = use_skip_connection

    @property
    def filter_list(self) -> list:
        """Conv/deconv tower filter sizes.

        Mirrors the formula in the upstream NTv3 config
        (InstaDeepAI/ntv3_base_model/configuration_ntv3_pretrained.py):
        ``np.linspace(conv_init_embed_dim, embed_dim,
        num_downsamples + 1).astype(int).tolist()``. Computed as a
        property so the stored attribute count matches HF's
        round-trip serialisation expectations.
        """
        import numpy as np
        return list(
            np.linspace(self.conv_init_embed_dim, self.embed_dim,
                         self.num_downsamples + 1).astype(int)
        )


class _BaseNTv3Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids"]
    def __init__(self, *, vocab_file: Optional[str], unk_token: str, pad_token: str, mask_token: str, **kwargs: Any) -> None:
        if vocab_file and os.path.isfile(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as h:
                loaded = json.load(h)
            self._token_to_id = {str(k): int(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {str(t): i for i, t in enumerate(loaded)}
        else:
            self._token_to_id = {t: i for i, t in enumerate(["<unk>", "<pad>", "<mask>", "<cls>", "<eos>", "<bos>", "A", "T", "C", "G", "N"])}
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        super().__init__(unk_token=unk_token, pad_token=pad_token, mask_token=mask_token, **kwargs)
    def get_vocab(self): return dict(self._token_to_id)
    @property
    def vocab_size(self): return len(self._token_to_id)
    def _convert_token_to_id(self, token): return self._token_to_id.get(token, self._token_to_id.get(self.unk_token, 0))
    def _convert_id_to_token(self, idx): return self._id_to_token.get(idx, self.unk_token)
    def save_vocabulary(self, save_dir, prefix=None):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, (f"{prefix}-" if prefix else "") + "vocab.json")
        with open(path, "w", encoding="utf-8") as h: json.dump(self._token_to_id, h, indent=2)
        return (path,)

class NTv3Tokenizer(_BaseNTv3Tokenizer):
    def __init__(self, vocab_file=None, **kwargs):
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("mask_token", "<mask>")
        super().__init__(vocab_file=vocab_file, **kwargs)
        self._std = {"A", "T", "C", "G", "N"}
    def _tokenize(self, text):
        return [c.upper() if c.upper() in self._std else str(self.unk_token) for c in text]


# -----------------------------------------------------------------------------
# 2b. Evo2 byte-level tokenizer (arcinstitute/evo2_*)
# -----------------------------------------------------------------------------

class StripedHyena2Config(PretrainedConfig):
    """Stub config so AutoConfig recognises arcinstitute Evo2 checkpoints."""
    model_type = "stripedhyena2"
    def __init__(self, vocab_size=512, **kwargs):
        self.vocab_size = vocab_size
        super().__init__(**kwargs)


class Evo2ByteTokenizer(PreTrainedTokenizer):
    """ASCII byte-level tokenizer for Evo2 (vocab = ord(c))."""
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names: Dict[str, str] = {}

    def __init__(self, vocab_size: int = 512, **kwargs):
        self._vocab_size = int(vocab_size)
        self._token_to_id = {chr(i): i for i in range(self._vocab_size)}
        self._id_to_token = {i: chr(i) for i in range(self._vocab_size)}
        kwargs.setdefault("unk_token", chr(0))
        kwargs.setdefault("pad_token", chr(1))
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._token_to_id)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        if not token:
            return self.unk_token_id or 0
        c = token[0]
        return ord(c) if ord(c) < self._vocab_size else (self.unk_token_id or 0)

    def _convert_id_to_token(self, idx: int) -> str:
        return self._id_to_token.get(int(idx), self.unk_token)

    def save_vocabulary(self, save_dir, prefix=None):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, (f"{prefix}-" if prefix else "") + "evo2_vocab.json")
        with open(path, "w", encoding="utf-8") as h:
            json.dump({"vocab_size": self._vocab_size}, h)
        return (path,)

class CharTokenizer:
    """Fallback character-level tokenizer for HyenaDNA-like models."""
    def __init__(self, vocab=None):
        self.vocab = vocab or {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
        self.unk_token_id = self.vocab.get("N", 11)
        self.pad_token_id = 0
        self.model_max_length = 32768
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            ids = [self.vocab.get(c.upper(), self.unk_token_id) for c in text]
            return {"input_ids": torch.tensor([ids])}
        all_ids = [[self.vocab.get(c.upper(), self.unk_token_id) for c in t] for t in text]
        if kwargs.get("padding"):
            ml = max(len(x) for x in all_ids)
            for x in all_ids: x.extend([self.pad_token_id] * (ml - len(x)))
        return {"input_ids": torch.tensor(all_ids)}
    def encode(self, text, **kwargs):
        if isinstance(text, str): return [self.vocab.get(c.upper(), self.unk_token_id) for c in text]
        return [[self.vocab.get(c.upper(), self.unk_token_id) for c in t] for t in text]

# -----------------------------------------------------------------------------
# 3. Applying Remaining Patches
# -----------------------------------------------------------------------------

def apply_triton_patch() -> None:
    """GB10 / Blackwell (sm_121) Triton bring-up.

    PyTorch 2.9 advertises max sm_120, but the system CUDA 13.0 ptxas ships
    native sm_121 support. Routing Triton through the system ptxas and
    declaring the arch with a +PTX fallback (matching the recipe established
    in lg-asm/run_optimized.sh) makes Triton kernels JIT cleanly on the GB10.

    Earlier revisions of this file rerouted ptxas through a wrapper that
    downgraded sm_121→sm_90; that produced kernel images the device could
    not actually load (`no kernel image available for execution`). We now
    keep the toolchain canonical.
    """
    if not torch.cuda.is_available():
        return
    try:
        cap = torch.cuda.get_device_capability()
    except Exception:
        return

    # Only Blackwell-class devices need explicit hints.
    if cap >= (12, 0):
        system_ptxas = "/usr/local/cuda/bin/ptxas"
        if os.path.exists(system_ptxas):
            os.environ.setdefault("TRITON_PTXAS_PATH", system_ptxas)
        # Tell PyTorch to advertise the real arch with PTX fallback so JIT
        # can lower to the device. Don't clobber a user-supplied list.
        arch = f"{cap[0]}.{cap[1]}+PTX"
        cur = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        if arch not in cur:
            os.environ["TORCH_CUDA_ARCH_LIST"] = (cur + ";" + arch).strip(";") if cur else arch
        # Clear any prior arch override that would force a wrong target.
        if os.environ.get("TRITON_OVERRIDE_ARCH"):
            os.environ.pop("TRITON_OVERRIDE_ARCH", None)

def register_transformers_classes():
    """Register local classes and patch dynamic module loading."""
    # 2. Intercept and bypass dynamic modules
    try:
        import transformers.dynamic_module_utils as dmu
        _orig_get_class = dmu.get_class_from_dynamic_module
        def _get_class_patched(class_ref, pretrained_model_name_or_path, **kwargs):
            cref = str(class_ref).lower()
            path_lower = str(pretrained_model_name_or_path).lower()
            # NTv3-specific overrides (these models have custom code we replace)
            if "ntv3" in cref and "token" in cref:
                return NTv3Tokenizer
            if "ntv3" in cref and "config" in cref:
                return Ntv3PreTrainedConfig
            # For other tokenizer references, only fall through to
            # PreTrainedTokenizerFast if the underlying model is genuinely
            # NTv3 (the original purpose of this patch). Otherwise let the
            # standard dynamic loader fetch the real class (HyenaDNA, Caduceus
            # etc.) so they don't get silently downgraded.
            if "tokenizer" in cref and "ntv3" in path_lower:
                return PreTrainedTokenizerFast
            return _orig_get_class(class_ref, pretrained_model_name_or_path, **kwargs)
        
        # Propagate patch across already loaded modules
        dmu.get_class_from_dynamic_module = _get_class_patched
        for name, module in list(sys.modules.items()):
            if name.startswith("transformers.") and hasattr(module, "get_class_from_dynamic_module"):
                if getattr(module, "get_class_from_dynamic_module") is _orig_get_class:
                    setattr(module, "get_class_from_dynamic_module", _get_class_patched)
    except Exception: pass

    # 3. Nuclear auto_map strip from AutoConfig
    try:
        from transformers import AutoConfig, AutoTokenizer
        _orig_from_dict = AutoConfig.from_dict
        def _from_dict_patched(dict_obj, **kwargs):
            if dict_obj: dict_obj.pop("auto_map", None)
            return _orig_from_dict(dict_obj, **kwargs)
        AutoConfig.from_dict = _from_dict_patched
        
        # Register with Auto Factories immediately using local classes
        AutoConfig.register("ntv3", Ntv3PreTrainedConfig)
        AutoConfig.register("ntv3_base_model", Ntv3PreTrainedConfig)
        AutoTokenizer.register(Ntv3PreTrainedConfig, slow_tokenizer_class=NTv3Tokenizer, fast_tokenizer_class=PreTrainedTokenizerFast)

        # Evo2 (arcinstitute) byte-level tokenizer registration
        try:
            AutoConfig.register("stripedhyena2", StripedHyena2Config)
            AutoTokenizer.register(StripedHyena2Config, slow_tokenizer_class=Evo2ByteTokenizer)
        except Exception:
            pass
    except Exception: pass

apply_triton_patch()
register_transformers_classes()
# Compatibility-patches load is silent by default. Set
# DNATOK_COMPAT_VERBOSE=1 to see the load notice (e.g. when debugging
# import order on a new host).
import os as _os
if _os.environ.get("DNATOK_COMPAT_VERBOSE", "0") not in ("0", "", "false"):
    print("[INFO] DNAtok Offline Compatibility Patches (Self-Contained) Applied Successfully.")
