"""Runtime GGUF loading for the DramaBox audio DiT.

Backend: **calcuis/gguf** (`pip install gguf-connector`) — the reader
(`gguf_connector.reader.GGUFReader`) and the torch dequant kernels
(`gguf_connector.quant2.dequantize_tensor`, all K-quants). calcuis's pip lib
ships those kernels but not the tensor/ops glue (that lives in its ComfyUI
node), so the small glue here — `GGMLTensor`, `GGMLCastLinear` — is ours,
following the well-known city96/ComfyUI-GGUF pattern.

Why quantized weights are stored as **buffers, not Parameters**: wrapping a
`GGMLTensor` in `torch.nn.Parameter` strips the subclass (Parameter re-creates
the tensor), losing the `tensor_type`/`tensor_shape` metadata the dequant
needs. Buffers keep the subclass, and `nn.Module.to(device)` moves them via
`GGMLTensor.to()` (which copies the metadata), so ComfyUI's ModelPatcher can
stream the quantized weights to the GPU exactly like any other model.

This build path is deliberately manual (not `SingleGPUModelBuilder.build`)
because that builder's `load_state_dict(assign=True)` would re-wrap every
weight in a Parameter and destroy the metadata.
"""
import json
import logging

import torch

from ..vendor import mdb_ltx_core  # noqa: F401  (sys.path bootstrap)

import comfy.model_management as mm

from ..model import patcher as patching
from ..model.dit import AudioOnlyConfigurator

log = logging.getLogger("ComfyUI-DramaBox")

_DIT_PREFIX = "model.diffusion_model."


# ---------------------------------------------------------------------------
# GGMLTensor — quantized-weight carrier (ported from city96/ComfyUI-GGUF, MIT;
# calcuis/gguf shares this pattern). Holds packed quant bytes + the metadata
# the dequant kernels read.
# ---------------------------------------------------------------------------
class GGMLTensor(torch.Tensor):
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape

    def __new__(cls, *args, tensor_type, tensor_shape, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


def _dequant_fn():
    """Lazy import of calcuis's torch dequant (clear error if missing)."""
    try:
        from gguf_connector.quant2 import dequantize_tensor
        return dequantize_tensor
    except ImportError as e:
        raise RuntimeError(
            "[DramaBox] GGUF loading needs calcuis/gguf: `pip install gguf-connector`. "
            f"({e})"
        ) from e


# ---------------------------------------------------------------------------
# GGMLCastLinear — dequantizes its weight on-the-fly in forward, mirroring the
# vendored Fp8CastLinear pattern. Weight/bias are GGMLTensor buffers.
# ---------------------------------------------------------------------------
class GGMLCastLinear(torch.nn.Linear):
    _deq = None  # set to the dequant fn by load_dit_gguf

    def forward(self, x):
        deq = GGMLCastLinear._deq
        w = deq(self.weight, x.dtype)
        b = deq(self.bias, x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w, b)


def _reclass_and_bufferize(linear: torch.nn.Linear, w, b):
    """Turn an nn.Linear into a GGMLCastLinear whose weight/bias are GGMLTensor
    buffers (see module docstring on why buffers, not Parameters)."""
    linear.__class__ = GGMLCastLinear
    for name, val in (("weight", w), ("bias", b)):
        linear._parameters.pop(name, None)
        linear._buffers.pop(name, None)
        if val is not None:
            linear.register_buffer(name, val, persistent=False)
        else:
            object.__setattr__(linear, name, None)


# ---------------------------------------------------------------------------
# Config + key handling
# ---------------------------------------------------------------------------
def _config_from_reader(reader) -> dict:
    field = reader.fields.get("dramabox.config")
    if field is None:
        log.warning("[DramaBox] GGUF has no 'dramabox.config' KV — falling back to "
                    "built-in dramabox-dit-v1 architecture defaults.")
        return {}
    try:
        raw = field.parts[field.data[-1]]
        return json.loads(bytes(raw).decode("utf-8"))
    except Exception as e:
        log.warning(f"[DramaBox] Could not parse dramabox.config KV ({e}); using defaults.")
        return {}


def _strip_prefix(name: str) -> str:
    return name[len(_DIT_PREFIX):] if name.startswith(_DIT_PREFIX) else name


def _set_module_tensor(model: torch.nn.Module, dotted: str, tensor: torch.Tensor, is_param: bool):
    *path, leaf = dotted.split(".")
    mod = model
    for p in path:
        mod = getattr(mod, p)
    if is_param:
        mod._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
    else:
        mod._parameters.pop(leaf, None)
        mod.register_buffer(leaf, tensor, persistent=False)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_dit_gguf(gguf_path: str, compute_dtype: torch.dtype):
    """Build the audio-only DiT with GGUF weights (dequantized on-the-fly) and
    return (ModelPatcher, config). Weights stay quantized in VRAM; each Linear
    dequantizes its weight per forward via calcuis's torch kernels."""
    deq = _dequant_fn()
    GGMLCastLinear._deq = deq

    try:
        from gguf_connector.reader import GGUFReader
    except ImportError as e:
        raise RuntimeError(
            "[DramaBox] GGUF loading needs calcuis/gguf: `pip install gguf-connector`. "
            f"({e})"
        ) from e

    reader = GGUFReader(gguf_path)
    config = _config_from_reader(reader)

    with torch.device("meta"):
        model = AudioOnlyConfigurator.from_config(config)
    model.requires_grad_(False)

    # Index GGUF tensors by stripped name → GGMLTensor. GGUF stores dims
    # reversed (ggml lists the fastest-varying dim first); the logical torch
    # shape is the reverse of the stored shape.
    ggml = {}
    for t in reader.tensors:
        name = _strip_prefix(t.name)
        logical_shape = torch.Size(int(x) for x in reversed(list(t.shape)))
        ggml[name] = GGMLTensor(torch.from_numpy(t.data),
                                tensor_type=t.tensor_type, tensor_shape=logical_shape)

    # Which target tensors are Linear weights/biases (→ stay quantized as
    # GGMLTensor buffers on a GGMLCastLinear) vs. plain tensors (→ dequant now).
    linear_tensors = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            base = f"{mod_name}." if mod_name else ""
            linear_tensors[base + "weight"] = (mod, "weight")
            if mod.bias is not None:
                linear_tensors[base + "bias"] = (mod, "bias")

    target_names = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    param_names = {n for n, _ in model.named_parameters()}
    missing = []

    # 1) Linear weights/biases → GGMLTensor buffers on a GGMLCastLinear.
    by_mod = {}
    for tname, (mod, attr) in linear_tensors.items():
        gt = ggml.get(tname)
        if gt is None:
            missing.append(tname)
            continue
        by_mod.setdefault(id(mod), {"mod": mod})[attr] = gt
    for entry in by_mod.values():
        _reclass_and_bufferize(entry["mod"], entry.get("weight"), entry.get("bias"))

    # 2) Everything else (scale_shift_table params, adaln norms/embeds) →
    #    dequantize to compute_dtype now (small, kept unquantized by convert).
    for tname in target_names - set(linear_tensors):
        gt = ggml.get(tname)
        if gt is None:
            missing.append(tname)
            continue
        _set_module_tensor(model, tname, deq(gt, compute_dtype), tname in param_names)

    if missing:
        raise RuntimeError(
            f"[DramaBox] GGUF {gguf_path} is missing {len(missing)} weights the audio DiT "
            f"requires, e.g. {sorted(missing)[:6]}. Not a DramaBox audio-only DiT GGUF, or "
            "the conversion dropped keys."
        )

    meta_left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                 if t.is_meta]
    if meta_left:
        raise RuntimeError(f"[DramaBox] GGUF load left meta tensors: {meta_left[:6]}")

    model.eval()
    size_gb = patching.module_vram_bytes(model) / 1e9
    log.info(f"[DramaBox] Audio DiT (GGUF) loaded: ~{size_gb:.1f} GB quantized, "
             f"dequant compute dtype {compute_dtype}")
    return patching.make_patcher(model), config
