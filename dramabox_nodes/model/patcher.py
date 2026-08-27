"""ModelPatcher plumbing shared by all DramaBox components.

Wrapping each component in a ModelPatcher and loading it with
`comfy.model_management.load_models_gpu` is what makes this pack a citizen
of ComfyUI's memory manager: comfy tracks the VRAM each patcher occupies,
evicts DramaBox models automatically when another workflow needs the
space, and honours --lowvram / --reserve-vram without any DramaBox-side
logic. (v1 kept a monolithic TTSServer that comfy could not see, which is
why re-running its loader was an easy OOM.)
"""
import logging

import torch

import comfy.model_management as mm
import comfy.model_patcher

log = logging.getLogger("ComfyUI-DramaBox")


def make_patcher(module: torch.nn.Module) -> comfy.model_patcher.ModelPatcher:
    return comfy.model_patcher.ModelPatcher(
        module,
        load_device=mm.get_torch_device(),
        offload_device=mm.unet_offload_device(),
    )


def load_patchers(patchers, extra_inference_memory: float = 0.0):
    """Bring the given patchers onto the GPU (evicting other models if comfy
    decides that's needed), reserving `extra_inference_memory` bytes on top
    for activations. Call this immediately before running the modules —
    ComfyUI issue #12440 is the cautionary tale for skipping it (parts of
    ACE-Step silently sampled on CPU)."""
    patchers = [p for p in patchers if p is not None]
    if patchers:
        mm.load_models_gpu(patchers, memory_required=extra_inference_memory)


def module_vram_bytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters()) + sum(
        b.numel() * b.element_size() for b in module.buffers()
    )


def vram_report() -> str:
    dev = mm.get_torch_device()
    try:
        free, total = mm.get_free_memory(dev), mm.get_total_memory(dev)
        used_gb = (total - free) / (1024 ** 3)
        return f"{used_gb:.2f} / {total / (1024 ** 3):.2f} GB used on {dev}"
    except Exception:
        return "VRAM report unavailable"
