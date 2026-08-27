"""VRAM utility node.

v2.0: the manual evict/rebuild machinery is gone — every model component
is ModelPatcher-wrapped and registered with comfy.model_management, so
ComfyUI itself evicts DramaBox models under memory pressure (e.g. when an
image workflow runs next). This node remains as an explicit "free
everything now" convenience for graph authors who want VRAM released at a
specific point rather than at next-load time.
"""

# MODIFIED by Mad Pony Interactive (fork of comfyui-melodramabox 2.1.0).
# Changes are marked `MPI-607` below and described in the repository NOTICE file.
import gc
import logging

import comfy.model_management as mm

from .model.patcher import vram_report

log = logging.getLogger("ComfyUI-DramaBox")


class AnyType(str):
    """Wildcard socket type that matches anything (standard ComfyUI idiom)."""
    def __ne__(self, other):
        return False


any_type = AnyType("*")


class DramaBoxUnloadModels:
    """Passthrough: unloads all comfy-managed models (including DramaBox's
    DiT/VAE patchers) and frees the Gemma text encoder if connected. Wire
    between two nodes to force the release at that point in the graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"trigger": (any_type, {})},
            "optional": {
                "text_encoder": ("DRAMABOX_TEXTENC", {"tooltip":
                    "Connect to also free the Gemma LLM (it isn't patcher-managed "
                    "when 4-bit-quantized, so comfy can't evict it on its own)."}),
            },
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "unload"
    CATEGORY = "DramaBox/vram"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # MPI-607: this node exists ONLY for its side effect. Without this,
        # ComfyUI caches it whenever its inputs are unchanged and the unload
        # never happens -- which is invisible except as an OOM further down the
        # graph. NaN != NaN, so the node always re-executes.
        return float("NaN")

    def unload(self, trigger, text_encoder=None):
        if text_encoder is not None:
            text_encoder.free()
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        log.info(f"[DramaBox] Models unloaded. {vram_report()}")
        return (trigger,)
