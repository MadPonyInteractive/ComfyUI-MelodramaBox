"""Audio-only DiT loader.

Builds the 3.3B-parameter, 48-layer audio-only LTXModel (DramaBox's
fine-tuned `ltx-2.3-22b-dev-audio-only` branch) directly from
`dramabox-dit-v1.safetensors` — a port of the AudioOnlyConfigurator inside
upstream's src/inference_server.py, reading the model config from the
checkpoint's safetensors metadata with the same fallback defaults.

The model is built on the offload device (CPU) and wrapped in a
ModelPatcher; comfy moves it to the GPU when sampling actually needs it.
"""
import json
import logging

import torch
from safetensors import safe_open

from ..vendor import mdb_ltx_core  # noqa: F401  (sys.path bootstrap)
from mdb_ltx_core.loader import DummyRegistry
from mdb_ltx_core.loader.sd_ops import SDOps
from mdb_ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from mdb_ltx_core.model.model_protocol import ModelConfigurator
from mdb_ltx_core.model.transformer.attention import AttentionFunction
from mdb_ltx_core.model.transformer.model import LTXModel, LTXModelType
from mdb_ltx_core.model.transformer.rope import LTXRopeType
from mdb_ltx_core.model.transformer.text_projection import create_caption_projection

import comfy.model_management as mm

from . import patcher as patching

log = logging.getLogger("ComfyUI-DramaBox")


class AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
    @classmethod
    def from_config(cls, cfg):
        t = cfg.get("transformer", {})
        cp = None
        if not t.get("caption_proj_before_connector", False):
            with torch.device("meta"):
                cp = create_caption_projection(t, audio=True)
        return LTXModel(
            model_type=LTXModelType.AudioOnly,
            audio_num_attention_heads=t.get("audio_num_attention_heads", 32),
            audio_attention_head_dim=t.get("audio_attention_head_dim", 64),
            audio_in_channels=t.get("audio_in_channels", 128),
            audio_out_channels=t.get("audio_out_channels", 128),
            num_layers=t.get("num_layers", 48),
            audio_cross_attention_dim=t.get("audio_cross_attention_dim", 2048),
            norm_eps=t.get("norm_eps", 1e-6),
            attention_type=AttentionFunction(t.get("attention_type", "default")),
            positional_embedding_theta=10000.0,
            audio_positional_embedding_max_pos=[20.0],
            timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=t.get("use_middle_indices_grid", True),
            rope_type=LTXRopeType(t.get("rope_type", "interleaved")),
            double_precision_rope=t.get("frequencies_precision", False) == "float64",
            apply_gated_attention=t.get("apply_gated_attention", False),
            audio_caption_projection=cp,
            cross_attention_adaln=t.get("cross_attention_adaln", False),
        )


def read_checkpoint_config(dit_path: str) -> dict:
    with safe_open(dit_path, framework="pt") as f:
        meta = f.metadata() or {}
    if "config" in meta:
        return json.loads(meta["config"])
    log.warning(
        f"[DramaBox] {dit_path} has no 'config' in its safetensors metadata — "
        "falling back to the known dramabox-dit-v1 architecture defaults."
    )
    return {}


DIT_SD_OPS = SDOps("AO").with_matching(prefix="model.diffusion_model.").with_replacement(
    "model.diffusion_model.", "")


def load_dit(dit_path: str, dtype: torch.dtype):
    """Build the audio-only DiT from `dit_path` on the offload device and
    return (ModelPatcher, checkpoint_config)."""
    config = read_checkpoint_config(dit_path)
    builder = SingleGPUModelBuilder(
        model_path=dit_path,
        model_class_configurator=AudioOnlyConfigurator,
        model_sd_ops=DIT_SD_OPS,
        registry=DummyRegistry(),
    )
    offload = mm.unet_offload_device()
    model = builder.build(device=offload, dtype=dtype).eval()

    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_left:
        raise RuntimeError(
            f"[DramaBox] DiT checkpoint {dit_path} is missing weights for: "
            f"{meta_left[:8]}{'...' if len(meta_left) > 8 else ''} — the file is "
            "corrupt/incomplete or not a DramaBox audio-only DiT checkpoint."
        )

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    size_gb = patching.module_vram_bytes(model) / 1e9
    log.info(f"[DramaBox] Audio DiT built: {n_params:.1f}B params, {size_gb:.1f} GB ({dtype})")
    return patching.make_patcher(model), config
