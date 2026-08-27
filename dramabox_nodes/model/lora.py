"""DramaBox LoRA application via ComfyUI's native patch system.

v1 injected LoRAs by wrapping upstream's private `server._velocity_model`
with peft at runtime. v2 converts the trained LoRA's peft-style keys to
ComfyUI weight patches on the DiT's ModelPatcher (`add_patches`), the same
mechanism every core LoRA loader uses — so patches compose, apply at
computed strength on load, and unapply cleanly when the patcher unloads.

Supported key styles (upstream train.py saves peft format):
    base_model.model.<module>.lora_A.weight / .lora_B.weight
    <module>.lora_down.weight / .lora_up.weight
Alpha is read from an adjacent adapter_config.json when present; when
absent, no alpha scaling is applied — correct for upstream's default
rank 128 / alpha 128 training config (scale = 1.0).
"""
import json
import logging
import os

import comfy.utils

log = logging.getLogger("ComfyUI-DramaBox")

_PREFIXES = ("base_model.model.", "base_model.", "diffusion_model.", "model.")
_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight")
_UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight")


def _read_alpha(lora_path: str):
    cfg_path = os.path.join(os.path.dirname(lora_path), "adapter_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                return float(json.load(f).get("lora_alpha"))
        except Exception:
            pass
    return None


def _strip_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def apply_lora_to_dit(dit_patcher, lora_path: str, strength: float):
    """Return a cloned DiT patcher with the LoRA applied at `strength`."""
    sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    alpha = _read_alpha(lora_path)
    model_keys = set(dit_patcher.model.state_dict().keys())

    patches = {}
    unmatched = []
    for key, down in sd.items():
        suffix = next((s for s in _DOWN_SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            continue
        base = key[: -len(suffix)]
        up = None
        for down_sfx, up_sfx in zip(_DOWN_SUFFIXES, _UP_SUFFIXES):
            if suffix == down_sfx and base + up_sfx in sd:
                up = sd[base + up_sfx]
                break
        if up is None:
            unmatched.append(key)
            continue
        model_key = _strip_prefix(base) + ".weight"
        if model_key not in model_keys:
            unmatched.append(key)
            continue
        patches[model_key] = ("lora", (up, down, alpha, None))

    if not patches:
        raise RuntimeError(
            f"[DramaBox] No LoRA weights in {os.path.basename(lora_path)} matched the "
            f"audio DiT. First unmatched keys: {unmatched[:5]}. Is this a "
            f"DramaBox-trained LoRA?"
        )

    new_patcher = dit_patcher.clone()
    applied = new_patcher.add_patches(patches, strength)
    log.info(f"[DramaBox] LoRA {os.path.basename(lora_path)}: {len(applied)}/{len(patches)} "
             f"weight patches applied at strength {strength}"
             + (f" ({len(unmatched)} keys unmatched)" if unmatched else ""))
    return new_patcher
