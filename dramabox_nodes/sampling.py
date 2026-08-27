"""Native flow-matching sampling for the DramaBox audio DiT.

A port of upstream TTSServer.generate()'s sampling core with ComfyUI
integration the original lacks:
- `comfy.model_management.load_models_gpu` right before the loop (comfy
  decides what to evict; nothing silently runs on CPU),
- a `comfy.utils.ProgressBar` advancing per denoising step,
- interrupt checks every step (Cancel in the UI aborts within one step),
- a configurable step count (v1's presets carried a `steps` value that
  upstream hardcoded to 30 and silently ignored).
"""
import logging
from dataclasses import replace

import torch

import comfy.model_management as mm
import comfy.utils

from .vendor import mdb_ltx_core  # noqa: F401  (sys.path bootstrap)
from mdb_ltx_core.components.diffusion_steps import EulerDiffusionStep
from mdb_ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from mdb_ltx_core.components.noisers import GaussianNoiser
from mdb_ltx_core.components.patchifiers import AudioPatchifier
from mdb_ltx_core.components.schedulers import LTX2Scheduler
from mdb_ltx_core.model.transformer.model import X0Model
from mdb_ltx_core.tools import AudioLatentTools
from mdb_ltx_core.types import AudioLatentShape, VideoPixelShape
from mdb_ltx_pipelines.utils.denoisers import GuidedDenoiser
from mdb_ltx_pipelines.utils.helpers import post_process_latent
from mdb_dramabox.audio_conditioning import AudioConditionByReferenceLatent
from mdb_dramabox.duration_estimator import estimate_speech_duration

from . import config
from .model import patcher as patching

log = logging.getLogger("ComfyUI-DramaBox")

# Rough activation headroom for the 3.3B DiT forward passes (guider runs up
# to 3 sequential forwards per step: cond / uncond / perturbed).
DIT_INFERENCE_HEADROOM_BYTES = 3 * 1024**3


def estimate_duration(prompt: str, multiplier: float = 1.1) -> float:
    """Sentence-aware speech-duration estimate (upstream's estimator)."""
    return max(3.0, round(estimate_speech_duration(prompt) * multiplier, 1))


def auto_rescale_for_cfg(cfg: float) -> float:
    """CFG-aware std-rescale schedule preventing output clipping at high cfg
    (piecewise fit from upstream's empirical sweep — see inference_server.py)."""
    if cfg <= 2.0:
        return 0.0
    if cfg <= 3.0:
        return 0.6 * (cfg - 2.0)
    if cfg <= 4.0:
        return 0.6 + 0.2 * (cfg - 3.0)
    if cfg <= 8.0:
        return 0.8
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))


def _patch_silence_prior(latent: torch.Tensor) -> torch.Tensor:
    """End-of-clip silence-prior fix: the base DiT learned a hard "clip ends
    here" prior at the first patch-aligned latent frame past its 20 s
    training length (frames 512/513), which renders as a ~30 ms silence dip
    near 20.4 s in longer clips. Linear interpolation across the two frames
    removes it (upstream's fix, applied post-unpatchify)."""
    if latent.shape[2] <= 513:
        return latent
    f0, f1 = 511, 514
    n = f1 - f0
    patched = latent.clone()
    for f in (512, 513):
        t = (f - f0) / n
        patched[:, :, f, :] = (1.0 - t) * latent[:, :, f0, :] + t * latent[:, :, f1, :]
    return patched


@torch.inference_mode()
def generate_audio_latent(
    dit_patcher,
    positive: torch.Tensor,
    negative,
    duration_s: float,
    steps: int,
    cfg_scale: float,
    stg_scale: float,
    rescale_scale: float,
    seed: int,
    voice_ref_latent=None,
    voice_ref_strength: float = 1.0,
) -> torch.Tensor:
    """Run the flow-matching loop and return an unpatchified audio latent
    (B, C, frames, mel_bins) ready for the VAE decoder."""
    device = mm.get_torch_device()
    model = dit_patcher.model
    dtype = next(model.parameters()).dtype

    # Target latent shape from requested duration (25 latent fps, frames
    # rounded to the 8n+1 grid the patchifier expects).
    fps = config.AUDIO_LATENT_FPS
    n_frames = int(round(duration_s * fps)) + 1
    n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
    pixel_shape = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
    target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
    audio_tools = AudioLatentTools(patchifier=AudioPatchifier(patch_size=1), target_shape=target_shape)

    state = audio_tools.create_initial_state(device=device, dtype=dtype)

    if voice_ref_latent is not None:
        cond = AudioConditionByReferenceLatent(
            latent=voice_ref_latent.to(device=device, dtype=dtype),
            strength=voice_ref_strength,
        )
        state = cond.apply_to(state, audio_tools)

    generator = torch.Generator(device=device).manual_seed(seed)
    state = GaussianNoiser(generator=generator)(state, noise_scale=1.0)

    use_cfg = cfg_scale > 1.0 and negative is not None
    guider = MultiModalGuider(
        params=MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            stg_blocks=list(config.STG_BLOCKS),
            rescale_scale=rescale_scale,
            modality_scale=1.0,
        ),
        negative_context=negative.to(device=device, dtype=dtype) if use_cfg else None,
    )
    denoiser = GuidedDenoiser(
        v_context=None,
        a_context=positive.to(device=device, dtype=dtype),
        video_guider=None,
        audio_guider=guider,
    )

    sigmas = LTX2Scheduler().execute(steps=steps, latent=state.latent).to(device)

    patching.load_patchers([dit_patcher], extra_inference_memory=DIT_INFERENCE_HEADROOM_BYTES)
    x0 = X0Model(model)
    stepper = EulerDiffusionStep()

    n_steps = len(sigmas) - 1
    pbar = comfy.utils.ProgressBar(n_steps)
    log.info(f"[DramaBox] Sampling {duration_s:.1f}s audio: {n_steps} steps, "
             f"cfg={cfg_scale} stg={stg_scale} rescale={rescale_scale:.2f} seed={seed} | "
             f"{patching.vram_report()}")

    for step_idx in range(n_steps):
        mm.throw_exception_if_processing_interrupted()
        _, denoised_audio = denoiser(x0, None, state, sigmas, step_idx)
        denoised_audio = post_process_latent(denoised_audio, state.denoise_mask, state.clean_latent)
        state = replace(state, latent=stepper.step(state.latent, denoised_audio, sigmas, step_idx))
        pbar.update(1)

    state = audio_tools.clear_conditioning(state)
    state = audio_tools.unpatchify(state)
    return _patch_silence_prior(state.latent)


def prepare_voice_ref_waveform(waveform: torch.Tensor, sample_rate: int,
                               ref_seconds: float = None) -> torch.Tensor:
    """Upstream's reference conditioning prep: mono→stereo, tile/trim to
    exactly `ref_seconds`, peak-normalize to -4 dBFS. Input (C, T) or
    (B, C, T); returns (1, 2, T_ref)."""
    ref_seconds = ref_seconds or config.VOICE_REF_ENCODE_SECONDS
    w = waveform
    if w.dim() == 2:
        if w.shape[0] == 1:
            w = w.repeat(2, 1)
        w = w.unsqueeze(0)
    elif w.dim() == 3 and w.shape[1] == 1:
        w = w.repeat(1, 2, 1)
    target_samples = int(ref_seconds * sample_rate)
    if w.shape[-1] < target_samples:
        w = w.repeat(1, 1, (target_samples // w.shape[-1]) + 1)
    w = w[..., :target_samples]
    peak = w.abs().max()
    if peak > 0:
        w = w * (10 ** (config.VOICE_REF_PEAK_DBFS / 20) / peak)
    return w


def equal_power_crossfade(prev: torch.Tensor, nxt: torch.Tensor,
                          sample_rate: int, fade_ms: float = 50.0) -> torch.Tensor:
    """Equal-power crossfade concat `[prev | nxt]`, both (C, T) — constant
    perceived loudness through the join (a linear fade dips ~3 dB)."""
    fade_samples = int(round(fade_ms * 1e-3 * sample_rate))
    fade_samples = max(1, min(fade_samples, prev.shape[-1], nxt.shape[-1]))
    if fade_samples <= 1:
        return torch.cat([prev, nxt], dim=-1)
    t = torch.linspace(0.0, 1.0, fade_samples, device=prev.device, dtype=prev.dtype)
    fade_out = torch.cos(t * torch.pi / 2)
    fade_in = torch.sin(t * torch.pi / 2)
    mixed = prev[..., -fade_samples:] * fade_out + nxt[..., :fade_samples] * fade_in
    return torch.cat([prev[..., :-fade_samples], mixed, nxt[..., fade_samples:]], dim=-1)
