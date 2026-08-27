"""
audio_prep.py
==============
Shared helpers for preparing **zero-shot voice-cloning reference audio**.

DramaBox needs no per-speaker training or LoRA to clone a voice: an
optional 10+ second reference clip is enough to steer timbre at inference
time (upstream README: "an optional 10-second voice reference clones the
target timbre"). That's what "zero-shot" means here, as opposed to
DramaBoxLoRALoader, which is for a *trained* voice/style adapter.

This module handles the practical mess around that feature:
  - accepting one or many reference clips and stitching them into a single
    usable reference (useful when you only have several short clips of a
    speaker, none individually hitting the 10s recommendation)
  - trimming leading/trailing silence so the reference is dense with voiced
    audio rather than dead air
  - resampling to the audio VAE's native rate
  - capping to a sane max length (encoding cost grows with reference
    length for negligible quality gain past ~30s)
  - basic sanity checks (silence-only clip, clipping/too-quiet levels)
"""
import logging

import torch

from . import config

log = logging.getLogger("ComfyUI-DramaBox")


def _resample(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return waveform
    import torchaudio
    return torchaudio.functional.resample(waveform, sr, target_sr)


def _trim_silence(waveform: torch.Tensor, sr: int, threshold_db: float) -> torch.Tensor:
    """Trim leading/trailing near-silence so the reference is voice-dense.
    Simple energy-threshold trim rather than a full VAD model — good enough
    for cleaning up room-tone padding on a cloning clip without pulling in
    heavier dependencies."""
    if waveform.numel() == 0:
        return waveform
    amp = waveform.abs()
    peak = amp.max()
    if peak <= 1e-6:
        return waveform  # fully silent clip; nothing to trim, caller should warn
    thresh = peak * (10 ** (-threshold_db / 20))
    mask = (amp > thresh).any(dim=0)
    nonzero = mask.nonzero()
    if nonzero.numel() == 0:
        return waveform
    start, end = nonzero[0].item(), nonzero[-1].item() + 1
    return waveform[:, start:end]


def _crossfade_concat(clips, sr: int, crossfade_ms: int) -> torch.Tensor:
    if len(clips) == 1:
        return clips[0]
    fade_len = int(sr * crossfade_ms / 1000)
    out = clips[0]
    for nxt in clips[1:]:
        fade_len_i = min(fade_len, out.shape[-1], nxt.shape[-1])
        if fade_len_i <= 0:
            out = torch.cat([out, nxt], dim=-1)
            continue
        fade_out = torch.linspace(1.0, 0.0, fade_len_i)
        fade_in = torch.linspace(0.0, 1.0, fade_len_i)
        head = out[:, :-fade_len_i]
        tail = out[:, -fade_len_i:] * fade_out + nxt[:, :fade_len_i] * fade_in
        rest = nxt[:, fade_len_i:]
        out = torch.cat([head, tail, rest], dim=-1)
    return out


_reuse_denoiser = None  # lazy singleton; False = unavailable this session


def denoise_reference(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Clean a mono (1, T) reference with NVIDIA RE-USE (SEMamba).

    Denoising the *reference* (not the generated output) gives the model a
    clean speaker anchor without suppressing generated paralinguistics —
    RE-USE on output audio would eat laughs/breaths/sighs as "noise".
    Degrades gracefully (returns the input) if the optional mamba-ssm deps
    or weights are unavailable. RE-USE is NSCLv1 (non-commercial).
    """
    global _reuse_denoiser
    if _reuse_denoiser is False:
        return waveform

    import comfy.model_management as mm
    device = mm.get_torch_device()

    if _reuse_denoiser is None:
        try:
            from .vendor import mdb_dramabox  # noqa: F401  (sys.path bootstrap)
            from mdb_dramabox.super_resolution import REUSEUpsampler
            _reuse_denoiser = REUSEUpsampler(target_sr=sr, device=device, chunk_size_s=1.0)
        except Exception as e:
            log.warning(f"[DramaBox] Voice-ref denoise disabled (RE-USE unavailable: {e}). "
                        "Install requirements-reuse deps to enable, or set denoise=False.")
            _reuse_denoiser = False
            return waveform

    mono = waveform.mean(dim=0).contiguous() if waveform.dim() == 2 else waveform.contiguous()
    try:
        cleaned, _ = _reuse_denoiser(mono, in_sr=sr)
    except Exception as e:
        log.warning(f"[DramaBox] RE-USE denoise failed ({e}) — using the raw reference.")
        return waveform
    if cleaned.dim() == 1:
        cleaned = cleaned.unsqueeze(0)
    elif cleaned.dim() == 2 and cleaned.shape[0] != 1:
        cleaned = cleaned[:1]
    return cleaned.to(device="cpu", dtype=waveform.dtype)


def prepare_zero_shot_reference(
    clips,
    trim_silence: bool = True,
    target_sr: int = None,
    max_duration_sec: float = None,
    min_recommended_sec: float = None,
) -> dict:
    """
    clips: list of (waveform[C,T] torch.Tensor, sample_rate) tuples, in the
    order they should be stitched. A single-item list is the common case
    (one reference clip); multiple items let several short clips of the
    same speaker be combined into one usable reference.

    Returns a DRAMABOX_VOICE_REF dict: {"waveform", "sample_rate",
    "duration", "num_source_clips"}.
    """
    target_sr = target_sr or config.VOICE_REF_PREP_SAMPLE_RATE
    max_duration_sec = max_duration_sec or config.ZERO_SHOT_MAX_USEFUL_SEC
    min_recommended_sec = min_recommended_sec or config.ZERO_SHOT_MIN_RECOMMENDED_SEC

    prepped = []
    for waveform, sr in clips:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # downmix to mono
        waveform = _resample(waveform, sr, target_sr)
        if trim_silence:
            trimmed = _trim_silence(waveform, target_sr, config.ZERO_SHOT_TRIM_SILENCE_DB)
            if trimmed.shape[-1] == 0:
                log.warning("[DramaBox] A zero-shot reference clip is silent after trimming; skipping it.")
                continue
            waveform = trimmed
        prepped.append(waveform)

    if not prepped:
        raise ValueError(
            "No usable audio left after preparing zero-shot voice reference clips "
            "(all clips were silent or empty)."
        )

    stitched = _crossfade_concat(prepped, target_sr, config.ZERO_SHOT_CROSSFADE_MS)

    duration = stitched.shape[-1] / target_sr
    if duration > max_duration_sec:
        keep_samples = int(max_duration_sec * target_sr)
        stitched = stitched[:, :keep_samples]
        duration = max_duration_sec
        log.info(f"[DramaBox] Zero-shot reference trimmed to {max_duration_sec:.0f}s cap.")
    if duration < min_recommended_sec:
        log.warning(
            f"[DramaBox] Zero-shot voice reference is only {duration:.1f}s after prep "
            f"(recommended {min_recommended_sec:.0f}s+). Cloned timbre may be less stable — "
            "consider providing additional clips of the same speaker to stitch together."
        )

    peak = stitched.abs().max().item()
    if peak < 0.02:
        log.warning("[DramaBox] Zero-shot reference audio is very quiet (peak "
                    f"{peak:.4f}); this can weaken voice cloning. Consider normalizing input levels.")
    elif peak > 0.999:
        log.warning("[DramaBox] Zero-shot reference audio appears clipped; this can "
                    "introduce artifacts into the cloned voice.")

    return {
        "waveform": stitched,
        "sample_rate": target_sr,
        "duration": duration,
        "num_source_clips": len(prepped),
    }
