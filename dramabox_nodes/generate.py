"""Conditioning + sampling + decoding nodes.

v2.0: this is a real pipeline now. DramaBoxTextEncode actually runs Gemma
(v1's version was a string-packaging no-op because upstream's TTSServer
only took prompt strings); the sampler runs the vendored flow-matching
loop tensor-native end-to-end with progress + interrupt; decoding is a
separate VAE node like every other ComfyUI pipeline. No temp WAV files
anywhere.
"""
import logging

import torch

from . import config
from . import sampling

log = logging.getLogger("ComfyUI-DramaBox")


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------
class DramaBoxTextEncode:
    """Encodes the prompt (and negative prompt) through Gemma-3 + the audio
    embeddings connector into DiT cross-attention conditioning.

    Prompt structure: `<speaker description>, "<dialogue>" <action> "<more
    dialogue>"` — quoted text is spoken (including onomatopoeia like
    "Hahaha"); unquoted text is a stage direction the model performs but
    doesn't vocalize.

    Thanks to ComfyUI's node caching, Gemma only runs when the prompt text
    (or encoder settings) change — re-queues with a new seed are free."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("DRAMABOX_TEXTENC",),
                "prompt": ("STRING", {"multiline": True,
                                      "default": 'A woman speaks warmly, "Hello, how are you today?"'}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": config.DEFAULT_NEGATIVE_PROMPT}),
            }
        }

    RETURN_TYPES = ("DRAMABOX_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "DramaBox/conditioning"

    def encode(self, text_encoder, prompt, negative_prompt):
        if not prompt or not prompt.strip():
            raise ValueError("DramaBoxTextEncode: prompt is empty.")
        encodings = text_encoder.encode([prompt, negative_prompt])
        return ({
            "chunks": [{"text": prompt, "positive": encodings[0].cpu()}],
            "negative": encodings[1].cpu(),
        },)


class DramaBoxLongFormTextEncode:
    """Long-form variant: splits the prompt into <=max_chunk_duration
    chunks with upstream's sentence-aware chunker, then encodes each chunk.
    The sampler generates per chunk (same voice reference throughout, so
    the speaker stays coherent) and the VAE-decode node crossfades the
    joins."""

    @classmethod
    def INPUT_TYPES(cls):
        base = DramaBoxTextEncode.INPUT_TYPES()
        base["required"]["target_chunk_duration"] = ("FLOAT", {
            "default": config.TARGET_CHUNK_DURATION_DEFAULT, "min": 5.0,
            "max": config.MAX_CHUNK_DURATION_HARD_CAP})
        base["required"]["max_chunk_duration"] = ("FLOAT", {
            "default": config.MAX_CHUNK_DURATION_HARD_CAP, "min": 5.0,
            "max": config.MAX_CHUNK_DURATION_HARD_CAP})
        base["required"]["duration_multiplier"] = ("FLOAT", {
            "default": 1.1, "min": 0.5, "max": 3.0, "step": 0.05,
            "tooltip": "Breathing room over the estimated speech duration, used to plan chunk sizes."})
        return base

    RETURN_TYPES = ("DRAMABOX_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "DramaBox/conditioning"

    def encode(self, text_encoder, prompt, negative_prompt,
               target_chunk_duration, max_chunk_duration, duration_multiplier):
        if not prompt or not prompt.strip():
            raise ValueError("DramaBoxLongFormTextEncode: prompt is empty.")

        from .vendor import mdb_dramabox  # noqa: F401  (sys.path bootstrap)
        from mdb_dramabox.text_chunker import chunk_prompt_for_duration

        chunks = chunk_prompt_for_duration(
            prompt,
            max_duration_s=max_chunk_duration,
            target_duration_s=target_chunk_duration,
            duration_multiplier=duration_multiplier,
        )
        log.info(f"[DramaBox] Long-form prompt split into {len(chunks)} chunk(s).")

        texts = [c.text for c in chunks]
        encodings = text_encoder.encode(texts + [negative_prompt])
        return ({
            "chunks": [{"text": t, "positive": enc.cpu()} for t, enc in zip(texts, encodings[:-1])],
            "negative": encodings[-1].cpu(),
        },)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
class DramaBoxSampler:
    """Flow-matching sampler for the DramaBox audio DiT. Outputs audio
    latents for DramaBoxVAEDecode.

    Handles both single-prompt and long-form (chunked) conditioning; long-
    form chunks use seed, seed+1, seed+2, ... and the same voice reference
    so the speaker stays coherent.

    `quality_preset` sets steps/cfg/stg together; pick `custom` to use the
    widget values (steps genuinely plumb into the scheduler now — in v1
    upstream hardcoded 30)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit": ("DRAMABOX_DIT",),
                "conditioning": ("DRAMABOX_CONDITIONING",),
                "quality_preset": (["draft", "default", "high", "custom"], {"default": "default"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200,
                                  "tooltip": "Used when quality_preset is 'custom'."}),
                "cfg_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 15.0, "step": 0.1,
                                        "tooltip": "Lower = more natural, higher = more text-faithful. "
                                                   "Used when quality_preset is 'custom'."}),
                "stg_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                                        "tooltip": "Spatio-temporal guidance strength. "
                                                   "Used when quality_preset is 'custom'."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "duration_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.5,
                    "tooltip": "0 = estimate from the prompt. Ignored for long-form (chunked) "
                               "conditioning, which sizes each chunk from its own text."}),
                "duration_multiplier": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 3.0, "step": 0.05,
                    "tooltip": "Breathing room over the estimated speech duration."}),
                "cfg_rescale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "-1 = automatic cfg-aware schedule (prevents clipping at high cfg). "
                               "0 disables; 0..1 fixes the rescale strength."}),
            },
            "optional": {
                "voice_ref": ("DRAMABOX_VOICE_REF", {"tooltip": "Zero-shot voice cloning reference; "
                              "omit for prompt-only speaker control"}),
            },
        }

    RETURN_TYPES = ("DRAMABOX_AUDIO_LATENT",)
    RETURN_NAMES = ("audio_latent",)
    FUNCTION = "generate"
    CATEGORY = "DramaBox/sampling"

    def generate(self, dit, conditioning, quality_preset, steps, cfg_scale, stg_scale,
                 seed, duration_seconds, duration_multiplier, cfg_rescale, voice_ref=None):
        preset = config.QUALITY_PRESETS.get(quality_preset)
        if preset is not None:
            steps, cfg_scale, stg_scale = preset["steps"], preset["cfg_scale"], preset["stg_scale"]

        rescale = sampling.auto_rescale_for_cfg(cfg_scale) if cfg_rescale < 0 else cfg_rescale

        ref_latent = voice_ref["latent"] if voice_ref is not None else None
        ref_strength = voice_ref["strength"] if voice_ref is not None else 1.0

        chunks = conditioning["chunks"]
        negative = conditioning.get("negative")
        latents = []
        for i, chunk in enumerate(chunks):
            if len(chunks) == 1 and duration_seconds > 0:
                duration = duration_seconds
            else:
                duration = sampling.estimate_duration(chunk["text"], duration_multiplier)
            if len(chunks) > 1:
                log.info(f"[DramaBox] Chunk {i + 1}/{len(chunks)}: ~{duration:.1f}s")
            latent = sampling.generate_audio_latent(
                dit_patcher=dit["patcher"],
                positive=chunk["positive"],
                negative=negative,
                duration_s=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                rescale_scale=rescale,
                seed=seed + i,
                voice_ref_latent=ref_latent,
                voice_ref_strength=ref_strength,
            )
            latents.append(latent.cpu())

        return ({"latents": latents},)


# ---------------------------------------------------------------------------
# VAE decode
# ---------------------------------------------------------------------------
class DramaBoxVAEDecode:
    """Decodes audio latents through the VAE decoder + BigVGAN vocoder to a
    48 kHz waveform. Long-form chunk latents are stitched with an
    equal-power crossfade (constant perceived loudness through joins)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_vae": ("DRAMABOX_AUDIO_VAE",),
                "audio_latent": ("DRAMABOX_AUDIO_LATENT",),
                "crossfade_ms": ("INT", {"default": 50, "min": 0, "max": 500,
                    "tooltip": "Crossfade between long-form chunks. Irrelevant for single-chunk output."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "DramaBox/sampling"

    def decode(self, audio_vae, audio_latent, crossfade_ms):
        out_waveform = None
        sample_rate = None
        for latent in audio_latent["latents"]:
            decoded = audio_vae.decode(latent)
            wav = decoded.waveform.cpu().float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if out_waveform is None:
                out_waveform, sample_rate = wav, decoded.sampling_rate
                continue
            if decoded.sampling_rate != sample_rate:
                raise RuntimeError(f"Sample-rate mismatch between chunks: "
                                   f"{sample_rate} vs {decoded.sampling_rate}")
            if wav.shape[0] != out_waveform.shape[0]:
                if wav.shape[0] == 1:
                    wav = wav.repeat(out_waveform.shape[0], 1)
                elif out_waveform.shape[0] == 1:
                    out_waveform = out_waveform.repeat(wav.shape[0], 1)
            out_waveform = sampling.equal_power_crossfade(
                out_waveform, wav, sample_rate, fade_ms=float(crossfade_ms))

        duration = out_waveform.shape[-1] / sample_rate
        log.info(f"[DramaBox] Decoded {duration:.1f}s of audio at {sample_rate} Hz")
        return ({"waveform": out_waveform.unsqueeze(0), "sample_rate": sample_rate},)
