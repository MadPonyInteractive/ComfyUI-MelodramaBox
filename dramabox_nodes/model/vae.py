"""Audio VAE (encoder/decoder) + BigVGAN vocoder from
dramabox-audio-components.safetensors, each in its own ModelPatcher.

The encoder is only needed for zero-shot voice references; the decoder +
vocoder only at the end of sampling — keeping them as separate patchers
lets comfy load exactly what a given step needs.
"""
import logging

import torch

from ..vendor import mdb_ltx_core  # noqa: F401  (sys.path bootstrap)
from mdb_ltx_core.loader import DummyRegistry
from mdb_ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from mdb_ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from mdb_ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from mdb_ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from mdb_ltx_core.types import Audio

import comfy.model_management as mm

from . import patcher as patching

log = logging.getLogger("ComfyUI-DramaBox")


def _build(components_path: str, configurator, sd_ops, dtype: torch.dtype, what: str):
    builder = SingleGPUModelBuilder(
        model_path=components_path,
        model_class_configurator=configurator,
        model_sd_ops=sd_ops,
        registry=DummyRegistry(),
    )
    module = builder.build(device=mm.unet_offload_device(), dtype=dtype).eval()
    meta_left = [n for n, p in module.named_parameters() if p.is_meta]
    if meta_left:
        raise RuntimeError(
            f"[DramaBox] Audio {what} is missing weights ({meta_left[:5]}...) — "
            f"{components_path} doesn't look like dramabox-audio-components.safetensors."
        )
    return patching.make_patcher(module)


class DramaBoxAudioVAE:
    """Holds encoder / decoder / vocoder patchers built from one
    audio-components checkpoint. Encoder is built lazily (voice refs only)."""

    def __init__(self, components_path: str, dtype: torch.dtype):
        self.components_path = components_path
        self.dtype = dtype
        self.decoder_patcher = _build(
            components_path, AudioDecoderConfigurator, AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, dtype, "VAE decoder")
        self.vocoder_patcher = _build(
            components_path, VocoderConfigurator, VOCODER_COMFY_KEYS_FILTER, dtype, "vocoder")
        self._encoder_patcher = None
        self.output_sample_rate = int(self.vocoder_patcher.model.output_sampling_rate)
        log.info(f"[DramaBox] Audio VAE ready (output {self.output_sample_rate} Hz)")

    @property
    def encoder_patcher(self):
        if self._encoder_patcher is None:
            self._encoder_patcher = _build(
                self.components_path, AudioEncoderConfigurator,
                AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, self.dtype, "VAE encoder")
        return self._encoder_patcher

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Waveform (B, C, T) → audio latent (B, C, frames, mel_bins).
        encode_audio resamples internally to the encoder's native rate."""
        patching.load_patchers([self.encoder_patcher])
        audio = Audio(waveform=waveform, sampling_rate=sample_rate)
        return vae_encode_audio(audio, self.encoder_patcher.model, None)

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> Audio:
        """Audio latent → Audio(waveform (C, T) float32, 48 kHz)."""
        patching.load_patchers([self.decoder_patcher, self.vocoder_patcher])
        device = mm.get_torch_device()
        return vae_decode_audio(
            latent.to(device=device, dtype=self.dtype),
            self.decoder_patcher.model,
            self.vocoder_patcher.model,
        )
