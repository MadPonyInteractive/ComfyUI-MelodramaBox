"""Latent upsampler model components."""

from mdb_ltx_core.model.upsampler.model import LatentUpsampler, upsample_video
from mdb_ltx_core.model.upsampler.model_configurator import LatentUpsamplerConfigurator

__all__ = [
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "upsample_video",
]
