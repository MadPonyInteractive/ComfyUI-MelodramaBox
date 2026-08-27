"""ComfyUI-native model components for DramaBox.

Each component (audio DiT, Gemma text encoder, audio VAE/vocoder) is built
from the vendored LTX-2 code and wrapped in its own
`comfy.model_patcher.ModelPatcher`, so ComfyUI's memory manager owns
loading, offloading, and eviction — the v1 monolithic-TTSServer wrapper is
gone.
"""
