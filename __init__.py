"""
ComfyUI-MelodramaBox
====================
Native ComfyUI nodes for resemble-ai/DramaBox — expressive, prompt-driven
TTS + zero-shot voice cloning built on the LTX-2.3 audio-only DiT (3.3B).

v2.0: fully componentized. The audio DiT, Gemma-3 text encoder, and audio
VAE/vocoder load separately from ComfyUI's standard model folders, each in
its own comfy ModelPatcher — ComfyUI's memory manager owns offloading and
eviction, generation is tensor-native end-to-end (no temp WAVs, no runtime
git clone of upstream), sampling has real progress/interrupt support, and
step counts actually apply. Inference model code is vendored at a pinned
upstream commit (dramabox_nodes/vendor/ATTRIBUTION.md).

Typical graph:
    DiTLoader ─────────────┐
    TextEncoderLoader → TextEncode ─┤
    AudioVAELoader ─┬─ VoiceReference ─┤
                    │                  ▼
                    │            Sampler → VAEDecode → SaveAudio
                    └──────────────────────────┘
"""

from .dramabox_nodes.loaders import (
    DramaBoxDiTLoader,
    DramaBoxTextEncoderLoader,
    DramaBoxAudioVAELoader,
    DramaBoxLoRALoader,
    DramaBoxVoiceReferenceLoader,
    DramaBoxVoiceReferenceBatch,
)
from .dramabox_nodes.generate import (
    DramaBoxTextEncode,
    DramaBoxLongFormTextEncode,
    DramaBoxSampler,
    DramaBoxVAEDecode,
)
from .dramabox_nodes.audio_io import (
    DramaBoxSaveAudio,
    DramaBoxApplyWatermark,
    DramaBoxWatermarkCheck,
)
from .dramabox_nodes.model_management import DramaBoxUnloadModels
from .dramabox_nodes.downloader import DramaBoxDownloadModels
from .dramabox_nodes.utils import DramaBoxEstimateDuration
from .dramabox_nodes.finetune_extract import DramaBoxExtractAudioDiT

NODE_CLASS_MAPPINGS = {
    "DramaBoxDownloadModels": DramaBoxDownloadModels,
    "DramaBoxDiTLoader": DramaBoxDiTLoader,
    "DramaBoxTextEncoderLoader": DramaBoxTextEncoderLoader,
    "DramaBoxAudioVAELoader": DramaBoxAudioVAELoader,
    "DramaBoxLoRALoader": DramaBoxLoRALoader,
    "DramaBoxVoiceReferenceLoader": DramaBoxVoiceReferenceLoader,
    "DramaBoxVoiceReferenceBatch": DramaBoxVoiceReferenceBatch,
    "DramaBoxTextEncode": DramaBoxTextEncode,
    "DramaBoxLongFormTextEncode": DramaBoxLongFormTextEncode,
    "DramaBoxSampler": DramaBoxSampler,
    "DramaBoxVAEDecode": DramaBoxVAEDecode,
    "DramaBoxSaveAudio": DramaBoxSaveAudio,
    "DramaBoxApplyWatermark": DramaBoxApplyWatermark,
    "DramaBoxWatermarkCheck": DramaBoxWatermarkCheck,
    "DramaBoxUnloadModels": DramaBoxUnloadModels,
    "DramaBoxEstimateDuration": DramaBoxEstimateDuration,
    "DramaBoxExtractAudioDiT": DramaBoxExtractAudioDiT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxDownloadModels": "DramaBox Download Models",
    "DramaBoxDiTLoader": "DramaBox DiT Loader",
    "DramaBoxTextEncoderLoader": "DramaBox Text Encoder Loader (Gemma-3)",
    "DramaBoxAudioVAELoader": "DramaBox Audio VAE Loader",
    "DramaBoxLoRALoader": "DramaBox LoRA Loader",
    "DramaBoxVoiceReferenceLoader": "DramaBox Zero-Shot Voice Reference",
    "DramaBoxVoiceReferenceBatch": "DramaBox Zero-Shot Voice Reference (Batch)",
    "DramaBoxTextEncode": "DramaBox Text Encode",
    "DramaBoxLongFormTextEncode": "DramaBox Text Encode (Long-Form)",
    "DramaBoxSampler": "DramaBox Sampler",
    "DramaBoxVAEDecode": "DramaBox VAE Decode (Audio)",
    "DramaBoxSaveAudio": "DramaBox Save Audio",
    "DramaBoxApplyWatermark": "DramaBox Apply Watermark (Perth)",
    "DramaBoxWatermarkCheck": "DramaBox Watermark Detector",
    "DramaBoxUnloadModels": "DramaBox Unload Models",
    "DramaBoxEstimateDuration": "DramaBox Estimate Duration",
    "DramaBoxExtractAudioDiT": "DramaBox Extract Audio DiT (from full LTX-2.3)",
}

WEB_DIRECTORY = None
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
