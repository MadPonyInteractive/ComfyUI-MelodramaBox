"""Gemma-3-12B text encoder + audio embeddings connector.

A port of the text-encoding half of upstream's PromptEncoder block
(vendor/mdb_ltx_pipelines/utils/blocks.py) with ComfyUI-native memory
behavior:

- the Gemma LLM is the single largest VRAM consumer in the pipeline
  (~8 GB pre-quantized 4-bit, ~24 GB bf16), so it is built lazily — only
  when a prompt actually needs encoding. ComfyUI's node-output caching
  means that once a prompt's conditioning is computed, re-queues never
  touch Gemma again until the prompt text changes.
- bf16 Gemma is a plain nn.Module → wrapped in a ModelPatcher and fully
  managed by comfy (offload/eviction).
- 4-bit Gemma (bitsandbytes) cannot be moved between devices (transformers
  forbids .to() on bnb-quantized models), so it is loaded straight to the
  GPU and freed outright instead of offloaded. `mm.free_memory` is called
  first so comfy evicts whatever is needed to make room.
- the embeddings processor (audio connector, from the audio-components
  checkpoint) is small, stripped of its video branch exactly the way
  upstream's audio_only mode does it, and ModelPatcher-managed.
"""
import gc
import logging

import torch

from ..vendor import mdb_ltx_core  # noqa: F401  (sys.path bootstrap)
from mdb_ltx_core.loader import DummyRegistry
from mdb_ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from mdb_ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from mdb_ltx_core.utils import find_matching_file

import comfy.model_management as mm

from . import patcher as patching

log = logging.getLogger("ComfyUI-DramaBox")

GEMMA_4BIT_VRAM_BYTES = 9 * 1024**3   # ~8 GB weights + headroom


def _strip_video_branch(ep: torch.nn.Module) -> torch.nn.Module:
    """Remove the embeddings processor's video branch (audio-only mode).

    Must run BEFORE any .to(device): the audio-components checkpoint has no
    video_aggregate_embed / video_connector weights, so those submodules sit
    on the meta device and .to() would raise "cannot copy out of meta".
    Port of the audio_only surgery in upstream's PromptEncoder.__init__.
    """
    freed = 0
    if getattr(ep, "video_connector", None) is not None:
        try:
            freed += sum(p.numel() * p.element_size() for p in ep.video_connector.parameters() if not p.is_meta)
        except Exception:
            pass
        del ep.video_connector
        ep.video_connector = None

    fe = ep.feature_extractor
    if getattr(fe, "video_aggregate_embed", None) is not None:
        out_features = fe.video_aggregate_embed.out_features
        del fe.video_aggregate_embed

        class _DummyVideoEmbed(torch.nn.Module):
            def __init__(self, out_f):
                super().__init__()
                self.out_features = out_f

            def forward(self, x):
                return torch.zeros(x.shape[0], x.shape[1], self.out_features,
                                   device=x.device, dtype=x.dtype)

        fe.video_aggregate_embed = _DummyVideoEmbed(out_features)

    _orig_create = ep.create_embeddings

    def _audio_only_create(video_features, audio_features, additive_attention_mask, _ep=ep):
        m = additive_attention_mask
        while m.dim() > 2:
            m = m[:, 0]
        binary_mask = (m >= -1.0).to(torch.int64)
        audio_encoded = None
        if _ep.audio_connector is not None:
            audio_encoded, _ = _ep.audio_connector(audio_features, additive_attention_mask)
        return video_features, audio_encoded, binary_mask

    ep.create_embeddings = _audio_only_create
    if freed:
        log.info(f"[DramaBox] Embeddings processor: video branch stripped ({freed / 1e9:.1f} GB skipped)")
    return ep


class DramaBoxTextEncoder:
    """Lazy handle for the Gemma encoder + embeddings processor.

    Created cheaply by the loader node; heavy weights are built on the
    first `encode()` call.
    """

    def __init__(self, gemma_root: str, components_path: str, quantization: str,
                 dtype: torch.dtype, keep_loaded: bool):
        if quantization not in ("bnb_4bit", "none"):
            raise ValueError(f"Unknown text-encoder quantization: {quantization}")
        self.gemma_root = gemma_root
        self.components_path = components_path
        self.quantization = quantization
        self.dtype = dtype
        self.keep_loaded = keep_loaded

        self._gemma = None            # 4-bit path: GemmaTextEncoder on GPU
        self._gemma_patcher = None    # bf16 path: ModelPatcher
        self._ep_patcher = None       # embeddings processor ModelPatcher

    # -- builders ----------------------------------------------------------

    def _build_embeddings_processor(self):
        if self._ep_patcher is not None:
            return
        builder = SingleGPUModelBuilder(
            model_path=self.components_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=DummyRegistry(),
        )
        ep = builder.build(device=mm.unet_offload_device(), dtype=self.dtype)
        ep = _strip_video_branch(ep).eval()
        meta_left = [n for n, p in ep.named_parameters() if p.is_meta]
        if meta_left:
            raise RuntimeError(
                f"[DramaBox] Embeddings processor is missing weights ({meta_left[:5]}...) — "
                f"{self.components_path} doesn't look like dramabox-audio-components.safetensors."
            )
        self._ep_patcher = patching.make_patcher(ep)

    def _build_gemma_4bit(self):
        import json
        import os
        from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration
        from mdb_ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
        from mdb_ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer

        device = mm.get_torch_device()
        # Ask comfy to clear room before transformers allocates outside its view.
        mm.free_memory(GEMMA_4BIT_VRAM_BYTES, device)

        prequantized = False
        cfg_path = os.path.join(self.gemma_root, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    prequantized = "quantization_config" in json.load(f)
            except Exception:
                pass

        from_kwargs = {"device_map": str(device), "torch_dtype": self.dtype}
        if prequantized:
            log.info("[DramaBox] Loading pre-quantized Gemma (checkpoint's own bnb-4bit config)")
        else:
            log.info("[DramaBox] Loading Gemma with runtime bitsandbytes nf4 quantization")
            from_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
            )
        hf_model = Gemma3ForConditionalGeneration.from_pretrained(self.gemma_root, **from_kwargs)
        tokenizer = LTXVGemmaTokenizer(
            str(find_matching_file(self.gemma_root, "tokenizer.model").parent), 1024
        )
        self._gemma = GemmaTextEncoder(model=hf_model, tokenizer=tokenizer, dtype=self.dtype)
        log.info(f"[DramaBox] Gemma 4-bit loaded. {patching.vram_report()}")

    def _build_gemma_bf16(self):
        module_ops = module_ops_from_gemma_root(self.gemma_root)
        model_folder = find_matching_file(self.gemma_root, "model*.safetensors").parent
        weight_paths = tuple(str(p) for p in model_folder.rglob("*.safetensors"))
        builder = SingleGPUModelBuilder(
            model_path=weight_paths,
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GEMMA_LLM_KEY_OPS,
            module_ops=(GEMMA_MODEL_OPS, *module_ops),
            registry=DummyRegistry(),
        )
        encoder = builder.build(device=mm.unet_offload_device(), dtype=self.dtype).eval()
        self._gemma_patcher = patching.make_patcher(encoder)
        log.info("[DramaBox] Gemma bf16 built on offload device (ModelPatcher-managed)")

    # -- public API --------------------------------------------------------

    @torch.inference_mode()
    def encode(self, prompts: list) -> list:
        """Encode prompts → list of audio-conditioning tensors (one per prompt)."""
        self._build_embeddings_processor()

        if self.quantization == "bnb_4bit":
            if self._gemma is None:
                self._build_gemma_4bit()
            patching.load_patchers([self._ep_patcher])
            encoder = self._gemma
        else:
            if self._gemma_patcher is None:
                self._build_gemma_bf16()
            patching.load_patchers([self._gemma_patcher, self._ep_patcher])
            encoder = self._gemma_patcher.model

        ep = self._ep_patcher.model
        encodings = []
        for prompt in prompts:
            hidden_states, mask = encoder.encode(prompt)
            out = ep.process_hidden_states(hidden_states, mask)
            if out.audio_encoding is None:
                raise RuntimeError("[DramaBox] Embeddings processor returned no audio encoding "
                                   "(audio connector missing from checkpoint?)")
            encodings.append(out.audio_encoding)
            del hidden_states

        if not self.keep_loaded:
            self.free()
        return encodings

    def free(self):
        """Release the Gemma LLM entirely (used by low-VRAM policy and the
        unload node). The embeddings processor stays patcher-managed."""
        if self._gemma is not None:
            self._gemma = None
        self._gemma_patcher = None
        gc.collect()
        mm.soft_empty_cache()
        log.info(f"[DramaBox] Text encoder freed. {patching.vram_report()}")
