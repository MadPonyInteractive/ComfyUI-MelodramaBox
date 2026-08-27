"""Component loader nodes.

v2.0: the monolithic TTSServer wrapper is gone. Each model component loads
from ComfyUI's standard folders into its own ModelPatcher:

    DramaBoxDiTLoader          models/diffusion_models -> DRAMABOX_DIT
    DramaBoxTextEncoderLoader  models/text_encoders    -> DRAMABOX_TEXTENC
    DramaBoxAudioVAELoader     models/vae              -> DRAMABOX_AUDIO_VAE

ComfyUI's memory manager owns all loading/offloading/eviction; the
dropdowns offer a "(download)" entry when a known DramaBox file isn't in
the folder yet.
"""

# MODIFIED by Mad Pony Interactive (fork of comfyui-melodramabox 2.1.0).
# Changes are marked `MPI-607` below and described in the repository NOTICE file.
import logging
import os

import torch

from . import config
from . import downloader

log = logging.getLogger("ComfyUI-DramaBox")

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _gguf_in_roots(folder_name):
    """MPI-607: .gguf files under any registered `folder_name` root.

    ComfyUI's supported_pt_extensions has no '.gguf', so get_filename_list can
    never return one. Scanned here instead of adding '.gguf' to that global set,
    which every other installed pack shares.
    """
    import os
    try:
        import folder_paths
        roots = folder_paths.get_folder_paths(folder_name)
    except Exception:
        return []
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, names in os.walk(root):
            for n in names:
                if n.endswith(".gguf"):
                    rel = os.path.relpath(os.path.join(dirpath, n), root)
                    out.append(rel.replace(os.sep, "/"))
    return sorted(set(out))


def _folder_choices(folder_name: str, known_filename: str, fallback_dir: str):
    """Files in a ComfyUI model folder, plus a download entry for
    `known_filename` when it isn't present yet."""
    try:
        import folder_paths
        files = [f for f in folder_paths.get_filename_list(folder_name)
                 if f.endswith((".safetensors", ".sft", ".gguf"))]
        files = sorted(set(files) | set(_gguf_in_roots(folder_name)))
    except Exception:
        files = [f for f in os.listdir(fallback_dir)
                 if f.endswith((".safetensors", ".gguf"))] if os.path.isdir(fallback_dir) else []
    if known_filename not in files:
        files = [f"{known_filename} (download)"] + files
    return files


def _resolve_model_file(folder_name: str, name: str, known_filename: str,
                        download_fn, fallback_dir: str) -> str:
    if name == f"{known_filename} (download)":
        return download_fn()
    try:
        import folder_paths
        path = folder_paths.get_full_path(folder_name, name)
    except Exception:
        path = os.path.join(fallback_dir, name)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"[DramaBox] Model file not found: {name} in {folder_name}")
    return path


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------
class DramaBoxDiTLoader:
    """Loads the DramaBox audio-only DiT (the separated + fine-tuned
    LTX-2.3 audio branch, 3.3B params) into a ModelPatcher. Built on the
    offload device; comfy moves it to GPU when sampling needs it and evicts
    it under memory pressure like any other diffusion model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_folder_choices("diffusion_models", config.DIT_FILENAME, config.DIT_DIR),),
                "dtype": (["bf16", "fp16"], {"default": "bf16",
                    "tooltip": "bf16 matches the published checkpoint. fp16 saves nothing "
                               "(same size) and risks overflow; use only if your GPU lacks bf16."}),
            },
        }

    RETURN_TYPES = ("DRAMABOX_DIT",)
    RETURN_NAMES = ("dit",)
    FUNCTION = "load"
    CATEGORY = "DramaBox/loaders"

    def load(self, model_name, dtype):
        path = _resolve_model_file("diffusion_models", model_name, config.DIT_FILENAME,
                                   downloader.ensure_dit, config.DIT_DIR)
        if path.endswith(".gguf"):
            from .gguf.loader import load_dit_gguf
            patcher, ckpt_config = load_dit_gguf(path, _DTYPES[dtype])
        else:
            from .model.dit import load_dit
            patcher, ckpt_config = load_dit(path, _DTYPES[dtype])
        return ({"patcher": patcher, "config": ckpt_config, "path": path},)


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------
class DramaBoxTextEncoderLoader:
    """Gemma-3-12B text encoder + audio embeddings connector.

    The Gemma LLM is the biggest VRAM consumer in the pipeline, and thanks
    to ComfyUI's node-output caching it only runs when the prompt text
    changes — so this loader returns a lazy handle; weights load on first
    encode.

    `keep_loaded=False` frees Gemma entirely after each encode (for 12 GB
    GPUs); on a 24 GB card the default True keeps it warm for instant
    prompt tweaks."""

    _VARIANTS = {
        "gemma-3-12b-it 4-bit (~8 GB, recommended)": (
            config.TEXT_ENCODER_4BIT_REPO, config.TEXT_ENCODER_4BIT_DIRNAME, "bnb_4bit"),
        "gemma-3-12b-it bf16 (~24 GB, HF-gated)": (
            config.TEXT_ENCODER_BF16_REPO, config.TEXT_ENCODER_BF16_DIRNAME, "none"),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": (list(cls._VARIANTS.keys()),),
                "components_name": (_folder_choices("vae", config.AUDIO_COMPONENTS_FILENAME, config.VAE_DIR),
                    {"tooltip": "dramabox-audio-components.safetensors — also holds the "
                                "audio embeddings connector this encoder needs."}),
                "keep_loaded": ("BOOLEAN", {"default": True,
                    "tooltip": "Off: free the ~8 GB Gemma LLM right after each prompt encode "
                               "(slower prompt changes, much lower VRAM floor — for 12 GB GPUs)."}),
            },
        }

    RETURN_TYPES = ("DRAMABOX_TEXTENC",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load"
    CATEGORY = "DramaBox/loaders"

    def load(self, text_encoder, components_name, keep_loaded):
        repo_id, dirname, quantization = self._VARIANTS[text_encoder]
        gemma_root = downloader.ensure_text_encoder(repo_id, dirname)
        components_path = _resolve_model_file("vae", components_name, config.AUDIO_COMPONENTS_FILENAME,
                                              downloader.ensure_audio_components, config.VAE_DIR)
        from .model.text_encoder import DramaBoxTextEncoder
        handle = DramaBoxTextEncoder(
            gemma_root=gemma_root,
            components_path=components_path,
            quantization=quantization,
            dtype=torch.bfloat16,
            keep_loaded=keep_loaded,
        )
        return (handle,)


# ---------------------------------------------------------------------------
# Audio VAE
# ---------------------------------------------------------------------------
class DramaBoxAudioVAELoader:
    """Audio VAE (encoder/decoder) + BigVGAN vocoder from
    dramabox-audio-components.safetensors, each in its own ModelPatcher.
    Output is 48 kHz (the vocoder's bandwidth extension)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components_name": (_folder_choices("vae", config.AUDIO_COMPONENTS_FILENAME, config.VAE_DIR),),
                "dtype": (["bf16", "fp16"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("DRAMABOX_AUDIO_VAE",)
    RETURN_NAMES = ("audio_vae",)
    FUNCTION = "load"
    CATEGORY = "DramaBox/loaders"

    def load(self, components_name, dtype):
        path = _resolve_model_file("vae", components_name, config.AUDIO_COMPONENTS_FILENAME,
                                   downloader.ensure_audio_components, config.VAE_DIR)
        from .model.vae import DramaBoxAudioVAE
        return (DramaBoxAudioVAE(path, _DTYPES[dtype]),)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class DramaBoxLoRALoader:
    """Applies a DramaBox-trained LoRA to the DiT via ComfyUI's native
    patch system (add_patches on a cloned ModelPatcher) — patches compose
    like any core LoRA loader and unapply cleanly on unload. Reads from
    ComfyUI's standard models/loras folder."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            lora_files = folder_paths.get_filename_list("loras")
        except Exception:
            lora_files = []
        return {
            "required": {
                "dit": ("DRAMABOX_DIT",),
                "lora_name": (lora_files if lora_files else ["<none found in models/loras>"],),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "lora_path_override": ("STRING", {"default": "",
                    "tooltip": "Absolute path to a LoRA outside models/loras; overrides lora_name"}),
            },
        }

    RETURN_TYPES = ("DRAMABOX_DIT",)
    RETURN_NAMES = ("dit",)
    FUNCTION = "apply"
    CATEGORY = "DramaBox/loaders"

    def apply(self, dit, lora_name, strength, lora_path_override=""):
        if lora_path_override:
            lora_path = lora_path_override
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA not found at {lora_path} (lora_path_override).")
        else:
            try:
                import folder_paths
                lora_path = folder_paths.get_full_path("loras", lora_name)
            except Exception:
                lora_path = None
            if lora_path is None or not os.path.exists(lora_path):
                raise FileNotFoundError(
                    f"LoRA '{lora_name}' not found in ComfyUI's models/loras folder. Drop a "
                    f".safetensors LoRA there, or use lora_path_override."
                )
        if strength == 0.0:
            return (dit,)
        from .model.lora import apply_lora_to_dit
        out = dict(dit)
        out["patcher"] = apply_lora_to_dit(dit["patcher"], lora_path, strength)
        return (out,)


# ---------------------------------------------------------------------------
# Voice reference — zero-shot cloning
# ---------------------------------------------------------------------------
def _load_audio_file(path: str):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    return torch.from_numpy(data).T.contiguous(), sr           # (C, T)


def _encode_reference(clips, audio_vae, trim_silence, max_duration_sec,
                      denoise, ref_seconds, strength):
    from . import audio_prep
    from . import sampling

    ref = audio_prep.prepare_zero_shot_reference(
        clips, trim_silence=trim_silence, max_duration_sec=max_duration_sec)
    waveform, sr = ref["waveform"], ref["sample_rate"]

    if denoise:
        waveform = audio_prep.denoise_reference(waveform, sr)

    prepared = sampling.prepare_voice_ref_waveform(waveform, sr, ref_seconds)
    latent = audio_vae.encode(prepared, sr)
    log.info(f"[DramaBox] Voice reference encoded: {ref['duration']:.1f}s from "
             f"{ref['num_source_clips']} clip(s) -> latent {tuple(latent.shape)}")
    return {
        "latent": latent.cpu(),
        "strength": strength,
        "waveform": ref["waveform"],   # kept for chaining into another ref node
        "sample_rate": sr,
        "duration": ref["duration"],
    }


class DramaBoxVoiceReferenceLoader:
    """Zero-shot voice cloning: turns ~10+ s of reference audio into a VAE
    latent the sampler conditions on (tensor-native — v1's temp-WAV
    round-trip is gone). Accepts a ComfyUI AUDIO input, a file path, or
    both; chain another reference node into `extra_ref` to stitch clips.

    `denoise` runs NVIDIA RE-USE on the reference (not the output) so the
    model gets a clean speaker anchor without suppressing generated laughs/
    breaths. RE-USE is NSCLv1-licensed (non-commercial) and needs the
    optional mamba-ssm deps; it degrades gracefully if unavailable."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_vae": ("DRAMABOX_AUDIO_VAE",),
                "trim_silence": ("BOOLEAN", {"default": True}),
                "max_duration_sec": ("FLOAT", {"default": config.ZERO_SHOT_MAX_USEFUL_SEC,
                                               "min": 1.0, "max": 120.0}),
                "denoise": ("BOOLEAN", {"default": True,
                    "tooltip": "Clean the reference with NVIDIA RE-USE before encoding. "
                               "RE-USE is non-commercial (NSCLv1) — turn off for commercial work."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "1.0 = reference fully frozen conditioning (upstream default)."}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "file_path": ("STRING", {"default": ""}),
                "extra_ref": ("DRAMABOX_VOICE_REF", {"tooltip": "Chain another reference node "
                              "here to stitch multiple clips into one"}),
            },
        }

    RETURN_TYPES = ("DRAMABOX_VOICE_REF",)
    RETURN_NAMES = ("voice_ref",)
    FUNCTION = "load"
    CATEGORY = "DramaBox/conditioning"

    def load(self, audio_vae, trim_silence, max_duration_sec, denoise, strength,
             audio=None, file_path="", extra_ref=None):
        clips = []
        if extra_ref is not None:
            clips.append((extra_ref["waveform"], extra_ref["sample_rate"]))
        if audio is not None:
            waveform, sr = audio["waveform"], audio["sample_rate"]
            if waveform.ndim == 3:
                waveform = waveform[0]
            clips.append((waveform, sr))
        if file_path:
            resolved = file_path if os.path.isabs(file_path) else os.path.join(config.VOICE_REF_DIR, file_path)
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"Voice reference file not found: {resolved}")
            clips.append(_load_audio_file(resolved))

        if not clips:
            raise ValueError("[DramaBox] Connect an AUDIO input, set file_path, or chain an "
                             "extra_ref — no reference audio was provided.")

        return (_encode_reference(clips, audio_vae, trim_silence, max_duration_sec,
                                  denoise, config.VOICE_REF_ENCODE_SECONDS, strength),)


class DramaBoxVoiceReferenceBatch:
    """Stitches a list of AUDIO inputs into a single zero-shot reference.
    See DramaBoxVoiceReferenceLoader."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_vae": ("DRAMABOX_AUDIO_VAE",),
                "audio_list": ("AUDIO",),
                "trim_silence": ("BOOLEAN", {"default": True}),
                "max_duration_sec": ("FLOAT", {"default": config.ZERO_SHOT_MAX_USEFUL_SEC,
                                               "min": 1.0, "max": 120.0}),
                "denoise": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("DRAMABOX_VOICE_REF",)
    RETURN_NAMES = ("voice_ref",)
    FUNCTION = "load"
    CATEGORY = "DramaBox/conditioning"

    def load(self, audio_vae, audio_list, trim_silence, max_duration_sec, denoise, strength):
        def _scalar(v):
            return v[0] if isinstance(v, list) else v
        audio_vae = _scalar(audio_vae)
        trim_silence, max_duration_sec = _scalar(trim_silence), _scalar(max_duration_sec)
        denoise, strength = _scalar(denoise), _scalar(strength)

        clips = []
        for audio in audio_list:
            waveform, sr = audio["waveform"], audio["sample_rate"]
            if waveform.ndim == 3:
                waveform = waveform[0]
            clips.append((waveform, sr))

        return (_encode_reference(clips, audio_vae, trim_silence, max_duration_sec,
                                  denoise, config.VOICE_REF_ENCODE_SECONDS, strength),)
