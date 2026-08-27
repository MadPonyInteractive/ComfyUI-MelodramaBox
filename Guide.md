# ComfyUI-MelodramaBox — Guide

Everyday usage: install, the nodes, voice cloning, quality controls, and LoRA
loading. For architecture, VRAM tuning, GGUF, finetune extraction, and
troubleshooting, see **[Advanced.md](Advanced.md)**.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-MelodramaBox
pip install -r ComfyUI-MelodramaBox/requirements.txt
```

No runtime `git clone` and no `git` executable needed. Weights download
automatically the first time a loader runs, into ComfyUI's standard model
folders:

| Component | Size | Folder | Repo |
|---|---|---|---|
| Audio DiT | ~6.6 GB | `models/diffusion_models/` | `ResembleAI/Dramabox` |
| Audio components (VAE + vocoder + connector) | ~1.9 GB | `models/vae/` | `ResembleAI/Dramabox` |
| Gemma-3 12B text encoder | ~8 GB (4-bit) | `models/text_encoders/` | `unsloth/gemma-3-12b-it-bnb-4bit` |

A v1 install's Hugging Face cache copies are migrated into these folders
automatically rather than re-downloaded. `DramaBox Download Models`
pre-fetches everything ahead of a run.

`bitsandbytes` (for the recommended 4-bit Gemma) is an optional dependency;
without it, choose the bf16 text encoder (~24 GB, HF-gated) in the loader.

## Node overview

**Loaders** (`DramaBox/loaders`)
- `DramaBox DiT Loader` — loads the audio DiT (`.safetensors` or `.gguf`) into a ModelPatcher.
- `DramaBox Extract Audio DiT (from full LTX-2.3)` — extracts a standalone audio-only DiT from a full LTX-2.3 finetune (see [Advanced → Using other LTX-2.3 finetunes](Advanced.md#using-other-ltx-23-finetunes)).
- `DramaBox Text Encoder Loader (Gemma-3)` — Gemma-3 12B + the audio embeddings connector. Pick 4-bit (~8 GB, recommended) or bf16 (~24 GB, HF-gated). `keep_loaded=False` frees the LLM after each encode for 12 GB GPUs.
- `DramaBox Audio VAE Loader` — audio VAE (encoder/decoder) + BigVGAN vocoder; 48 kHz output.
- `DramaBox LoRA Loader` — applies a DramaBox-format LoRA to the DiT via ComfyUI's native patch system (`add_patches` on a cloned patcher), so it composes and unapplies like any core LoRA loader.
- `DramaBox Zero-Shot Voice Reference` / `(Batch)` — see [Zero-shot voice cloning](#zero-shot-voice-cloning).

**Conditioning** (`DramaBox/conditioning`)
- `DramaBox Text Encode` — runs Gemma + connector to produce DiT conditioning. Cached by ComfyUI, so Gemma only runs when the prompt text changes.
- `DramaBox Text Encode (Long-Form)` — splits a long prompt into chunks (sentence-aware) and encodes each; the sampler generates them with the same voice reference and the decoder crossfades the joins.

**Sampling** (`DramaBox/sampling`)
- `DramaBox Sampler` — flow-matching loop with progress + interrupt. Outputs audio latents.
- `DramaBox VAE Decode (Audio)` — decodes latents to a 48 kHz `AUDIO` waveform, stitching long-form chunks.

**Output** (`DramaBox/output`)
- `DramaBox Save Audio` — writes .wav/.flac via soundfile.
- `DramaBox Apply Watermark (Perth)` — applies the Resemble Perth imperceptible watermark.
- `DramaBox Watermark Detector` — reports Perth detection confidence.

**VRAM** (`DramaBox/vram`)
- `DramaBox Unload Models` — force-frees all comfy-managed models (and the 4-bit Gemma, which comfy can't evict on its own) at a chosen point in the graph.

**Utils** (`DramaBox/utils`)
- `DramaBox Estimate Duration` — prompt-length estimate, for planning long-form chunking.

All audio in/out uses ComfyUI's native `AUDIO` type, so DramaBox nodes connect
directly to `LoadAudio`, `SaveAudio`, and other audio-native custom nodes.

## Prompt format

Text inside quotes is spoken; text outside quotes is a stage direction the
model performs but doesn't read aloud:

```
A woman speaks warmly, "Hello, how are you today?" She sighs. "It's been a long week."
```

Onomatopoeia inside quotes (e.g. `"Hahaha"`, `"Mmmmm"`) is vocalized.

## Zero-shot voice cloning

`DramaBox Zero-Shot Voice Reference` accepts a ComfyUI `AUDIO` input or a file
path (or both, stitched together), trims silence, resamples, and encodes the
clip to a VAE latent. It takes the `DRAMABOX_AUDIO_VAE` as an input (the encoder
lives there) and outputs a `DRAMABOX_VOICE_REF` you wire into the sampler.

- Chain multiple short clips of the same speaker (`extra_ref` input, or the
  `Batch` node) to combine them into one reference that clears the 10 s
  recommendation.
- Omit `voice_ref` entirely to let the prompt's speaker description alone
  control the voice.
- `denoise` (on the reference node) runs RE-USE/SEMamba denoising on the
  reference — a clean speaker anchor without eating generated laughs/breaths.
  **RE-USE is licensed NSCLv1 (non-commercial)**, so turn it off for commercial
  work. It degrades gracefully if its optional deps are missing; see
  [Advanced → Windows install for RE-USE CUDA kernels](Advanced.md#windows-install-for-re-use-cuda-kernels).

## Quality controls

Exposed on `DramaBox Sampler`:

| Param | Default | Notes |
|---|---|---|
| `quality_preset` | default | `draft`/`default`/`high` set steps + cfg + stg together; `custom` uses the widgets |
| `steps` | 30 | Denoising steps (12 draft / 30 default / 50 high) |
| `cfg_scale` | 2.5 | Lower = more natural, higher = more text-faithful |
| `stg_scale` | 1.5 | Spatio-temporal guidance |
| `duration_seconds` | 0 | 0 = estimate from prompt; >0 forces a length (single-chunk only) |
| `duration_multiplier` | 1.1 | Breathing room over the estimated length |
| `cfg_rescale` | -1 | -1 = automatic cfg-aware anti-clipping schedule; 0 disables; 0–1 fixes it |
| `seed` | 42 | `control_after_generate` enabled |

`dtype` (bf16/fp16) is chosen on the DiT and VAE loaders; text-encoder precision
(4-bit vs bf16) is chosen on the text encoder loader.

## LoRA (loading only)

This pack **loads** DramaBox-format LoRAs (trained voice/style adapters) but
does **not train** them. `DramaBox LoRA Loader` picks a `.safetensors` LoRA from
`models/loras` (or an absolute path) and applies it to the DiT via ComfyUI's
native `add_patches`, composing with the safetensors, GGUF, and extracted DiTs
alike.

To train your own adapter, use Resemble's upstream
[DramaBox](https://github.com/resemble-ai/DramaBox) training pipeline
(`src/preprocess.py` + `accelerate launch src/train.py`) directly, then drop the
resulting `.safetensors` into `models/loras`. The most reusable target is a LoRA
on the **audio-only base** (DramaBox's lineage), which loads here with no
extraction.
