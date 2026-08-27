# Conversion scripts

Turn an LTX-2.3 finetune (or the DramaBox DiT) into a small, GGUF-quantized
audio DiT that loads in ComfyUI-MelodramaBox's **DramaBox DiT Loader**.

These run standalone — no ComfyUI import needed. Requirements:

```bash
pip install safetensors numpy torch gguf-connector
```

## What GGUF buys you (measured, RTX 3090)

| DiT | file | peak VRAM (generation) |
|---|---|---|
| bf16 | 6.6 GB | 16.6 GB |
| **Q8_0** (near-lossless) | **3.5 GB** | **13.5 GB** |
| Q5_0 | ~2.3 GB | lower |
| Q4_0 | ~1.9 GB | lower |

Q8_0 is the recommended default for speech. The pure-Python quantizer covers
the legacy GGML types (Q8_0/Q5_0/Q4_0/Q5_1/Q4_1). The K-quants
(Q4_K/Q5_K/Q6_K) need llama.cpp's `llama-quantize` — convert to `F16` here,
then quantize with llama.cpp (the loader reads K-quants fine, it just can't
*produce* them in pure Python).

## 1. `dit_to_gguf.py` — audio-only DiT → GGUF

For DramaBox's own DiT, or any audio-only DiT you already have:

```bash
python dit_to_gguf.py \
    --src  .../models/diffusion_models/dramabox-dit-v1.safetensors \
    --dst  .../models/diffusion_models/dramabox-dit-v1-Q8_0.gguf \
    --quant Q8_0
```

## 2. `ltx_finetune_to_gguf.py` — full LTX-2.3 finetune → audio-only GGUF

Extracts the audio branch from a **full** LTX-2.3 finetune (fp8 or bf16),
then GGUF-quantizes it. From a HuggingFace URL only the ~3.6 GB audio branch
is fetched (HTTP range reads), not the whole 20–40 GB file.

```bash
python ltx_finetune_to_gguf.py \
    --src        https://huggingface.co/<user>/<repo>/resolve/main/<finetune>.safetensors \
    --reference  .../models/diffusion_models/dramabox-dit-v1.safetensors \
    --name       my_finetune \
    --quant      Q8_0 \
    --out-dir    .../models/diffusion_models
```

- `--reference` is DramaBox's DiT (`dramabox-dit-v1.safetensors`); its tensor
  names define exactly which tensors to extract. You need it locally.
- `--extract-only` stops after the bf16 audio-only `.safetensors`.
- `--keep-safetensors` keeps that intermediate file (deleted by default).

Extraction is **self-validating**: it keeps precisely the reference DiT's
tensor names and hard-fails if the source is missing any, so a success means
the result will load.

## ⚠️ Read before extracting a full finetune

The audio-only forward runs **without** the audio↔video cross-attention the
full model was trained with. A **video** finetune's audio branch, extracted
this way, typically produces continuous/tonal **non-speech** rather than
clean TTS — tested with `TenStrip/LTX2.3-10Eros`: it extracts and generates
cleanly, but the audio is not usable speech. **Loadability is guaranteed;
usable speech is not — listen to a sample first.**

The reliable target for better/different voices is a finetune or **LoRA of
the audio-only base** (DramaBox's own lineage), which loads directly (LoRA
via the DramaBox LoRA Loader; a full audio-only finetune via the DiT Loader),
no extraction needed.
