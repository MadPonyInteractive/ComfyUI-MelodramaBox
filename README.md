# ComfyUI-MelodramaBox

> ### Why this fork exists
>
> This is a **fork**, maintained by [Mad Pony Interactive](https://github.com/MadPonyInteractive)
> for use in [Cubric Vision](https://github.com/MadPonyInteractive). It is based on
> `comfyui-melodramabox` **2.1.0** from the [ComfyUI Registry](https://registry.comfy.org/nodes/comfyui-melodramabox).
>
> The registry release names `github.com/doggeddalle/ComfyUI-MelodramaBox` as its
> source repository, but **that URL 404s** — there is no upstream repository to open
> a pull request against. Cubric Vision installs its ComfyUI nodes by pinned git
> commit, and it needs three bug fixes that the published zip does not have, so the
> only way to ship them was to publish a fork.
>
> The three fixes: the shared text-encoder root is honoured instead of only the first
> one (otherwise ~8 GB of Gemma-3 is re-downloaded over a copy that is already there),
> the documented `.gguf` DiT path is actually selectable, and the unload node stops
> being cached out — it exists purely for its side effect, so a cached run silently
> did nothing. Each is marked in place with an `MPI-607` comment and described in
> [NOTICE](NOTICE).
>
> **All credit for the pack is doggeddalle's.** The Apache-2.0 licence is unchanged.
> If the upstream repository reappears, these fixes are offered back gladly.


Expressive, prompt-driven **text-to-speech with zero-shot voice cloning** for
ComfyUI — a native implementation of [Resemble AI's DramaBox](https://github.com/resemble-ai/DramaBox)
(the **LTX-2.3 3.3B audio-only** DiT by Lightricks). **Audio only.**

Write what you want said and how it's performed — the prompt controls speaker
identity, emotion, delivery, laughs, sighs, and pauses. Add a ~10-second
reference clip to clone a target voice, with no training needed.

```
A woman speaks warmly, "Hello — it's good to finally meet you." She laughs softly.
```

## Why this pack

- **Native ComfyUI, not a wrapper.** Each model component (DiT, Gemma-3 text
  encoder, audio VAE + vocoder) is a real `ModelPatcher`, so ComfyUI manages
  VRAM, offloading, and eviction for you — with a live progress bar and Cancel.
- **Tuned for real GPUs.** Runs comfortably on a 24 GB card; **12 GB** cards
  work via offload, and lower via GGUF.
- **GGUF quantization** of the audio DiT — smaller files, less VRAM ([details](Advanced.md#gguf)).
- **Zero-shot voice cloning** from a short reference clip ([guide](Guide.md#zero-shot-voice-cloning)).
- **48 kHz** output, long-form generation, and Perth watermarking.
- **Use other LTX-2.3 finetunes** by extracting their audio branch ([details](Advanced.md#using-other-ltx-23-finetunes)).

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-MelodramaBox
pip install -r ComfyUI-MelodramaBox/requirements.txt
```

Restart ComfyUI. Model weights (~16 GB total) download automatically the first
time you run a loader. Full details, including the model list and the bf16
alternative to `bitsandbytes`, are in the **[Guide](Guide.md#install)**.

## The graph

```
DiT Loader ─────────────────┐   (audio DiT; .safetensors or .gguf)
Text Encoder Loader → Text Encode ─┤   (Gemma-3 12B; 4-bit or bf16)
Audio VAE Loader ─┬─ Voice Reference ─┤   (optional cloning clip)
                  │                    ▼
                  │             Sampler → VAE Decode → Save Audio
                  └───────────────────────┘
```

Start from a ready-made workflow in [`example_workflows/`](example_workflows):
`dramabox_standard_inference.json` (24 GB), `dramabox_lowvram_12gb.json` (12 GB),
or `dramabox_advanced_inference.json` (long-form + LoRA + voice ref).

## VRAM at a glance

| Setup | Peak VRAM | Card |
|---|---|---|
| bf16 DiT + 4-bit Gemma, all warm | ~16.6 GB | 24 GB (3090) |
| Same, text encoder freed per encode | fits 12 GB | 3060 / 4070 |
| **Q8_0 GGUF DiT** + 4-bit Gemma | **~13.5 GB** | 16 GB |
| Q5_0 / Q4_0 GGUF DiT | lower | 8–12 GB |

Measured on an RTX 3090. See **[Advanced → VRAM management](Advanced.md#vram-management)**.

## Documentation

- **[Guide.md](Guide.md)** — install, every node, voice cloning, quality
  controls, LoRA loading.
- **[Advanced.md](Advanced.md)** — architecture, VRAM tuning, GGUF, using other
  LTX-2.3 finetunes, troubleshooting, Windows RE-USE setup.
- **[conversion_scripts/](conversion_scripts)** — standalone tools to turn any
  LTX-2.3 finetune into a small GGUF audio DiT.

## License

This pack's own code is **Apache 2.0** (see [`LICENSE`](LICENSE)). The DramaBox
model and the vendored LTX-2 inference code are under the **LTX-2 Community
License**; optional RE-USE reference denoising is **NSCLv1 (non-commercial)**.
See [Advanced → License](Advanced.md#license) for the full breakdown.
