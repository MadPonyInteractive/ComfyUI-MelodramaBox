# ComfyUI-MelodramaBox — Advanced

Architecture, VRAM tuning, GGUF quantization, using other LTX-2.3 finetunes,
troubleshooting, the Windows RE-USE build, and licensing. For basic usage see
the **[Guide](Guide.md)**.

## Contents

- [Architecture](#architecture)
- [VRAM management](#vram-management)
- [GGUF](#gguf)
- [Using other LTX-2.3 finetunes](#using-other-ltx-23-finetunes)
- [Notes and troubleshooting](#notes-and-troubleshooting)
- [Windows install for RE-USE CUDA kernels](#windows-install-for-re-use-cuda-kernels)
- [License](#license)

## Architecture

v2 is a ground-up rework. Earlier versions wrapped upstream's monolithic
`TTSServer` by `git clone`-ing resemble-ai/DramaBox at runtime and driving it
through file paths — which meant ComfyUI's memory manager never saw the model,
"low VRAM" mode destroyed and rebuilt the whole ~16 GB server per generation,
audio round-tripped through temp WAVs, and there was no progress bar or
interrupt.

DramaBox is an IC-LoRA fine-tune of the **LTX-2.3 3.3B audio-only DiT** — i.e.
Lightricks' already-separated audio branch. v2 loads that architecture
directly. The inference model code (the DiT, audio VAE, BigVGAN vocoder, Gemma
text-encoder wrappers, flow-matching sampler) is **vendored** at a pinned
upstream commit under `dramabox_nodes/vendor/` (see `vendor/ATTRIBUTION.md`), so
there is no runtime `git clone`, no top-level `src` namespace collision, and no
on-disk monkeypatching.

Each model component loads separately into its own
`comfy.model_patcher.ModelPatcher`, registered with `comfy.model_management`.
ComfyUI itself owns loading, offloading, and eviction — so DramaBox models get
evicted automatically when an image or video workflow needs the VRAM,
`--lowvram` / `--reserve-vram` work, and generation is tensor-native end to end
(no temp files). Sampling has a real `ProgressBar` and responds to Cancel within
one step.

## VRAM management

Every component is a `ModelPatcher` registered with ComfyUI's memory manager, so
most VRAM handling is automatic: components load to GPU only when a step needs
them, and comfy evicts them under pressure (e.g. when the next workflow is an
image model).

**RTX 3090 / 24 GB (comfortable).** Leave `keep_loaded=True` on the text
encoder. With 4-bit Gemma (~8 GB) + bf16 DiT (~6.6 GB) + audio VAE (~2 GB) the
whole pipeline stays warm, so repeated generations skip all reloads. Use
`dramabox_standard_inference.json`.

**12 GB (3060 / 4070, workable).** Set `keep_loaded=False` on the text encoder
so the ~8 GB Gemma LLM is freed right after each prompt encode, leaving the DiT
+ VAE resident for sampling. Use `dramabox_lowvram_12gb.json`. A quantized
([GGUF](#gguf)) DiT lowers this further.

**Anywhere.** `DramaBox Unload Models` frees everything on demand at a chosen
point in the graph (also frees the 4-bit Gemma, which bitsandbytes pins so comfy
can't evict it automatically).

| Setup | Peak VRAM |
|---|---|
| bf16 DiT + 4-bit Gemma, all warm | ~16.6 GB |
| Q8_0 GGUF DiT + 4-bit Gemma | ~13.5 GB |

Measured on an RTX 3090, same prompt/seed.

## GGUF

The DiT Loader loads a `.gguf` audio DiT directly — weights stay quantized in
VRAM and each Linear dequantizes on-the-fly. Backend is
[calcuis/gguf](https://github.com/calcuis/gguf) (`pip install gguf-connector`).
No GGUF of the DramaBox audio DiT exists upstream (the published LTX-2.3 GGUFs
are the full 22B audio+video model); you produce one from the 3.3B audio DiT:

```bash
python -m dramabox_nodes.gguf.convert \
    --src models/diffusion_models/dramabox-dit-v1.safetensors \
    --dst models/diffusion_models/dramabox-dit-v1-Q8_0.gguf \
    --quant Q8_0            # or Q5_1 / Q5_0 / Q4_1 / Q4_0 / F16
```

The converter preserves the checkpoint's architecture config so the loader
rebuilds the exact model. Drop the `.gguf` in `models/diffusion_models` and pick
it in the DiT Loader.

**Measured on the RTX 3090** (Q8_0 vs bf16, same prompt/seed): peak VRAM
**13.5 GB vs 16.6 GB**; DiT file **3.5 GB vs 6.6 GB**; audio coherent.

Quant types: calcuis's pure-Python quantizer covers the legacy GGML types —
**Q8_0** (near-lossless, ~3.5 GB, recommended), Q5_0 (~2.3 GB), Q4_0 (~1.9 GB).
The K-quants (Q4_K/Q5_K/Q6_K) need llama.cpp's `llama-quantize` (convert to F16
here, then quantize with llama.cpp); the loader can *read* K-quants, it just
can't produce them in pure Python.

See [`conversion_scripts/`](conversion_scripts) for standalone convert tools.

## Using other LTX-2.3 finetunes

DramaBox is the audio branch of LTX-2.3. A full LTX-2.3 finetune's checkpoint
contains that same audio branch plus the video stream and the audio↔video
cross-attention. `DramaBox Extract Audio DiT` (node) / `finetune_extract` (CLI)
pulls out **exactly** the tensors the audio-only model needs and drops the rest,
producing a standalone `*-audio-only.safetensors` that loads through the normal
DiT Loader, accepts LoRAs, and can be GGUF-converted — the same size class as
DramaBox's own 6.6 GB DiT (the ~14 GB video stream is discarded).

```bash
python -m dramabox_nodes.finetune_extract \
    --src  https://huggingface.co/<user>/<repo>/resolve/main/<full-ltx2.3>.safetensors \
    --dst  models/diffusion_models/<name>-audio-only.safetensors
```

For a remote HF URL only the audio branch (~3.6 GB) is fetched via HTTP range
reads — not the whole 20–40 GB file. fp8 sources are dequantized to bf16 during
extraction. The extraction is **self-validating**: it keeps the DramaBox
audio-only model's exact `state_dict()` keys and hard-fails if the source is
missing any.

**Honest caveat.** The audio-only forward disables the audio↔video
cross-attention the full model was trained with, so a *video* finetune's audio
branch, run standalone, may produce generic/degraded speech rather than the
finetune's flavor. Loadability is guaranteed; audio quality is an experiment —
listen before relying on it. (Tested with `TenStrip/LTX2.3-10Eros`: extracts and
runs cleanly, but its audio isn't usable speech.) The productive long-term
target is LoRAs or finetunes of the **3.3B audio-only base** (DramaBox's
lineage), which drop straight into the loader with no extraction.

The end-to-end "full finetune → audio-only → GGUF" pipeline is also available as
a standalone script — see [`conversion_scripts/`](conversion_scripts).

## Notes and troubleshooting

**Corrupt/truncated model downloads.** A multi-GB download can end up truncated
(interrupted transfer, disk full mid-write), and a bare `hf_hub_download` only
confirms *some* file exists, not that it loads. `downloader.py` runs a real
`safetensors` metadata read after every download and deletes-and-retries once on
failure, so a corrupt file is caught at download time rather than deep in the
sampler.

**Multi-speaker identity drift in long-form generations.** Each chunk is
conditioned independently (text + the same voice reference; no carried-forward
speaker state), so with two speakers, later chunks can lose track of who's who —
a model-architecture limitation. Keeping a single strong voice reference and
shorter chunks helps.

**`bitsandbytes` CUDA mismatch** (`Configured CUDA binary not found`).
Environment-level: your PyTorch build's CUDA version has no matching prebuilt
`bitsandbytes` binary. Reinstall PyTorch against a CUDA version `bitsandbytes`
ships, upgrade `bitsandbytes`, or pick the **bf16 text encoder** in the loader
(no bitsandbytes, ~24 GB VRAM, HF-gated) instead of the 4-bit one.

**RE-USE voice-ref denoising unavailable.** Denoising needs
`mamba-ssm`/`causal-conv1d` (optional). If they're missing it's skipped with a
logged note and generation proceeds on the raw reference — set `denoise=False`
on the reference node to silence it. The `utils`/`models` namespace collision
that used to break RE-USE on Windows is fixed permanently in the vendored code
(no setup node needed). See below for installing the kernels on Windows.

**`resemble-perth` watermark import fails** (`'NoneType' object is not
callable`). `resemble-perth` swallows a missing-`librosa` ImportError; run
`pip install librosa`.

## Windows install for RE-USE CUDA kernels

RE-USE's `mamba-ssm`/`causal-conv1d` kernels give a large speedup over its
pure-PyTorch fallback for voice-ref denoising, but upstream's own
`requirements-reuse.txt` platform-gates both to `; platform_system == "Linux"` —
Windows users get no install guidance from upstream, and without the kernels
RE-USE either runs the much slower pure-PyTorch path or is unavailable depending
on what else is installed. This documents a confirmed-working Windows path.

1. **Cross-platform deps first** — no build step, work as-is on Windows:
   ```
   pip install librosa resampy
   ```
2. **`mamba-ssm` via a prebuilt wheel.** Dao-AILab doesn't publish Windows
   wheels, but third-party prebuilt wheels exist — e.g.
   [`loscrossos/lib_mamba`](https://github.com/loscrossos/lib_mamba)'s GitHub
   releases, named to encode the exact combo they were built against:
   `mamba_ssm-{version}+cu{XXX}torch{Y.Y.Y}-cp{XY}-cp{XY}-win_amd64.whl`. Check
   your own versions first and pick the matching asset:
   ```
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   python --version
   ```
   No build or CUDA toolkit needed for this one — it's a prebuilt binary.
3. **`causal-conv1d` via a local source build.** No Windows wheels are published,
   so it has to be built from source. You'll need `ninja`, a working MSVC/C++
   build toolchain (Visual Studio Build Tools), and a CUDA toolkit install —
   then:
   ```
   pip install ninja
   pip install -e .   # from a local clone/download of the causal-conv1d source
   ```
4. **The CUDA-toolkit-matching principle.** This step most often silently fails.
   `causal-conv1d`'s build compiles against whatever CUDA toolkit
   `CUDA_HOME`/`CUDA_PATH` points at — and with multiple toolkits installed side
   by side, the system default may point at the newest, **not** the version your
   PyTorch build targets. Check what PyTorch wants:
   ```
   python -c "import torch; print(torch.version.cuda)"
   ```
   Then point the build at the *matching* toolkit before `pip install -e .`
   (example matches a `torch.version.cuda == "12.9"` build):
   ```powershell
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
   $env:CUDA_HOME = $env:CUDA_PATH
   nvcc --version   # confirm it reports v12.9
   ```
   If the matching toolkit isn't installed, install it from NVIDIA first. A
   mismatched toolkit produces confusing build/load errors rather than a clear
   version complaint, so check this before debugging further.
5. **If the build fails or you'd rather skip it**, RE-USE still works via its
   pure-PyTorch fallback (slower, not broken) — the kernels are a speed
   optimization, not a requirement. Set `denoise=False` on the voice-reference
   node to skip RE-USE entirely.

## License

This node pack's own original code (everything in this repo outside
`dramabox_nodes/vendor/`) is licensed under **Apache 2.0** — see
[`LICENSE`](LICENSE).

The **DramaBox model** and the **vendored LTX-2 inference code**
(`dramabox_nodes/vendor/`, see `vendor/ATTRIBUTION.md`) are distributed under the
**LTX-2 Community License**; anything in this pack that touches upstream
weights/code follows those terms. **RE-USE** (optional reference denoising) is
separately licensed **NSCLv1 (non-commercial)** — keep it opt-in for commercial
work. See upstream's `LICENSE` files for details.
