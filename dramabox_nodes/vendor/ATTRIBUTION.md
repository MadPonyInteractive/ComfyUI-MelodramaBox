# Vendored code attribution

This directory contains code vendored from:

- **Repository:** https://github.com/resemble-ai/DramaBox
- **Commit:** `a70a5818e103c1c9fef22409c1e0c707ebf4f8a7` (2026-05-23)
- **License:** LTX-2 Community License (see `LICENSE-LTX2.txt`)
- **Copyright:** Lightricks Ltd. (LTX-2 core) and Resemble AI (DramaBox)

## What is vendored

| Vendored path | Upstream path | Notes |
|---|---|---|
| `mdb_ltx_core/` | `ltx2/ltx_core/` | Complete package (transformer, audio VAE, vocoder, Gemma text-encoder wrappers, loaders, schedulers, quantization). |
| `mdb_ltx_pipelines/utils/` | `ltx2/ltx_pipelines/utils/` | Pipeline building blocks, denoisers, samplers, helpers. The top-level video pipelines are not vendored; `__init__.py` is replaced with a minimal stub. |
| `mdb_dramabox/` | `src/{audio_conditioning,duration_estimator,text_chunker,super_resolution}.py` | DramaBox-specific conditioning, duration estimation, prompt chunking, and optional RE-USE voice-reference denoising. |

## Modifications

1. Top-level package names renamed (`ltx_core` → `mdb_ltx_core`,
   `ltx_pipelines` → `mdb_ltx_pipelines`) and all imports rewritten
   accordingly, to avoid namespace collisions inside a shared ComfyUI
   process.
2. `mdb_dramabox/super_resolution.py`: RE-USE snapshot import made
   namespace-collision-safe on Windows/ComfyUI (regular-package markers +
   scoped `sys.modules` handling for the snapshot's top-level `models`/
   `utils` packages), replacing the on-disk patching the node pack
   previously applied to a runtime clone.
3. `mdb_ltx_pipelines/utils/media_io.py`: `import av` (PyAV) made lazy and
   annotations deferred (`from __future__ import annotations`), so the
   module imports without PyAV installed. Only the video/image muxing
   functions touch `av`, and DramaBox (audio-only) never calls them —
   avoiding a heavy PyAV dependency on every install.
4. `mdb_ltx_pipelines/utils/__init__.py`: emptied. Upstream re-exported the
   full video pipeline (blocks) here; the node pack imports the audio
   pieces it needs by full submodule path, so the re-export would only pull
   the video-generation chain (video VAE, upsampler, quantization) into a
   TTS pack for no benefit.

Why vendor at all: the node pack previously `git clone`d upstream at
runtime and imported through `sys.path` injection, which caused top-level
`src` namespace collisions, made behavior change under it mid-session, and
required monkeypatching upstream files on disk. Vendoring at a pinned
commit removes that whole class of failure.

RE-USE / SEMamba (used by `super_resolution.py`) is downloaded separately
at runtime by upstream code and is licensed under NVIDIA's NSCLv1
(non-commercial). It is opt-in and clearly flagged in the nodes that use
it. No RE-USE code is vendored here.
