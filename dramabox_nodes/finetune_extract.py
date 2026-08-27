"""Extract a standalone audio-only DiT from a full LTX-2.3 checkpoint.

DramaBox is the audio branch of LTX-2.3 (`_class_name = AVTransformer3DModel`).
A full LTX-2.3 finetune's checkpoint contains that same audio branch plus the
video stream and the audio<->video cross-attention. This tool keeps **exactly**
the tensors the DramaBox audio-only model needs and drops the rest, producing a
`*-audio-only.safetensors` that loads through the normal `DramaBoxDiTLoader`,
accepts LoRAs, and can be GGUF-converted — same size class as DramaBox's own
6.6 GB DiT (the ~14 GB video stream is discarded).

Self-validating: the "which keys to keep" set is the DramaBox audio-only
model's own `state_dict()` names, so a successful extraction is guaranteed to
fill every parameter the model requires (or it hard-fails listing what's
missing).

fp8 sources (e.g. 10Eros `fp8mixed`) are dequantized to bf16 during extraction
so the output is a plain, universally-loadable checkpoint; GGUF-quantize it
afterward for low VRAM.

CAVEAT: a full finetune's audio branch, run standalone (the audio-only forward
disables the audio<->video cross-attention it was trained with), may produce
generic/degraded speech rather than the finetune's flavor. Loadability is
guaranteed; audio quality is an experiment.

Usage (CLI):
    python -m dramabox_nodes.finetune_extract \
        --src  <path-or-hf-url to full LTX-2.3 .safetensors> \
        --dst  models/diffusion_models/<name>-audio-only.safetensors
"""
import argparse
import json
import logging
import struct
import urllib.request

log = logging.getLogger("ComfyUI-DramaBox")

_STRIP_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
# Output keys carry DramaBox's own prefix so the extracted file is key-format
# identical to dramabox-dit-v1.safetensors and loads via the same DIT_SD_OPS
# (which requires and strips this prefix).
_OUT_PREFIX = "model.diffusion_model."

# safetensors dtype string -> (numpy dtype, itemsize). fp8 handled specially.
_ST_DTYPE = {
    "F64": ("f8", 8), "F32": ("f4", 4), "F16": ("f2", 2), "BF16": ("bf16", 2),
    "I64": ("i8", 8), "I32": ("i4", 4), "I16": ("i2", 2), "I8": ("i1", 1),
    "U8": ("u1", 1), "BOOL": ("?", 1),
    "F8_E4M3": ("f8e4m3", 1), "F8_E5M2": ("f8e5m2", 1),
}


def _strip(name: str) -> str:
    for p in _STRIP_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def _target_keys_and_config(reference: str = None):
    """Return (set_of_target_tensor_names, config_dict) — the exact set of
    tensors the DramaBox audio-only DiT needs, and its architecture config.

    Two modes:
    - `reference` given → read the names + config directly from a known
      audio-only DiT safetensors (e.g. dramabox-dit-v1.safetensors). This is
      dependency-light (safetensors only, no torch/ComfyUI) and is what the
      standalone conversion scripts use.
    - else → build the DramaBox `AudioOnlyConfigurator` model on the meta
      device and read its `state_dict()` names (needs torch + the vendored
      model code; used inside ComfyUI).
    """
    if reference:
        header, _, _ = _read_header(_resolve_remote(reference))
        names = {_strip(k) for k in header if k != "__metadata__"}
        md = header.get("__metadata__") or {}
        config = json.loads(md["config"]) if "config" in md else {}
        return names, config

    import torch
    from .model.dit import AudioOnlyConfigurator
    config = _dramabox_config()
    with torch.device("meta"):
        model = AudioOnlyConfigurator.from_config(config)
    names = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    return names, config


def _dramabox_config() -> dict:
    """Read the config metadata from the installed DramaBox DiT (for the exact
    architecture the audio-only model expects)."""
    import os
    from . import config as cfg
    from .model.dit import read_checkpoint_config
    for d in (cfg.DIT_DIR,):
        p = os.path.join(d, cfg.DIT_FILENAME)
        if os.path.exists(p):
            return read_checkpoint_config(p)
    try:
        import folder_paths
        p = folder_paths.get_full_path("diffusion_models", cfg.DIT_FILENAME)
        if p:
            return read_checkpoint_config(p)
    except Exception:
        pass
    log.warning("[DramaBox] DramaBox DiT not found for config reference; using built-in defaults.")
    return {}


# ---------------------------------------------------------------------------
# safetensors header reading (local or remote via HTTP range)
# ---------------------------------------------------------------------------
def _read_header(src: str):
    """Return (header_dict, data_start_offset, is_remote)."""
    if src.startswith(("http://", "https://")):
        n = struct.unpack("<Q", _range_get(src, 0, 7))[0]
        hdr = json.loads(_range_get(src, 8, 8 + n - 1).decode("utf-8"))
        return hdr, 8 + n, True
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
        return hdr, 8 + n, False


def _range_get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                               "User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=120).read()


def _resolve_remote(src: str) -> str:
    """Turn an HF blob/tree URL into a resolve (raw) URL."""
    if "huggingface.co" in src and "/resolve/" not in src:
        src = src.replace("/blob/", "/resolve/")
    return src


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract(src: str, dst: str, keep_components: bool = False, reference: str = None) -> dict:
    """Extract the audio-only DiT tensors from `src` into `dst`.
    `reference` (optional): an audio-only DiT safetensors whose tensor names
    define the extraction target (dependency-light path for standalone use;
    see `_target_keys_and_config`). Returns a summary dict."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    src = _resolve_remote(src)
    target_names, config = _target_keys_and_config(reference)
    header, data_start, is_remote = _read_header(src)

    # Map stripped-name -> raw header entry, for names in the target set.
    matched = {}
    for raw_name, info in header.items():
        if raw_name == "__metadata__":
            continue
        sname = _strip(raw_name)
        if sname in target_names:
            matched[sname] = (raw_name, info)

    missing = sorted(target_names - set(matched))
    if missing:
        raise RuntimeError(
            f"[DramaBox] Source is missing {len(missing)} audio-only DiT tensors, e.g. "
            f"{missing[:8]}. Not an LTX-2.3 checkpoint with a compatible audio branch."
        )

    total_kept = len(matched)
    total_src = sum(1 for k in header if k != "__metadata__")
    log.info(f"[DramaBox] Extracting {total_kept} audio-DiT tensors from {total_src} "
             f"({'remote' if is_remote else 'local'} source); dropping "
             f"{total_src - total_kept} video/AV-cross tensors.")

    out = {}
    fh = None if is_remote else open(src, "rb")
    try:
        for sname, (raw_name, info) in matched.items():
            b0, b1 = info["data_offset"] if False else info["data_offsets"]
            start, end = data_start + b0, data_start + b1
            if is_remote:
                blob = _range_get(src, start, end - 1)
            else:
                fh.seek(start)
                blob = fh.read(end - start)
            out[_OUT_PREFIX + sname] = _to_bf16_tensor(blob, info["dtype"], info["shape"], np, torch)
    finally:
        if fh is not None:
            fh.close()

    meta = {"config": json.dumps(config)} if config else {}
    save_file(out, dst, metadata=meta)
    kept_bytes = sum(t.numel() * t.element_size() for t in out.values())
    log.info(f"[DramaBox] Wrote {dst}: {total_kept} tensors, {kept_bytes/1e9:.2f} GB (bf16).")
    return {"dst": dst, "kept": total_kept, "dropped": total_src - total_kept,
            "size_gb": kept_bytes / 1e9}


def _to_bf16_tensor(blob: bytes, dtype: str, shape, np, torch):
    """Decode a raw tensor blob to a bf16 torch tensor (dequantizing fp8)."""
    if dtype in ("F8_E4M3", "F8_E5M2"):
        tdt = torch.float8_e4m3fn if dtype == "F8_E4M3" else torch.float8_e5m2
        t = torch.frombuffer(bytearray(blob), dtype=tdt).reshape(shape)
        return t.to(torch.bfloat16)
    if dtype == "BF16":
        t = torch.frombuffer(bytearray(blob), dtype=torch.bfloat16).reshape(shape)
        return t.clone()
    np_dt = {"F32": np.float32, "F16": np.float16, "F64": np.float64,
             "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
             "U8": np.uint8, "BOOL": np.bool_}[dtype]
    arr = np.frombuffer(bytearray(blob), dtype=np_dt).reshape(shape)
    return torch.from_numpy(arr.copy()).to(torch.bfloat16)


class DramaBoxExtractAudioDiT:
    """Extracts a standalone audio-only DiT from a full LTX-2.3 checkpoint
    (safetensors) into models/diffusion_models, ready for DramaBoxDiTLoader.
    Source can be a local file in models/diffusion_models or an HF URL
    (only the ~3.6 GB audio branch is fetched via range reads, not the whole
    20-40 GB file). See module docstring for the standalone-audio caveat."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("STRING", {"default": "",
                    "tooltip": "Local filename in models/diffusion_models, an absolute path, "
                               "or a HuggingFace URL to a full LTX-2.3 .safetensors."}),
                "output_name": ("STRING", {"default": "ltx-audio-only.safetensors"}),
                "confirm": ("BOOLEAN", {"default": False,
                    "tooltip": "Extraction downloads/reads several GB. Set True to run."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_dit_path",)
    FUNCTION = "run"
    CATEGORY = "DramaBox/extract"
    OUTPUT_NODE = True

    def run(self, source, output_name, confirm):
        import os
        from . import config as cfg
        if not confirm:
            return ("Set confirm=True to run extraction (reads several GB).",)
        src = source.strip()
        if not src:
            raise ValueError("source is empty.")
        if not src.startswith(("http://", "https://")) and not os.path.isabs(src):
            resolved = os.path.join(cfg.DIT_DIR, src)
            if not os.path.exists(resolved):
                try:
                    import folder_paths
                    fp = folder_paths.get_full_path("diffusion_models", src)
                    resolved = fp or resolved
                except Exception:
                    pass
            src = resolved
        if not output_name.endswith(".safetensors"):
            output_name += ".safetensors"
        dst = os.path.join(cfg.DIT_DIR, output_name)
        summary = extract(src, dst)
        log.info(f"[DramaBox] Extraction summary: {summary}")
        return (dst,)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Extract an audio-only DiT from a full LTX-2.3 checkpoint.")
    ap.add_argument("--src", required=True, help="full LTX-2.3 .safetensors (local path or HF URL)")
    ap.add_argument("--dst", required=True, help="output <name>-audio-only.safetensors")
    args = ap.parse_args()
    summary = extract(args.src, args.dst)
    log.info(f"Done: {summary}")


if __name__ == "__main__":
    main()
