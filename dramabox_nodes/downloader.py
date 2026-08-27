"""Model downloading into ComfyUI's standard model folders.

v2.0: weights live where every other ComfyUI model lives —
models/diffusion_models (DiT), models/vae (audio components),
models/text_encoders/<snapshot dir> (Gemma) — instead of the bare
Hugging Face cache. If a v1 install already has a file in the HF cache,
it is copied over rather than re-downloaded.

Every safetensors download gets a real functional integrity check
(safe_open metadata read) with one force_download retry, so a truncated
multi-GB transfer is caught here instead of surfacing as a cryptic
"invalid python storage" crash mid-generation.
"""

# MODIFIED by Mad Pony Interactive (fork of comfyui-melodramabox 2.1.0).
# Changes are marked `MPI-607` below and described in the repository NOTICE file.
import logging
import os
import shutil

from . import config

log = logging.getLogger("ComfyUI-DramaBox")

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    _HAS_HF_HUB = True
except ImportError:
    _HAS_HF_HUB = False

try:
    from safetensors import safe_open
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


def _require_hf_hub():
    if not _HAS_HF_HUB:
        raise RuntimeError(
            "huggingface_hub is not installed. Run:\n"
            "  pip install -r custom_nodes/ComfyUI-MelodramaBox/requirements.txt"
        )


def _is_valid_safetensors_file(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if not _HAS_SAFETENSORS:
        return True  # can't verify without the package — don't block on it
    try:
        if os.path.getsize(path) == 0:
            return False
        with safe_open(path, framework="pt") as f:
            f.metadata()
        return True
    except Exception as e:
        log.warning(f"[DramaBox] Integrity check failed for {path}: {e!r}")
        return False


def _migrate_from_hf_cache(repo_id: str, filename: str, dest: str) -> bool:
    """Copy a file from a v1-era HF cache entry into `dest` if present."""
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
    except Exception:
        return False
    if not _is_valid_safetensors_file(cached):
        return False
    log.info(f"[DramaBox] Migrating {filename} from HF cache into {os.path.dirname(dest)} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(cached, dest)
    return _is_valid_safetensors_file(dest)


def _ensure_file(repo_id: str, filename: str, dest_dir: str, size_hint: str) -> str:
    """Ensure `dest_dir/filename` exists and is a loadable safetensors file."""
    _require_hf_hub()
    dest = os.path.join(dest_dir, filename)
    if _is_valid_safetensors_file(dest):
        return dest
    os.makedirs(dest_dir, exist_ok=True)

    if _migrate_from_hf_cache(repo_id, filename, dest):
        return dest

    log.info(f"[DramaBox] Downloading {filename} ({size_hint}) to {dest_dir} ...")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)
    if _is_valid_safetensors_file(dest):
        return dest

    log.warning(f"[DramaBox] {filename} failed integrity check — re-downloading once.")
    try:
        os.remove(dest)
    except OSError:
        pass
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir, force_download=True)
    if not _is_valid_safetensors_file(dest):
        raise RuntimeError(
            f"[DramaBox] {filename} still fails integrity check after a fresh re-download "
            f"({dest}). Check disk space and HuggingFace Hub status, or place the file "
            f"there manually."
        )
    return dest


def ensure_dit() -> str:
    return _ensure_file(config.DRAMABOX_REPO, config.DIT_FILENAME, config.DIT_DIR, "~6.6 GB")


def ensure_audio_components() -> str:
    return _ensure_file(config.DRAMABOX_REPO, config.AUDIO_COMPONENTS_FILENAME,
                        config.VAE_DIR, "~1.9 GB")


def _is_valid_text_encoder_snapshot(snapshot_dir: str) -> bool:
    if not os.path.exists(os.path.join(snapshot_dir, "config.json")):
        return False
    weight_files = [f for f in os.listdir(snapshot_dir) if f.endswith(".safetensors")]
    return bool(weight_files) and all(
        _is_valid_safetensors_file(os.path.join(snapshot_dir, f)) for f in weight_files
    )


def _text_encoder_roots():
    """Every registered `text_encoders` root, in ComfyUI's own order.

    MPI-607: extra_model_paths.yaml roots are appended to this list, never
    prepended, so `[0]` alone never sees the shared Cubric model store.
    """
    try:
        import folder_paths
        return list(folder_paths.get_folder_paths("text_encoders"))
    except Exception:
        return [config.TEXT_ENCODER_DIR]


def ensure_text_encoder(repo_id: str = None, dirname: str = None) -> str:
    """Ensure a Gemma snapshot exists under models/text_encoders/<dirname>.
    Returns the snapshot directory."""
    _require_hf_hub()
    repo_id = repo_id or config.TEXT_ENCODER_4BIT_REPO
    dirname = dirname or config.TEXT_ENCODER_4BIT_DIRNAME

    # MPI-607: honour every registered text_encoders root, not just [0].
    for _root in _text_encoder_roots():
        _cand = os.path.join(_root, dirname)
        if _is_valid_text_encoder_snapshot(_cand):
            log.info("[DramaBox] Text encoder found at %s" % _cand)
            return _cand

    dest = os.path.join(config.TEXT_ENCODER_DIR, dirname)

    log.info(f"[DramaBox] Downloading text encoder {repo_id} to {dest} ...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=dest,
                          ignore_patterns=["*.md", "*.gguf", "*.gg", "original/*"])
    except Exception as e:
        if repo_id == config.TEXT_ENCODER_BF16_REPO:
            raise RuntimeError(
                f"[DramaBox] Download of {repo_id} failed ({e}). This repo is gated behind "
                "Google's license — accept it on huggingface.co and authenticate "
                "(`huggingface-cli login` or HF_TOKEN) first."
            ) from e
        raise
    if not _is_valid_text_encoder_snapshot(dest):
        raise RuntimeError(
            f"[DramaBox] Text encoder snapshot at {dest} fails integrity check after "
            f"download — check disk space and HuggingFace Hub status."
        )
    return dest


class DramaBoxDownloadModels:
    """Downloads every core model into ComfyUI's model folders in one click.
    The loaders offer per-file download entries in their dropdowns, so this
    node is optional — useful for warming a fresh install or a headless box."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "download_dit": ("BOOLEAN", {"default": True}),
                "download_audio_components": ("BOOLEAN", {"default": True}),
                "download_text_encoder_4bit": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "download"
    CATEGORY = "DramaBox/setup"
    OUTPUT_NODE = True

    def download(self, download_dit, download_audio_components, download_text_encoder_4bit):
        lines = []
        if download_dit:
            lines.append(f"DiT: {ensure_dit()}")
        if download_audio_components:
            lines.append(f"Audio components: {ensure_audio_components()}")
        if download_text_encoder_4bit:
            lines.append(f"Text encoder: {ensure_text_encoder()}")
        status = "\n".join(lines) if lines else "Nothing selected."
        log.info("[DramaBox] " + status.replace("\n", " | "))
        return (status,)
