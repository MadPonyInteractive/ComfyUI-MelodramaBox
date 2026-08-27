"""Output-side nodes: saving, watermark application, watermark detection.

Saving goes through soundfile directly — v1 monkeypatched
torchaudio.save/load process-wide to survive torchaudio>=2.9's
torchcodec-only backend; using soundfile here removes both the monkeypatch
and the torchaudio dependency from the save path entirely.
"""
import logging
import os

log = logging.getLogger("ComfyUI-DramaBox")

try:
    import folder_paths
    _OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    _OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

_perth_watermarker = None


def _get_perth():
    global _perth_watermarker
    if _perth_watermarker is None:
        import perth
        _perth_watermarker = perth.PerthImplicitWatermarker()
    return _perth_watermarker


class DramaBoxSaveAudio:
    """Saves AUDIO to ComfyUI's output/ directory as .wav or .flac,
    following the native SaveAudio filename_prefix + counter convention.
    Passthrough AUDIO so it can sit mid-graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "dramabox"}),
                "format": (["wav", "flac"], {"default": "wav"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "saved_path")
    FUNCTION = "save"
    CATEGORY = "DramaBox/output"
    OUTPUT_NODE = True

    def save(self, audio, filename_prefix, format):
        import soundfile as sf

        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        if waveform.ndim == 3:
            waveform = waveform[0]

        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:05d}.{format}"
            path = os.path.join(_OUTPUT_DIR, filename)
            if not os.path.exists(path):
                break
            counter += 1

        sf.write(path, waveform.cpu().float().numpy().T, sr)  # (T, C)
        log.info(f"[DramaBox] Saved audio to {path}")
        return (audio, path)


class DramaBoxApplyWatermark:
    """Applies the Resemble Perth imperceptible watermark. v1 buried this
    inside generation (upstream defaults it on); as a standalone node the
    graph makes the choice explicit and it can be applied after any
    processing chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply"
    CATEGORY = "DramaBox/output"

    def apply(self, audio):
        import numpy as np
        import torch

        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        batched = waveform.ndim == 3
        wav = waveform[0] if batched else waveform
        wav = wav.cpu().float()

        mono = wav.mean(dim=0).numpy() if wav.shape[0] > 1 else wav[0].numpy()
        marked = _get_perth().apply_watermark(mono, sample_rate=sr)
        marked_t = torch.from_numpy(np.asarray(marked, dtype=np.float32)).unsqueeze(0)
        out = marked_t if wav.shape[0] == 1 else marked_t.repeat(wav.shape[0], 1)
        if batched:
            out = out.unsqueeze(0)
        log.info("[DramaBox] Perth watermark applied.")
        return ({"waveform": out, "sample_rate": sr},)


class DramaBoxWatermarkCheck:
    """Runs the Resemble Perth detector and reports detection confidence —
    a QA step for confirming a clip carries the watermark."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("confidence", "report")
    FUNCTION = "check"
    CATEGORY = "DramaBox/output"

    def check(self, audio):
        try:
            detector = _get_perth()
        except ImportError as e:
            return (0.0, f"resemble-perth not installed: {e}")

        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        if waveform.ndim == 3:
            waveform = waveform[0]
        mono = waveform.mean(dim=0).cpu().numpy()

        confidence = float(detector.get_watermark(mono, sample_rate=sr))
        report = f"Perth watermark confidence: {confidence:.4f}"
        log.info(f"[DramaBox] {report}")
        return (confidence, report)
