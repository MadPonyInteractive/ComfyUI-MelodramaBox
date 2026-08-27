"""Prompt-analysis helper nodes."""
from . import config

class DramaBoxEstimateDuration:
    """Estimates output duration from prompt text using the same
    sentence-aware estimator the sampler uses to size its latent (v1 used a
    separate heuristic that could disagree with actual generation length).
    Useful for planning long-form chunk sizes ahead of a run."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration_multiplier": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 3.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("estimated_seconds", "report")
    FUNCTION = "estimate"
    CATEGORY = "DramaBox/utils"

    def estimate(self, prompt, duration_multiplier):
        from . import sampling
        seconds = sampling.estimate_duration(prompt, duration_multiplier)

        if seconds > config.MAX_CHUNK_DURATION_HARD_CAP:
            fit_note = (
                f"Exceeds the single-pass cap ({config.MAX_CHUNK_DURATION_HARD_CAP:.0f}s) — "
                f"use DramaBox Long-Form Text Encode so the sampler chunks it."
            )
        else:
            fit_note = f"Fits in a single pass ({config.MAX_CHUNK_DURATION_HARD_CAP:.0f}s cap)."
        report = f"~{seconds:.1f}s estimated (multiplier {duration_multiplier}). {fit_note}"
        return (round(seconds, 2), report)
