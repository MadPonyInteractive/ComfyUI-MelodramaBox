"""
Central configuration: model repo IDs, filenames, and folder layout.

v2.0: inference no longer goes through upstream's TTSServer — the model
code is vendored (see vendor/ATTRIBUTION.md) and weights live in ComfyUI's
standard model folders instead of the bare Hugging Face cache:

    models/diffusion_models/dramabox-dit-v1.safetensors     (audio DiT, 6.6 GB)
    models/vae/dramabox-audio-components.safetensors        (VAE + vocoder +
                                                             embeddings connector, 1.9 GB)
    models/text_encoders/<gemma snapshot dir>/              (Gemma-3-12B)
    models/loras/                                            (DramaBox LoRAs)

models/dramabox/ holds voice-reference source files and temp space.
"""
import os

try:
    import folder_paths
    MODELS_DIR = os.path.join(folder_paths.models_dir, "dramabox")
    DIT_DIR = folder_paths.get_folder_paths("diffusion_models")[0]
    VAE_DIR = folder_paths.get_folder_paths("vae")[0]
    TEXT_ENCODER_DIR = folder_paths.get_folder_paths("text_encoders")[0]
except ImportError:
    # Allows the package to be imported/tested outside a running ComfyUI
    _root = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    MODELS_DIR = os.path.join(_root, "dramabox")
    DIT_DIR = os.path.join(_root, "diffusion_models")
    VAE_DIR = os.path.join(_root, "vae")
    TEXT_ENCODER_DIR = os.path.join(_root, "text_encoders")

VOICE_REF_DIR = os.path.join(MODELS_DIR, "voice_refs")
TEMP_DIR = os.path.join(MODELS_DIR, "tmp")

for _d in (VOICE_REF_DIR, TEMP_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Model weights (HF Hub)
# ---------------------------------------------------------------------------
DRAMABOX_REPO = "ResembleAI/Dramabox"
DIT_FILENAME = "dramabox-dit-v1.safetensors"                         # ~6.6 GB
AUDIO_COMPONENTS_FILENAME = "dramabox-audio-components.safetensors"  # ~1.9 GB

# Gemma-3-12B text encoder. The 4-bit unsloth snapshot is pre-quantized
# (bnb nf4) and is what upstream's own inference path uses by default; the
# google repo is full-precision bf16 (~24 GB) and gated behind Google's
# license (needs HF auth). Resemble's config.json nominally names
# google/gemma-3-12b-it-qat-q4_0-unquantized; the unsloth 4-bit checkpoint
# is the equivalent upstream actually loads at inference.
TEXT_ENCODER_4BIT_REPO = "unsloth/gemma-3-12b-it-bnb-4bit"   # ~8 GB
TEXT_ENCODER_BF16_REPO = "google/gemma-3-12b-it"             # ~24 GB, gated
TEXT_ENCODER_4BIT_DIRNAME = "gemma-3-12b-it-bnb-4bit"
TEXT_ENCODER_BF16_DIRNAME = "gemma-3-12b-it"

# ---------------------------------------------------------------------------
# Generation defaults (from upstream inference_server.py and the checkpoint
# config metadata)
# ---------------------------------------------------------------------------
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent, robotic, distorted, noise, static, "
    "muffled, unclear, unnatural, monotone"
)

# steps now actually plumb through to the scheduler (v1 silently ignored them
# because upstream hardcoded steps=30 inside TTSServer.generate).
QUALITY_PRESETS = {
    "draft":   {"steps": 12, "cfg_scale": 2.0, "stg_scale": 1.0},
    "default": {"steps": 30, "cfg_scale": 2.5, "stg_scale": 1.5},
    "high":    {"steps": 50, "cfg_scale": 3.0, "stg_scale": 1.8},
}

# STG perturbs this transformer block (upstream: stg_blocks=[29]).
STG_BLOCKS = [29]

# Latent frame rate of the audio branch: 25 latent frames per second.
AUDIO_LATENT_FPS = 25.0

# Output sample rate. The audio VAE decodes at 24 kHz and the vocoder's
# BigVGAN BWE brings it to 48 kHz — vocoder.output_sampling_rate is the
# source of truth at runtime; this constant is for display/estimates only.
# (v1 wrongly said 24000 here.)
OUTPUT_SAMPLE_RATE = 48000

# Base DiT trained on clips <=~20s; usable up to ~45s (end-of-clip
# silence-prior patch applied in sampling.py past the 20.4s latent boundary).
MAX_CHUNK_DURATION_HARD_CAP = 45.0
TARGET_CHUNK_DURATION_DEFAULT = 37.0

# ---------------------------------------------------------------------------
# Zero-shot voice cloning
# ---------------------------------------------------------------------------
ZERO_SHOT_MIN_RECOMMENDED_SEC = 10.0
ZERO_SHOT_MAX_USEFUL_SEC = 30.0
ZERO_SHOT_TRIM_SILENCE_DB = 40.0
ZERO_SHOT_CROSSFADE_MS = 30
# Upstream tiles/pads the reference to exactly this many seconds and
# peak-normalizes to -4 dBFS before VAE encoding.
VOICE_REF_ENCODE_SECONDS = 10.0
VOICE_REF_PEAK_DBFS = -4.0
# Reference clips are prepped at this rate; encode_audio resamples
# internally to whatever the VAE encoder expects.
VOICE_REF_PREP_SAMPLE_RATE = 48000
