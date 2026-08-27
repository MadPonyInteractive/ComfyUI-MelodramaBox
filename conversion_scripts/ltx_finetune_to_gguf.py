#!/usr/bin/env python3
"""One-shot: full LTX-2.3 finetune (fp8 or bf16 safetensors) -> audio-only GGUF.

Chains the two steps so anyone with an LTX-2.3 finetune can get a small,
GGUF-quantized audio DiT that loads in ComfyUI-MelodramaBox:

    1. EXTRACT the audio branch (drops the ~14 GB video stream + AV-cross;
       keeps exactly the tensors the DramaBox audio-only DiT needs). fp8 is
       dequantized to bf16. From a HuggingFace URL only the ~3.6 GB audio
       branch is fetched via HTTP range reads — not the whole 20-40 GB file.
    2. CONVERT that audio-only DiT to GGUF (Q8_0 by default; Q8_0 is
       near-lossless and the safe choice for speech).

Standalone: needs only `safetensors`, `numpy`, `torch`, and
`gguf-connector` (`pip install gguf-connector`). Does NOT require ComfyUI to
be importable — the extraction target keys come from a REFERENCE audio-only
DiT (dramabox-dit-v1.safetensors), which you must have locally.

Examples
--------
Full finetune on HuggingFace -> Q8_0 GGUF:

    python ltx_finetune_to_gguf.py \
        --src https://huggingface.co/<user>/<repo>/resolve/main/<finetune>.safetensors \
        --reference /path/to/dramabox-dit-v1.safetensors \
        --name my_finetune \
        --quant Q8_0 \
        --out-dir /path/to/ComfyUI/models/diffusion_models

Local full finetune, keep the intermediate bf16 audio-only file too:

    python ltx_finetune_to_gguf.py \
        --src /models/10Eros_v1.4_fp8mixed_learned.safetensors \
        --reference /models/dramabox-dit-v1.safetensors \
        --name 10eros --quant Q8_0 --keep-safetensors

READ THIS. The audio-only forward runs WITHOUT the audio<->video
cross-attention the full model was trained with. A *video* finetune's audio
branch, extracted this way, usually produces continuous/tonal non-speech
rather than clean TTS. Loadability is guaranteed; usable speech is not —
listen to a sample before relying on it. Finetunes/LoRAs of the audio-only
base (DramaBox's lineage) are the reliable target.
"""
import argparse
import logging
import os
import sys

# Make `dramabox_nodes` importable when run from the conversion_scripts/ dir.
_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ltx_finetune_to_gguf")

QUANTS = ["Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0", "F16"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="Full LTX-2.3 finetune .safetensors (local path or HF URL).")
    ap.add_argument("--reference", required=True,
                    help="An audio-only DiT safetensors whose keys define the target "
                         "(normally dramabox-dit-v1.safetensors).")
    ap.add_argument("--name", required=True, help="Base name for outputs, e.g. 'my_finetune'.")
    ap.add_argument("--out-dir", default=".", help="Output directory (default: current dir).")
    ap.add_argument("--quant", default="Q8_0", choices=QUANTS,
                    help="GGUF quant type (default Q8_0, near-lossless). K-quants need llama.cpp.")
    ap.add_argument("--extract-only", action="store_true", help="Stop after extracting the audio-only DiT.")
    ap.add_argument("--keep-safetensors", action="store_true",
                    help="Keep the intermediate bf16 audio-only .safetensors (deleted otherwise).")
    args = ap.parse_args()

    from dramabox_nodes.finetune_extract import extract
    from dramabox_nodes.gguf.convert import convert

    os.makedirs(args.out_dir, exist_ok=True)
    audio_st = os.path.join(args.out_dir, f"{args.name}-audio-only.safetensors")

    log.info(f"[1/2] Extracting audio branch: {args.src}")
    summary = extract(args.src, audio_st, reference=args.reference)
    log.info(f"      kept {summary['kept']} tensors, dropped {summary['dropped']}, "
             f"{summary['size_gb']:.2f} GB -> {audio_st}")

    if args.extract_only:
        log.info("Done (extract-only).")
        return

    gguf_out = os.path.join(args.out_dir, f"{args.name}-audio-only-{args.quant}.gguf")
    log.info(f"[2/2] Converting to GGUF ({args.quant}): {gguf_out}")
    convert(audio_st, gguf_out, args.quant)

    if not args.keep_safetensors:
        try:
            os.remove(audio_st)
            log.info(f"      removed intermediate {audio_st} (use --keep-safetensors to keep)")
        except OSError:
            pass

    log.info(f"Done -> {gguf_out}")
    log.info("Drop it in ComfyUI/models/diffusion_models and pick it in the DramaBox DiT Loader.")


if __name__ == "__main__":
    main()
