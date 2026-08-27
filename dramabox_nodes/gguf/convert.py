"""Quantize a DramaBox audio-only DiT safetensors to GGUF.

Backend: **calcuis/gguf** (`gguf_connector`) — `writer.GGUFWriter`,
`quant.quantize`, `reader.GGMLQuantizationType`.

Usage:
    python -m dramabox_nodes.gguf.convert \
        --src /path/to/dramabox-dit-v1.safetensors \
        --dst /path/to/dramabox-dit-v1-Q5_K.gguf \
        --quant Q5_K

Works on the DramaBox DiT and on any audio-only DiT produced by
`dramabox_nodes.finetune_extract` (they share the architecture and the
`config` safetensors metadata, preserved verbatim into the GGUF so the loader
rebuilds the exact model).

Rules (matching how ComfyUI-GGUF converts diffusion models):
- 2D+ weights whose last dim is a multiple of the quant block size are
  quantized to the requested type;
- 1D tensors (norms, biases, scale_shift_tables) and any weight that can't be
  block-aligned stay F16.

Quant types: calcuis's pure-Python quantizer implements the "legacy" GGML
types (Q8_0, Q5_0/Q5_1, Q4_0/Q4_1) — Q8_0 is near-lossless and the safe choice
for speech. The K-quants (Q4_K/Q5_K/Q6_K) need llama.cpp's C `llama-quantize`
and are *not* produced here (though the loader can read them if you make one
with llama.cpp from the F16 output). VRAM for the 3.3B DiT: Q8_0 ~3.5 GB,
Q5_0 ~2.3 GB, Q4_0 ~1.9 GB.

This produces an artifact that doesn't exist upstream. If you publish it,
the LTX-2 Community License permits redistribution of derivatives with
attribution — credit Lightricks / Resemble AI in the model card.
"""
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dramabox.gguf.convert")

# GGML types calcuis can quantize to in pure Python (K-quants need llama.cpp).
QUANT_TYPES = ["Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0", "F16"]


def _imports():
    try:
        from gguf_connector.writer import GGUFWriter
        from gguf_connector.quant import quantize
        from gguf_connector.reader import GGML_QUANT_SIZES, GGMLQuantizationType
    except ImportError as e:
        raise SystemExit(
            "calcuis/gguf is required for conversion:\n    pip install gguf-connector\n"
            f"({e})"
        )
    return GGUFWriter, quantize, GGML_QUANT_SIZES, GGMLQuantizationType


def convert(src: str, dst: str, quant: str) -> None:
    import numpy as np
    import torch
    from safetensors import safe_open
    GGUFWriter, quantize, GGML_QUANT_SIZES, QT = _imports()

    if quant not in QUANT_TYPES:
        raise SystemExit(f"--quant must be one of {QUANT_TYPES}, got {quant!r}")
    target = getattr(QT, quant)
    block_size = GGML_QUANT_SIZES[target][0] if quant != "F16" else 1

    with safe_open(src, framework="pt") as f:
        metadata = f.metadata() or {}
        keys = list(f.keys())
        if "config" not in metadata:
            log.warning("Source has no 'config' metadata — the GGUF loader will fall back "
                        "to built-in dramabox-dit-v1 architecture defaults.")

        writer = GGUFWriter(dst, arch="dramabox_audio_dit")
        if "config" in metadata:
            writer.add_string("dramabox.config", metadata["config"])
            t = json.loads(metadata["config"]).get("transformer", {})
            writer.add_uint32("dramabox.num_layers", int(t.get("num_layers", 48)))
        writer.add_string("dramabox.source_file", src.replace("\\", "/").rsplit("/", 1)[-1])
        writer.add_string("dramabox.quant", quant)

        n_quant = n_kept = 0
        for key in keys:
            arr = f.get_tensor(key).to(dtype=torch.float32).cpu().numpy()
            quantizable = (quant != "F16" and arr.ndim >= 2
                           and arr.shape[-1] % block_size == 0)
            if quantizable:
                writer.add_tensor(key, quantize(arr, target), raw_dtype=target)
                n_quant += 1
            else:
                writer.add_tensor(key, arr.astype(np.float16), raw_dtype=QT.F16)
                n_kept += 1

        log.info(f"Writing {dst}: {n_quant} tensors -> {quant}, {n_kept} kept F16")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    log.info("Done.")


def main():
    ap = argparse.ArgumentParser(description="Quantize a DramaBox audio DiT to GGUF (calcuis/gguf).")
    ap.add_argument("--src", required=True, help="audio-only DiT safetensors (DramaBox or extracted)")
    ap.add_argument("--dst", required=True, help="output .gguf path")
    ap.add_argument("--quant", default="Q8_0", choices=QUANT_TYPES)
    args = ap.parse_args()
    convert(args.src, args.dst, args.quant)


if __name__ == "__main__":
    main()
