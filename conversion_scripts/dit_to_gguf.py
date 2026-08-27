#!/usr/bin/env python3
"""Convert an audio-only DiT safetensors -> GGUF.

Use this when you already have an audio-only DiT (DramaBox's own
`dramabox-dit-v1.safetensors`, or one produced by `ltx_finetune_to_gguf.py`
/ the extraction node) and just want to quantize it.

    python dit_to_gguf.py \
        --src /path/to/dramabox-dit-v1.safetensors \
        --dst /path/to/dramabox-dit-v1-Q8_0.gguf \
        --quant Q8_0

Quant types (calcuis pure-Python quantizer): Q8_0 (near-lossless, ~3.5 GB
for the 3.3B DiT, recommended), Q5_0 (~2.3 GB), Q4_0 (~1.9 GB), Q5_1, Q4_1,
F16. K-quants (Q4_K/Q5_K/Q6_K) need llama.cpp's `llama-quantize` — convert to
F16 here, then run llama-quantize; the loader can read the result.

Needs only `safetensors`, `numpy`, `torch`, `gguf-connector`.
"""
import argparse
import os
import sys

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)


def main():
    from dramabox_nodes.gguf.convert import QUANT_TYPES, convert
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="audio-only DiT .safetensors")
    ap.add_argument("--dst", required=True, help="output .gguf")
    ap.add_argument("--quant", default="Q8_0", choices=QUANT_TYPES)
    args = ap.parse_args()
    convert(args.src, args.dst, args.quant)


if __name__ == "__main__":
    main()
