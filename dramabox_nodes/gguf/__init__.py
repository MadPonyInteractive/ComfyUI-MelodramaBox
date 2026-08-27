"""GGUF support for the DramaBox audio DiT.

DramaBox's `dramabox-dit-v1.safetensors` IS the already-separated,
fine-tuned LTX-2.3 audio branch (3.3B) — so producing a GGUF of it is a
straight quantization of that file (convert.py), not an attempt to carve
audio tensors out of the 22B audio+video GGUFs (which is unsupported and
undocumented). No GGUF of this checkpoint exists upstream; convert.py
creates it.

Runtime loading of the resulting .gguf reuses city96's ComfyUI-GGUF ops
(GGMLTensor / GGMLOps) when that pack is installed — see loader.py.
"""
