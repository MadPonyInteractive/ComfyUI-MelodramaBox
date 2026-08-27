"""Vendored DramaBox / LTX-2 inference code.

Pinned from https://github.com/resemble-ai/DramaBox @ a70a5818e103c1c9fef22409c1e0c707ebf4f8a7
(see ATTRIBUTION.md). Top-level packages are renamed to collision-proof
names so they can never shadow (or be shadowed by) a pip-installed
`ltx_core`/`ltx_pipelines` or another node pack's copy:

    ltx_core      -> mdb_ltx_core
    ltx_pipelines -> mdb_ltx_pipelines  (utils subset only)
    src/*.py      -> mdb_dramabox

The vendored code imports its own modules absolutely (``from mdb_ltx_core
...``), so this directory must be on sys.path. Importing this package
performs that insertion once, idempotently.
"""
import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = str(_Path(__file__).resolve().parent)
if _VENDOR_DIR not in _sys.path:
    _sys.path.insert(0, _VENDOR_DIR)
