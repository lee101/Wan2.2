"""Filesystem locations used by the optimloras scripts.

Neither the downloaded Wan2.2 weights nor the VideoX-Fun checkout live in
this repository. Each location is read from an environment variable when
set and otherwise falls back to a path relative to this checkout, so the
scripts run from a fresh clone without needing edits.

Environment variables:
    VIDEOX_FUN_DIR  Path to the VideoX-Fun checkout (provides `videox_fun`).
    WAN_MODEL_DIR   Path to the downloaded Wan2.2 weights.
    VIDEOX_FUN_CONFIG  Path to the pipeline YAML under VideoX-Fun.
"""

import os
import sys
from pathlib import Path
from typing import List

# optimloras/paths.py -> the repository root is its parent.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Optional separate checkout; provides the `videox_fun` package.
VIDEOX_FUN_DIR = Path(
    os.environ.get("VIDEOX_FUN_DIR", str(REPO_ROOT / "VideoX-Fun"))
)

# Downloaded Wan2.2 weights. See the top-level README for download links.
DEFAULT_MODEL_DIR = Path(
    os.environ.get("WAN_MODEL_DIR", str(REPO_ROOT / "Wan2.2-I2V-A14B"))
)

DEFAULT_CONFIG_PATH = Path(
    os.environ.get(
        "VIDEOX_FUN_CONFIG",
        str(VIDEOX_FUN_DIR / "config" / "wan2.2" / "wan_civitai_i2v.yaml"),
    )
)


def add_videox_fun_to_path() -> str:
    """Put the VideoX-Fun checkout on sys.path if it is present.

    Returns the path that was added, whether or not it exists, so callers
    can report a useful error.
    """
    path = str(VIDEOX_FUN_DIR)
    if path not in sys.path:
        sys.path.append(path)
    return path


def candidate_model_dirs() -> List[Path]:
    """Likely locations of the Wan2.2 weights, most specific first."""
    return [
        DEFAULT_MODEL_DIR,
        REPO_ROOT / "Wan2.2T2V-A14B",
        REPO_ROOT / "Wan2.2-TI2V-5B",
    ]