"""Where MusicMix finds its code, data, model weights and helper interpreters.

Every location can be overridden with an environment variable; the defaults
reproduce the layout of the reference Linux setup (``/opt/musicmix``):

=====================  ==========================================  ====================================
variable               meaning                                     default
=====================  ==========================================  ====================================
``MUSICMIX_ROOT``      checkout (code, configs)                    this package's repository
``MUSICMIX_DATA``      run outputs, pools, local audio             ``MUSICMIX_ROOT``
``MUSICMIX_HOME``      installation prefix (venvs, models)         ``/opt/musicmix``
``MUSICMIX_MODELS``    pretrained weights                          ``$MUSICMIX_HOME/models``
``MUSICMIX_PY_MAIN``   main interpreter (pipeline, Demucs, ASR)    ``$MUSICMIX_HOME/venv/bin/python``
                                                                   or the running interpreter
``MUSICMIX_PY_NP1``    numpy<2 interpreter (CUE-DETR, MuQ, ...)    ``$MUSICMIX_HOME/venv-np1/bin/python``
``MUSICMIX_PY_ALLIN1`` All-In-One / madmom / essentia interpreter  ``$MUSICMIX_HOME/venv-allin1/bin/python``
``MUSICMIX_EXTERNAL``  sibling clones of external corpora          parent folder of ``MUSICMIX_ROOT``
``MUSICMIX_SCRATCH``   throw-away outputs (listening pages, ...)   ``$MUSICMIX_DATA/.cache/scratch``
``MUSICMIX_REPO``      GitHub ``owner/name`` for REST helpers      none: tools that need it must ask
``TORCH_HOME``         torch hub cache                             ``$MUSICMIX_MODELS/torch``
=====================  ==========================================  ====================================

Saved run files keep the paths of the machine that wrote them (Windows drive
paths or WSL ``/mnt/<drive>`` paths).  Those JSONs are hashed and must not be
edited, so readers localise the recorded value at open time (:func:`localize`).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Union

PathLike = Union[str, "os.PathLike[str]"]
# Workspace folder names -> where their contents live relative to the repo root.
ANCHORS = {"music_mixing_progetto": "root", "musicmix_cloud": "root", "giudizi json": "sibling"}
_FOREIGN = re.compile(r"^(?:[A-Za-z]:/|/mnt/[A-Za-z]/)")


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def repo_root() -> Path:
    env = os.environ.get("MUSICMIX_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Run outputs, pools and imported audio (``runs/``, ``local_audio/``)."""
    return _env_path("MUSICMIX_DATA", repo_root())


def install_home() -> Path:
    return _env_path("MUSICMIX_HOME", Path("/opt/musicmix"))


def models_dir() -> Path:
    return _env_path("MUSICMIX_MODELS", install_home() / "models")


def torch_home() -> Path:
    return _env_path("TORCH_HOME", models_dir() / "torch")


def external_dir() -> Path:
    """Folder holding sibling clones of external corpora (MCFlow, flowBook, books)."""
    return _env_path("MUSICMIX_EXTERNAL", repo_root().parent)


def scratch_dir(name: str = "") -> Path:
    base = _env_path("MUSICMIX_SCRATCH", data_root() / ".cache" / "scratch")
    return base / name if name else base


def _venv_python(variable: str, venv: str, fallback: str | None) -> str:
    value = os.environ.get(variable)
    if value:
        # Accept either an interpreter or a venv folder (``$MUSICMIX_PY_NP1=/opt/musicmix/venv-np1``).
        path = Path(value).expanduser()
        return str(path / "bin" / "python") if path.is_dir() else str(path)
    candidate = install_home() / venv / "bin" / "python"
    if candidate.exists() or fallback is None:
        return str(candidate)
    return fallback


def python_main() -> str:
    """Interpreter with the core pipeline (Beat This!, Demucs, faster-whisper)."""
    return _venv_python("MUSICMIX_PY_MAIN", "venv", sys.executable)


def python_np1() -> str:
    """Interpreter with numpy<2 (CUE-DETR, SongEval/MuQ, MusicGen)."""
    return _venv_python("MUSICMIX_PY_NP1", "venv-np1", None)


def python_allin1() -> str:
    """Interpreter with All-In-One, madmom and essentia."""
    return _venv_python("MUSICMIX_PY_ALLIN1", "venv-allin1", None)


def github_repo() -> str:
    """``owner/name`` of the GitHub repository used by REST helpers; no default on purpose."""
    value = os.environ.get("MUSICMIX_REPO", "").strip()
    if not value or "/" not in value:
        raise RuntimeError("set MUSICMIX_REPO=owner/name to use the GitHub helpers")
    return value


def localize(recorded: PathLike, root: PathLike | None = None) -> Path:
    """Return ``recorded`` if it exists, else its equivalent under this checkout.

    The remainder after the last workspace anchor is joined onto the repo root
    (``giudizi json`` onto the root's parent).  Unmatched paths are returned
    unchanged so callers' missing-file handling still applies.
    """
    path = Path(recorded)
    if path.exists():
        return path
    text = os.fspath(recorded).replace("\\", "/")
    if not _FOREIGN.match(text):
        # A relative Windows path keeps its meaning once separators are fixed.
        return path if path.is_absolute() else Path(text)
    parts = text.split("/")
    hits = [index for index, part in enumerate(parts) if part.lower() in ANCHORS]
    if not hits:
        return path
    anchor = hits[-1]
    base = Path(root) if root is not None else repo_root()
    if ANCHORS[parts[anchor].lower()] == "sibling":
        base = base.parent / "giudizi json"
    return base.joinpath(*[part for part in parts[anchor + 1:] if part])
