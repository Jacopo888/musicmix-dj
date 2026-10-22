"""TIV.lib (Ramires, Bernardes, Davies, Serra, DAFx 2020), MIT License, vendored unmodified.

Source: https://github.com/aframires/TIVlib (branch master, file TIVlib/tiv.py), fetched 2026-10-21;
provenance and hashes in PROVENANCE.json. Only ``tiv.py`` and ``LICENSE`` are copied; the package
``__init__`` of upstream (``from .tiv import *``) is reproduced here. ``tiv.py`` imports matplotlib at
module level (plotting helpers): the import is deferred to the first use through :func:`load`.
"""


def load():
    """Import and return the vendored ``tiv`` module (matplotlib is imported with it)."""
    from musicmix._vendor.tivlib import tiv

    return tiv
