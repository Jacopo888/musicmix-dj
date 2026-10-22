"""Gate della palestra dei difetti (controllo obbligatorio per ogni aggiornamento della pipeline): ``-m gym``.

Rifà riferimenti e varianti con il codice corrente (``tools/defect_gym.py refs run report --profile gate``) in una
cartella temporanea e fallisce se una soglia di ``configs/defect_gym_gate.json`` non è rispettata. Dura decine di
minuti (render reali, Beat This!, Rubber Band): escluso dalla suite normale, che contiene la versione rapida
(``tests/test_defect_gym.py``). Con ``MUSICMIX_GYM_RUN=<cartella>`` ricontrolla solo il gate su una run già fatta.

    python -m pytest -m gym tests/test_defect_gym_gate.py -s
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.gym


def _selected(request) -> bool:
    return "gym" in (request.config.getoption("markexpr") or "")


@pytest.mark.data("runs/dj_set_v3_bridge/set.json")
def test_defect_gym_gate(tmp_path, request):
    if not _selected(request):
        pytest.skip("palestra completa: lanciare con -m gym")
    if not (ROOT / "runs/dj_set_v3_bridge/set.json").is_file():
        pytest.skip("run del set v3 assente su questa macchina")
    out = Path(os.environ.get("MUSICMIX_GYM_RUN") or tmp_path)
    stages = ["report"] if os.environ.get("MUSICMIX_GYM_RUN") else ["refs", "run", "report"]
    cmd = [sys.executable, str(ROOT / "tools/defect_gym.py"), *stages, "--profile", "gate", "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=ROOT)
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    bad = [r for r in rep["gate"]["rules"] if r["status"] != "pass"]
    msg = "; ".join(f"{r['rule'].get('variant') or 'falsi allarmi ' + r['rule'].get('group')}: {r['status']} "
                    f"(n={r.get('n')}, valore={r.get('rate')})" for r in bad)
    assert rep["gate"]["passed"], f"gate della palestra non rispettato: {msg} (dettagli in {out / 'REPORT.md'})"
