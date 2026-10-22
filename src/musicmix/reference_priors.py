"""Prior appresi da set DJ di riferimento (``musicmix.reference-priors.1``), letti in modo additivo.

Il file ``configs/reference_priors.json`` è facoltativo: se manca, :func:`load_reference_priors`
restituisce ``None`` e :func:`prior_cost` vale sempre 0, quindi il comportamento del
pianificatore resta quello di default. Il file è prodotto da ``tools/reference_mix_stats.py``
a partire da mix analizzati (``musicmix.reference-mix.1``) e, facoltativamente, dai metadati
pubblici Raveform (elettronica, dominio diverso dal rap: peso basso).

Uso (adattatore, da chiamare dal pianificatore quando qualcuno lo collegherà; ``dj_set`` e
``dj_select`` non sono modificati qui)::

    pri = load_reference_priors()            # None se il file non c'è
    cost, parts = prior_cost(pri, technique="blend", overlap_beats=8,
                             entry_bar_in_phrase16=0, entry_section="verse")

Il costo è ``-Σ_d w_d · log(p_d(x) / u_d)`` per ogni grandezza fornita: ``p_d`` è la frequenza
del dominio ``d`` con lisciamento di Laplace (α = 1; con pochi dati resta vicina all'uniforme
``u_d``, quindi il costo tende a 0), ``w_d`` il peso del dominio. È una *distribuzione* di
scelte osservate, non una regola: valori rari costano un po' di più, nessuno è vietato.
Le etichette di sezione si usano solo se appartengono al vocabolario del dominio (le sezioni
EDM di Raveform — drop, breakdown — non sono equiparate a ritornello/strofa del rap).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCHEMA = "musicmix.reference-priors.1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "configs" / "reference_priors.json"

# grandezza -> (tipo, chiave nella distribuzione)
FEATURES = {
    "technique": "categorical",
    "overlap_beats": "binned",
    "entry_bar_in_phrase16": "categorical",
    "exit_bar_in_phrase16": "categorical",
    "entry_beats_from_downbeat": "binned",
    "exit_beats_from_downbeat": "binned",
    "entry_section": "categorical",
    "exit_section": "categorical",
    "entry_track_fraction": "binned",
    "exit_track_fraction": "binned",
    "tempo_change_pct": "binned",
    "play_beats": "binned",
    "entry_bars_to_chorus": "binned",
    "exit_prev_section": "categorical",
    "exit_bars_to_section_end": "binned",
    "base_continuous": "categorical",
    "b_drop_vs_a_exit_beats": "binned",
}


@dataclass
class ReferencePriors:
    data: dict
    path: Optional[str] = None

    @property
    def domains(self) -> dict:
        return self.data.get("domains", {})

    def log_ratio(self, domain: str, feature: str, value) -> Optional[float]:
        """log(p/u) del valore nel dominio; None se il dominio non ha la grandezza o il valore è fuori vocabolario."""
        dist = (self.domains.get(domain, {}).get("distributions") or {}).get(feature)
        if not dist or value is None:
            return None
        alpha = float(self.data.get("smoothing_alpha", 1.0))
        if FEATURES.get(feature) == "binned" or "bins" in dist:
            edges, counts = dist["bins"], dist["counts"]
            k = len(counts)
            if k == 0:
                return None
            v = float(value)
            idx = None
            for i in range(k):
                lo, hi = edges[i], edges[i + 1]
                if (lo <= v < hi) or (i == k - 1 and v == hi):
                    idx = i
                    break
            if idx is None:  # fuori dall'intervallo osservato: bin estremo più vicino
                idx = 0 if v < edges[0] else k - 1
            c = counts[idx]
        else:
            counts = dist["counts"]
            k = len(counts)
            key = str(value)
            if key not in counts:
                return None  # vocabolario diverso: niente costo
            c = counts[key]
        n = float(sum(counts.values()) if isinstance(counts, dict) else sum(counts))
        p = (c + alpha) / (n + alpha * k)
        return math.log(p * k)


def load_reference_priors(path: Optional[str | Path] = None) -> Optional[ReferencePriors]:
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA:
        raise ValueError(f"{p}: schema {data.get('schema_version')!r}, atteso {SCHEMA}")
    return ReferencePriors(data=data, path=str(p))


def prior_cost(priors: Optional[ReferencePriors], *, domains: Optional[list] = None, **features) -> tuple[float, dict]:
    """Costo additivo (0 senza prior) e dettaglio per dominio e grandezza."""
    if priors is None:
        return 0.0, {}
    total, parts = 0.0, {}
    for d, spec in priors.domains.items():
        if domains is not None and d not in domains:
            continue
        w = float(spec.get("weight", 0.0))
        if w <= 0:
            continue
        for f, v in features.items():
            if f not in FEATURES:
                raise KeyError(f"grandezza sconosciuta: {f}")
            lr = priors.log_ratio(d, f, v)
            if lr is None:
                continue
            c = -w * lr
            parts.setdefault(d, {})[f] = round(c, 4)
            total += c
    return total, parts


def transition_features(tr: dict) -> dict:
    """Grandezze di una transizione ``musicmix.reference-mix.1`` nel vocabolario dei prior."""
    be = (tr.get("b_entry") or {}).get("at_switch") or {}
    ax = (tr.get("a_exit") or {}).get("at_switch") or {}
    tempo = tr.get("tempo") or {}
    st = tempo.get("b_tempo_change_pct")
    return {
        "technique": tr.get("technique"),
        "overlap_beats": tr.get("overlap_beats"),
        "entry_bar_in_phrase16": be.get("bar_in_phrase16"),
        "exit_bar_in_phrase16": ax.get("bar_in_phrase16"),
        "entry_beats_from_downbeat": be.get("beats_from_downbeat"),
        "exit_beats_from_downbeat": ax.get("beats_from_downbeat"),
        "entry_section": be.get("section"),
        "exit_section": ax.get("section"),
        "entry_track_fraction": be.get("track_fraction"),
        "exit_track_fraction": ax.get("track_fraction"),
        "tempo_change_pct": st,
        # "A carica, B risolve" / "la base non deve interrompersi" (voti ciechi dell'utente): descrittori
        "entry_bars_to_chorus": be.get("bars_to_next_chorus"),
        "exit_prev_section": ax.get("prev_section"),
        "exit_bars_to_section_end": ax.get("bars_to_section_end"),
        "base_continuous": None if tr.get("overlap_beats") is None else bool(tr["overlap_beats"] >= -0.5),
    }
