"""Correzioni strutturate nei moduli d'ascolto (schema ``musicmix.dj-review.2``) e lettura dei giudizi esportati.

Un voto 1–10 su clip con lo stesso cue non può correggere un cue sbagliato (revisione esterna del 21/10/2026,
``conoscenza_mix/_meta/REVISIONE_ANDAMENTO_ASTRA_2026-10-21.md``): ogni modulo di voto ha quindi, oltre a
voto/note/tag, campi veloci che dicono *come* cambiare la transizione. Convenzioni (le stesse in tutte le pagine):

* ``a_exit``: uscita di A, spostata di ``-2/-1/+1/+2`` battute sulla griglia di A (``ok`` = giusta,
  ``fine_frase`` = alla fine della frase di A). Il contenuto di B allo switch non cambia: B arriva prima/dopo
  insieme all'uscita di A.
* ``b_entry``: punto d'ingresso nel brano B, spostato di ``-2/-1/+1/+2`` battute sulla griglia di B (``ok`` =
  giusto); l'uscita di A non cambia. ``-1`` = allo switch si sente una battuta di B in più (B parte da prima).
* ``b_pickup``: B deve entrare con il pickup (le sillabe prima del downbeat); ``b_section``: B deve entrare in
  un'altra sezione (``chorus``, ``verse``, ``intro``, ``breakdown``, ``altra``).
* ``wrong_b``: brano B sbagliato per questo momento (ripianificare con un altro brano); ``stay_longer_a``: restare
  su A più a lungo; ``technique``: tecnica sbagliata → quella indicata (chiavi ``type_key`` del runner).
* ``grid``: ``doppio_tempo`` / ``meta_tempo`` (la griglia di un brano è a tempo doppio o dimezzato);
  ``battute_sfasate`` resta il tag già esistente ed è riportato in ``grid_offbeat``.

Campo non compilato = ``None`` (non segnalato): non è un'approvazione esplicita. I documenti ``musicmix.dj-review.1``
(senza ``corrections``) restano leggibili: :func:`corrections_of` restituisce i campi vuoti con ``schema = 1``.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_REVIEW_2 = "musicmix.dj-review.2"
SCHEMA_REVIEW_1 = "musicmix.dj-review.1"

#: valori dello spostamento in battute (radio); ``ok`` = giusto così
SHIFT_CHOICES = (("-2", "−2"), ("-1", "−1"), ("ok", "ok"), ("+1", "+1"), ("+2", "+2"))
SHIFT_BARS = {"-2": -2, "-1": -1, "ok": 0, "+1": 1, "+2": 2}
A_EXIT_EXTRA = (("fine_frase", "alla fine della frase"),)
B_SECTIONS = (("chorus", "ritornello"), ("verse", "strofa"), ("intro", "intro"), ("breakdown", "breakdown/bridge"),
              ("altra", "altra sezione (nella nota)"))
GRID_CHOICES = (("doppio_tempo", "doppio tempo"), ("meta_tempo", "metà tempo"))
#: tecniche note (``type_key`` di dj_set + tecniche di dj_policy/dj_creative usate dal runner del set)
TECHNIQUES = (
    ("cut", "taglio sull'uno"), ("blend4", "blend 4 battute"), ("blend8", "blend 8 battute"),
    ("blend16", "blend 16 battute"), ("echo_out", "echo out"), ("filter_out", "filter out"),
    ("filter_into_chorus", "filter + ingresso sul ritornello"), ("backspin", "backspin"),
    ("build_into_chorus", "build verso il ritornello"), ("halving_build", "build a dimezzamento"),
    ("break_before_entry", "break prima dell'ingresso"), ("loop_extend", "loop di A"),
    ("acapella_over", "acapella sopra la base"), ("vocal_swap", "vocal swap"), ("echo_freeze", "echo freeze"),
    ("reverb_wash", "reverb wash"), ("note_bridge", "nota ponte"), ("tape_stop", "tape stop"),
    ("altra", "altra (nella nota)"),
)
TECHNIQUE_KEYS = tuple(k for k, _ in TECHNIQUES)
EMPTY = {"a_exit": None, "b_entry": None, "b_pickup": False, "b_section": None, "wrong_b": False,
         "stay_longer_a": False, "technique": None, "grid": None, "grid_offbeat": False}


# ------------------------------------------------------------------ normalizzazione
def corrections_of(doc: dict) -> dict:
    """Campi di correzione normalizzati di un giudizio (``.1`` → tutti vuoti, ``schema = 1``)."""
    raw = doc.get("corrections") if isinstance(doc.get("corrections"), dict) else None
    out = dict(EMPTY)
    out["schema"] = 2 if raw is not None else 1
    raw = raw or {}
    a = raw.get("a_exit")
    out["a_exit"] = str(a) if a is not None and str(a) in (*SHIFT_BARS, "fine_frase") else None
    b = raw.get("b_entry")
    out["b_entry"] = str(b) if b is not None and str(b) in SHIFT_BARS else None
    for k in ("b_pickup", "wrong_b", "stay_longer_a", "grid_offbeat"):
        out[k] = bool(raw.get(k))
    s = raw.get("b_section")
    out["b_section"] = s if s in {k for k, _ in B_SECTIONS} else None
    t = raw.get("technique")
    out["technique"] = str(t) if t else None
    g = raw.get("grid")
    out["grid"] = g if g in {k for k, _ in GRID_CHOICES} else None
    if "battute_sfasate" in (doc.get("tags") or []):
        out["grid_offbeat"] = True
    return out


def is_corrected(c: dict) -> bool:
    """True se almeno un campo chiede un cambiamento (``ok`` e campi vuoti non contano)."""
    return bool((c.get("a_exit") not in (None, "ok")) or (c.get("b_entry") not in (None, "ok")) or c.get("b_pickup")
                or c.get("b_section") or c.get("wrong_b") or c.get("stay_longer_a") or c.get("technique")
                or c.get("grid") or c.get("grid_offbeat"))


def describe(c: dict) -> list[str]:
    """Frasi brevi in italiano (per report e pagine)."""
    out = []
    if c.get("a_exit") == "fine_frase":
        out.append("A esce alla fine della frase")
    elif c.get("a_exit") not in (None, "ok"):
        out.append(f"uscita di A {c['a_exit']} battute")
    if c.get("b_entry") not in (None, "ok"):
        out.append(f"ingresso di B {c['b_entry']} battute")
    if c.get("b_pickup"):
        out.append("B entra con il pickup")
    if c.get("b_section"):
        out.append(f"B in un'altra sezione: {dict(B_SECTIONS).get(c['b_section'], c['b_section'])}")
    if c.get("wrong_b"):
        out.append("brano B sbagliato per questo momento")
    if c.get("stay_longer_a"):
        out.append("restare su A più a lungo")
    if c.get("technique"):
        out.append(f"tecnica → {dict(TECHNIQUES).get(c['technique'], c['technique'])}")
    if c.get("grid"):
        out.append(dict(GRID_CHOICES)[c["grid"]])
    if c.get("grid_offbeat"):
        out.append("battuta sfasata")
    return out


# ------------------------------------------------------------------ lettura giudizi
def read_docs(paths: Iterable[Path]) -> list[dict]:
    """Tutti i documenti di giudizio in cartelle/file JSON: singoli doc o esportazioni "Copia JSON".

    Ogni doc riceve ``_file`` e ``_file_sha256`` (provenienza); i file non sono mai modificati.
    """
    docs: list[dict] = []
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files += sorted(p.rglob("*.json")) if p.is_dir() else [p]
    for f in files:
        try:
            raw = f.read_bytes()
            d = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        sha = hashlib.sha256(raw).hexdigest()
        if isinstance(d, dict) and isinstance(d.get("judgments"), dict):
            for did, body in d["judgments"].items():
                if isinstance(body, dict):
                    docs.append({"doc_id": did, **body, "_file": str(f), "_file_sha256": sha})
        elif isinstance(d, dict) and (d.get("doc_id") or d.get("clip_id") or d.get("type")):
            docs.append({**d, "_file": str(f), "_file_sha256": sha})
    return docs


def latest(docs: Iterable[dict]) -> dict[str, dict]:
    """Un documento per ``doc_id`` (l'ultimo salvato)."""
    best: dict[str, dict] = {}
    for d in docs:
        k = str(d.get("doc_id") or d.get("clip_id") or "")
        if not k:
            continue
        if k not in best or str(d.get("saved_at", "")) >= str(best[k].get("saved_at", "")):
            best[k] = d
    return best


# ------------------------------------------------------------------ HTML / JS condivisi
def _radios(name: str, choices, corr: str) -> str:
    return "".join(f'<label class="chip"><input type="radio" name="{html.escape(name)}" value="{html.escape(k)}" '
                   f'data-corr="{corr}"><span>{html.escape(v)}</span></label>' for k, v in choices)


def _flag(key: str, label: str) -> str:
    return (f'<label class="chip"><input type="checkbox" data-corr-flag="{key}"><span>{html.escape(label)}</span>'
            f'</label>')


def fieldset_html(prefix: str, open_: bool = False) -> str:
    """Blocco ``<details>`` con i campi di correzione; ``prefix`` rende unici i nomi dei radio nella pagina."""
    p = html.escape(prefix)
    techs = '<option value="">—</option>' + "".join(f'<option value="{html.escape(k)}">{html.escape(v)}</option>'
                                                    for k, v in TECHNIQUES)
    secs = '<option value="">—</option>' + "".join(f'<option value="{html.escape(k)}">{html.escape(v)}</option>'
                                                   for k, v in B_SECTIONS)
    return f"""<details class="corr"{' open' if open_ else ''}>
  <summary>Correzioni (facoltative)</summary>
  <p class="small muted hint">Battute sulla griglia del brano: −1 = una battuta prima, +1 = una dopo. «Ingresso di B» è il punto nel brano B (con −1 allo switch si sente una battuta di B in più); per far arrivare B prima nel mix sposta l'uscita di A.</p>
  <fieldset><legend>Uscita di A</legend><div class="chips">{_radios(p + '-a_exit', SHIFT_CHOICES + A_EXIT_EXTRA, 'a_exit')}</div></fieldset>
  <fieldset><legend>Ingresso di B</legend><div class="chips">{_radios(p + '-b_entry', SHIFT_CHOICES, 'b_entry')}{_flag('b_pickup', 'B deve entrare con il pickup')}</div>
    <label class="sel small">B in un'altra sezione <select data-corr-select="b_section">{secs}</select></label></fieldset>
  <fieldset><legend>Scelta</legend><div class="chips">{_flag('wrong_b', 'brano B sbagliato per questo momento')}{_flag('stay_longer_a', 'resta su A più a lungo')}</div>
    <label class="sel small">Tecnica sbagliata → <select data-corr-select="technique">{techs}</select></label></fieldset>
  <fieldset><legend>Griglia</legend><div class="chips">{_radios(p + '-grid', GRID_CHOICES, 'grid')}</div>
    <p class="small muted">«battuta sfasata» è tra le scelte rapide.</p></fieldset>
  <button type="button" class="small" data-corr-clear>azzera correzioni</button>
</details>"""


#: funzioni JS: ``readCorr(root)``, ``fillCorr(root, c)``, ``corrText(c)``, ``corrUsed(c)``; ``root`` = il form
JS = r"""
const CORR_LABELS = __CORR_LABELS__;
function readCorr(root){
  const r = n => { const c = root.querySelector('input[data-corr="' + n + '"]:checked'); return c ? c.value : null; };
  const f = n => { const c = root.querySelector('input[data-corr-flag="' + n + '"]'); return !!(c && c.checked); };
  const s = n => { const c = root.querySelector('select[data-corr-select="' + n + '"]'); return c && c.value ? c.value : null; };
  const tags = Array.from(root.querySelectorAll('input[type=checkbox][value="battute_sfasate"]:checked'));
  return {a_exit: r("a_exit"), b_entry: r("b_entry"), b_pickup: f("b_pickup"), b_section: s("b_section"),
          wrong_b: f("wrong_b"), stay_longer_a: f("stay_longer_a"), technique: s("technique"), grid: r("grid"),
          grid_offbeat: tags.length > 0};
}
function fillCorr(root, c){
  c = c || {};
  root.querySelectorAll("input[data-corr]").forEach(i => { i.checked = c[i.dataset.corr] === i.value; });
  root.querySelectorAll("input[data-corr-flag]").forEach(i => { i.checked = !!c[i.dataset.corrFlag]; });
  root.querySelectorAll("select[data-corr-select]").forEach(s => { s.value = c[s.dataset.corrSelect] || ""; });
  const d = root.querySelector("details.corr"); if (d && corrUsed(c)) d.open = true;
}
function corrUsed(c){
  c = c || {};
  return !!((c.a_exit && c.a_exit !== "ok") || (c.b_entry && c.b_entry !== "ok") || c.b_pickup || c.b_section ||
            c.wrong_b || c.stay_longer_a || c.technique || c.grid || c.grid_offbeat);
}
function corrText(c){
  if (!c) return "";
  const L = CORR_LABELS, out = [];
  if (c.a_exit === "fine_frase") out.push("A esce a fine frase"); else if (c.a_exit && c.a_exit !== "ok") out.push("uscita A " + c.a_exit);
  if (c.b_entry && c.b_entry !== "ok") out.push("ingresso B " + c.b_entry);
  if (c.b_pickup) out.push("pickup di B");
  if (c.b_section) out.push("B: " + (L.sections[c.b_section] || c.b_section));
  if (c.wrong_b) out.push("B sbagliato");
  if (c.stay_longer_a) out.push("più A");
  if (c.technique) out.push("tecnica → " + (L.techniques[c.technique] || c.technique));
  if (c.grid) out.push(L.grid[c.grid] || c.grid);
  return out.join(", ");
}
document.querySelectorAll("[data-corr-clear]").forEach(b => b.addEventListener("click", () => {
  const form = b.closest("form"); fillCorr(form, {});
  form.querySelectorAll('input[type=checkbox][value="battute_sfasate"]').forEach(i => { i.checked = false; });
  form.dispatchEvent(new Event("input", {bubbles: true}));
}));
"""

CSS = """details.corr{border:1px dashed var(--border);border-radius:8px;padding:.3rem .6rem;margin:.5rem 0}
details.corr summary{cursor:pointer;font-weight:600;font-size:.9rem;min-height:32px;display:flex;align-items:center}
details.corr .hint{margin:.2rem 0 .4rem}
.sel{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:.4rem 0 0}
.sel select{font:inherit;max-width:100%;padding:.3rem;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text)}
"""


def js() -> str:
    labels = {"sections": dict(B_SECTIONS), "techniques": dict(TECHNIQUES), "grid": dict(GRID_CHOICES)}
    payload = json.dumps(labels, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    return JS.replace("__CORR_LABELS__", payload)


def bars_shift(value: Optional[str]) -> Optional[int]:
    """``"-1"`` → -1, ``"ok"`` → 0, ``None``/``"fine_frase"`` → None."""
    return SHIFT_BARS.get(value) if value is not None else None


def summary_counts(docs: Iterable[dict]) -> dict[str, Any]:
    """Conteggi rapidi dei campi usati (diagnostica)."""
    n = {"docs": 0, "schema2": 0, "corrected": 0}
    for d in docs:
        c = corrections_of(d)
        n["docs"] += 1
        n["schema2"] += c["schema"] == 2
        n["corrected"] += is_corrected(c)
    return n
