#!/usr/bin/env python3
"""Correzioni strutturate dei giudizi → piano corretto del set, passaggi di riferimento, ri-render mirato.

Legge i giudizi esportati (cartelle di JSON come ``giudizi_cloud/<run>/judgments``: un file per documento o
l'esportazione "Copia JSON") delle pagine di :mod:`build_listening_artifact`:

* approvazione del set (``type = set_approval_transizione``, ``musicmix.set-approval.1``): verdetto + correzioni;
* galleria a versioni ``musicmix.dj-review.2`` (``type = versione`` con ``cue`` e ``corrections``);
* torneo dei cue (``type = torneo`` e ``torneo_variante``, ``musicmix.cue-tournament.1``): la variante migliore
  dà il cue, le correzioni della variante si applicano sopra.

e il ``set.json`` a cui si riferiscono (letto con ``musicmix.paths.localize``, mai modificato). Per ogni transizione
traduce le correzioni in modifiche del piano sulla griglia dei brani (analisi ``dj_select`` di ``sources[slug]``):

* uscita di A ``±k`` battute → ``a_out_bar + k``; ``fine_frase`` → primo downbeat dopo la fine del verso di A che
  attraversa l'uscita (modello ``dj_phrase`` della run, ASR affidabile, entro 8 battute) o, senza modello, la prima
  battuta successiva con ``phrase_level >= 4`` [H]; ``resta su A`` → prima battuta ``>= a_out_bar + 8`` con
  ``phrase_level >= 8`` [H];
* ingresso di B ``±k`` → ``b_in_bar + k``; sezione → inizio di sezione di B con quell'etichetta più vicino al cue
  (pareggio: il primo); pickup → lead-in ``dj_phrase.b_lead_in`` (o ``entry_point``) sul nuovo downbeat; se il
  modello vede B a metà frase la richiesta va in ``needs_manual``; senza modello un battito [H];
* tecnica → ``type_key`` richiesto; brano B sbagliato / "da rifare con altro brano" → coppia vietata (ripianificazione
  dell'ordine); doppio/metà tempo e battuta sfasata → controllo manuale della griglia (non correggibile qui).

Scrive ``corrections_plan.json`` (``musicmix.corrections-plan.1``) con, per transizione, originale, correzione,
stato (``approvata``, ``corretta``, ``ripianifica``, ``manuale``, ``senza_giudizio``, ``senza_correzioni``) e
decisioni (ordine, uscita, ingresso, tecnica: approvate o spostate di quante battute; usate da
:mod:`decision_agreement`), più la sezione ``runner``: cue fissati (``pins``), coppie vietate, transizioni da
ri-renderizzare e transizioni congelate riutilizzabili. Il runner del set (``tools/run_dj_set_v2_books.py``,
``SetHooks.corrections``) usa solo questa sezione; ``rerender`` lancia il set v3 con il piano.

Passaggi di riferimento (``--references DIR``, schema ``musicmix.reference-transition.1``): solo transizioni
verificate all'ascolto dall'utente — verdetto ``approvata`` senza correzioni nella pagina di approvazione (se il set
era un ri-render corretto, la correzione diventa così "approvata") e variante migliore di un torneo dei cue. Le
correzioni proposte ma non ancora riascoltate restano nel piano, non diventano riferimenti. I riferimenti sono dati di
**verifica**: non vanno usati per tarare soglie o regole (``use: "verifica"``).

    python tools/apply_corrections.py plan --set runs/dj_set_v3_bridge/set.json \\
        --judgments giudizi_cloud/dj_set_v3_bridge/approvazione --out runs/dj_set_v3_bridge_corr/corrections_plan.json \\
        --references giudizi_cloud/reference_transitions
    python tools/apply_corrections.py rerender --plan runs/.../corrections_plan.json \\
        --out runs/dj_set_v3_bridge_corr1 [--plan-only]
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from musicmix.paths import localize  # noqa: E402
import listening_corrections as LC  # noqa: E402

SCHEMA_PLAN = "musicmix.corrections-plan.1"
SCHEMA_REF = "musicmix.reference-transition.1"
SR = 48000
PHRASE_END_MAX_BARS = 8       # [H] la fine del verso deve cadere entro 8 battute dall'uscita
STAY_LONGER_BARS = 8          # [H] "resta su A" = almeno una frase da 8 in più
LINE_EXCERPT_CHARS = 60       # solo un estratto del verso (testi protetti): bastano le coordinate in campioni
SECTION_ALIASES = {"chorus": ("chorus", "hook"), "verse": ("verse",), "intro": ("intro",),
                   "breakdown": ("break", "breakdown", "bridge", "inst", "solo")}
SOURCE_PRIORITY = {"cue_tournament": 3, "set_approval": 2, "gallery": 1}


# --------------------------------------------------------------------------- utilità
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def resolve_path(recorded: str, base: Path | None = None) -> Path:
    p = localize(recorded)
    if not p.is_absolute():
        for cand in ([ROOT / p] + ([base / p] if base else [])):
            if cand.exists():
                return cand
        return ROOT / p
    return p


class Context:
    """Set congelato, analisi dei brani e modelli ``dj_phrase`` (sola lettura, caricati a richiesta)."""

    def __init__(self, set_path: Path):
        self.set_path = Path(set_path)
        self.raw = self.set_path.read_bytes()
        self.doc = json.loads(self.raw.decode("utf-8"))
        self.sha = hashlib.sha256(self.raw).hexdigest()
        self.run = self.doc.get("run") or self.set_path.parent.name
        self.dir = self.set_path.parent
        self.transitions = {int(t["index"]): t for t in self.doc.get("transitions") or []}
        self._an: dict[str, dict] = {}
        self._models: dict[str, Optional[dict]] = {}

    def analysis(self, slug: str) -> Optional[dict]:
        if slug not in self._an:
            src = (self.doc.get("sources") or {}).get(slug) or {}
            p = resolve_path(src["analysis"], self.dir) if src.get("analysis") else self.dir / "dj_plan" / "tracks" / f"{slug}.json"
            self._an[slug] = load_json(p) if p.is_file() else None
            if self._an[slug] is not None:
                self._an[slug]["_path"] = rel(p)
                self._an[slug]["_sha256"] = sha256_file(p)
        return self._an[slug]

    def model(self, slug: str) -> Optional[dict]:
        if slug not in self._models:
            p = self.dir / "phrase" / f"{slug}.json"
            m = None
            if p.is_file():
                d = load_json(p)
                m = d.get("model") if isinstance(d.get("model"), dict) else None
                if m is not None:
                    m = dict(m, _path=rel(p), _sha256=sha256_file(p))
            self._models[slug] = m
        return self._models[slug]


# --------------------------------------------------------------------------- griglia
def downbeats(A: dict) -> list[float]:
    return list((A.get("grid") or {}).get("downbeats_s") or [r["time_s"] for r in A.get("phrases") or []])


def bar_time(A: dict, bar: int) -> Optional[float]:
    db = downbeats(A)
    return float(db[bar]) if 0 <= bar < len(db) else None


def bar_of_time(A: dict, t: float) -> Optional[int]:
    db = downbeats(A)
    if not db:
        return None
    return int(min(range(len(db)), key=lambda i: abs(db[i] - t)))


def phrase_level(A: dict, bar: int) -> int:
    rows = A.get("phrases") or []
    return int(rows[bar].get("phrase_level") or 0) if 0 <= bar < len(rows) else 0


def phrase_end_bar(A: dict, bar: int, model: Optional[dict]) -> tuple[Optional[int], str]:
    """Battuta d'uscita "alla fine della frase" di A [H] (vedi docstring del modulo)."""
    t = bar_time(A, bar)
    if model and t is not None and (model.get("asr_quality") or {}).get("reliable", True):
        sr = int(model.get("sample_rate") or SR)
        x = int(round(t * sr))
        words = model.get("words") or []
        for ln in model.get("lines") or []:
            end = int(ln.get("end_sample") or 0)
            if end <= x - int(0.05 * sr) or float(ln.get("p_end") or 0) < 0.5:
                continue
            db = downbeats(A)
            nxt = next((j for j in range(bar, len(db)) if db[j] * sr >= end), None)
            if nxt is not None and nxt != bar and nxt - bar <= PHRASE_END_MAX_BARS:
                txt = ln.get("text") or " ".join(w.get("text", "") for w in words[ln.get("first_word", 0):ln.get("last_word", -1) + 1])
                return nxt, f"dj_phrase: fine del verso «{txt}» (p = {float(ln.get('p_end') or 0):.2f}), primo downbeat dopo"
            break
    rows = A.get("phrases") or []
    for j in range(bar + 1, len(rows)):
        if phrase_level(A, j) >= 4:
            return j, "griglia: prima battuta successiva con phrase_level >= 4 [H]"
    return None, "nessuna fine di frase dopo l'uscita"


def stay_longer_bar(A: dict, bar: int) -> tuple[Optional[int], str]:
    rows = A.get("phrases") or []
    for j in range(bar + STAY_LONGER_BARS, len(rows)):
        if phrase_level(A, j) >= 8:
            return j, f"resta su A: prima battuta >= +{STAY_LONGER_BARS} con phrase_level >= 8 [H]"
    j = bar + STAY_LONGER_BARS
    return (j, f"resta su A: +{STAY_LONGER_BARS} battute [H]") if j < len(rows) else (None, "A finisce prima")


def section_bar(B: dict, label: str, near: int) -> tuple[Optional[int], str]:
    names = SECTION_ALIASES.get(label)
    if not names:
        return None, f"sezione «{label}» non riconoscibile: vedi la nota"
    starts = [int(s["bar_index"]) for s in B.get("sections") or []
              if any(str(s.get("label", "")).lower().startswith(n) for n in names) and s.get("bar_index") is not None]
    starts = [j for j in starts if j != near] or starts
    if not starts:
        return None, f"B non ha sezioni «{label}» nell'analisi"
    j = min(starts, key=lambda x: (abs(x - near), x))
    return j, f"inizio di sezione «{label}» più vicino (battuta {j})"


def pickup_samples(model: Optional[dict], B: dict, t_b: float) -> tuple[int, str]:
    period = float((B.get("tempo") or {}).get("period_s") or 0.5)
    if model:
        from musicmix import dj_phrase as P
        try:
            li = P.b_lead_in(model, t_b)
            if li.get("status") == "lead_in" and int(li.get("lead_in_samples") or 0) > 0:
                return int(li["lead_in_samples"]), "dj_phrase.b_lead_in: " + "; ".join(li.get("reasons") or [])
            ep = P.entry_point(model, t_b)
            if int(ep.get("pickup_samples") or 0) > 0:
                return int(ep["pickup_samples"]), "dj_phrase.entry_point: " + "; ".join(ep.get("reasons") or [])
            # il modello vede B a metà frase o senza pickup: un lead-in imposto troncherebbe una parola (QA b_entry)
            return 0, "pickup non trovato da dj_phrase (" + str(li.get("status")) + "): " + "; ".join(
                (li.get("reasons") or []) + (ep.get("reasons") or []))
        except Exception as exc:  # il modello è solo un aiuto: si ripiega su un battito
            return int(round(period * SR)), f"pickup di un battito [H] (dj_phrase non disponibile: {exc})"
    return int(round(period * SR)), "pickup di un battito [H]: nessun modello dj_phrase per B"


# --------------------------------------------------------------------------- giudizi → voci per transizione
def _set_cue(tr: dict) -> dict:
    p = tr.get("plan") or {}
    return {"a_out_bar": p.get("a_out_bar"), "b_in_bar": p.get("b_in_bar"), "a_out_s": p.get("a_out_s"),
            "b_in_s": p.get("b_in_s"), "type_key": tr.get("type"), "b_lead_in_samples": 0}


def collect_entries(ctx: Context, docs: list[dict], run: Optional[str] = None) -> tuple[dict[int, list[dict]], list[dict], dict]:
    """Voci per indice di transizione, documenti ignorati (con motivo), giudizio complessivo del set (se c'è)."""
    run = run or ctx.run
    by_pair = {(t["a"], t["b"]): i for i, t in ctx.transitions.items()}
    entries: dict[int, list[dict]] = {}
    ignored: list[dict] = []
    set_doc: dict = {}
    variants = {(d.get("tournament_id"), d.get("variant_id")): d for d in LC.latest(
        [d for d in docs if d.get("type") == "torneo_variante"]).values()}
    for d in LC.latest(docs).values():
        typ = d.get("type", "versione")
        did = d.get("doc_id") or d.get("clip_id")

        def skip(reason):
            ignored.append({"doc_id": did, "type": typ, "reason": reason, "file": d.get("_file")})

        if typ == "set_approval":
            if d.get("run") == run:
                set_doc = d
            else:
                skip("run_diversa")
            continue
        if typ == "set_approval_transizione":
            if d.get("run") != run:
                skip("run_diversa"); continue
            i = int(d.get("transition_index") or 0)
            tr = ctx.transitions.get(i)
            if tr is None or (tr["a"], tr["b"]) != (d.get("a"), d.get("b")):
                skip("transizione_non_nel_set"); continue
            base = dict(_set_cue(tr))
            plan = d.get("plan") or {}
            for k in ("a_out_bar", "b_in_bar", "a_out_s", "b_in_s"):
                if plan.get(k) is not None:
                    base[k] = plan[k]
            entries.setdefault(i, []).append({"source": "set_approval", "doc": d, "verdict": d.get("verdict"),
                                              "corrections": LC.corrections_of(d), "base": base})
        elif typ == "versione":
            c = LC.corrections_of(d)
            cue = d.get("cue") if isinstance(d.get("cue"), dict) else None
            if c["schema"] == 1 or not cue:
                skip("senza_correzioni_strutturate (dj-review.1)"); continue
            i = by_pair.get((cue.get("a"), cue.get("b")))
            if i is None:
                skip("coppia_non_nel_set"); continue
            base = dict(_set_cue(ctx.transitions[i]))
            base.update({k: cue[k] for k in ("a_out_bar", "b_in_bar", "a_out_s", "b_in_s") if cue.get(k) is not None})
            base["type_key"] = d.get("kind") or base["type_key"]
            entries.setdefault(i, []).append({"source": "gallery", "doc": d, "verdict": None, "corrections": c, "base": base})
        elif typ == "torneo":
            pair = d.get("pair") or {}
            i = d.get("transition_index")
            i = int(i) if i and int(i) in ctx.transitions and (ctx.transitions[int(i)]["a"], ctx.transitions[int(i)]["b"]) == \
                (pair.get("a"), pair.get("b")) else by_pair.get((pair.get("a"), pair.get("b")))
            if i is None:
                skip("coppia_non_nel_set"); continue
            best = d.get("best")
            vdoc = variants.get((d.get("tournament_id"), best)) or {}
            c = LC.corrections_of(vdoc) if vdoc else LC.corrections_of({})
            base = dict(_set_cue(ctx.transitions[i]))
            if isinstance(d.get("best_cue"), dict):
                bc = d["best_cue"]
                base.update({k: bc.get(k) for k in ("a_out_bar", "b_in_bar", "a_out_s", "b_in_s")})
                base["b_lead_in_samples"] = int(bc.get("b_lead_in_samples") or 0)
            base["type_key"] = d.get("technique") or base["type_key"]
            entries.setdefault(i, []).append({"source": "cue_tournament", "doc": d, "variant_doc": vdoc or None,
                                              "verdict": "nessuna" if best == "nessuna" else "migliore",
                                              "corrections": c, "base": base})
        elif typ in ("torneo_variante", "preferenza", "transizione", "set"):
            continue  # varianti: lette col torneo; preferenze e pagina "set" v1: solo diagnostica
        else:
            skip("tipo_sconosciuto")
    return entries, ignored, set_doc


def choose_entry(items: list[dict]) -> tuple[dict, list[dict]]:
    items = sorted(items, key=lambda e: (SOURCE_PRIORITY[e["source"]], str(e["doc"].get("saved_at", ""))), reverse=True)
    return items[0], items[1:]


# --------------------------------------------------------------------------- risoluzione
def resolve(ctx: Context, index: int, entry: Optional[dict]) -> dict:
    """Record di una transizione: stato, cue originale e corretto, decisioni (per la metrica di accordo)."""
    tr = ctx.transitions[index]
    orig = _set_cue(tr)
    rec: dict[str, Any] = {"index": index, "a": tr["a"], "b": tr["b"], "original": orig, "corrected": None,
                           "changes": [], "needs_manual": [], "status": "senza_giudizio", "source": None,
                           "doc_ids": [], "verdict": None, "decisions": None}
    if entry is None:
        return rec
    d, c = entry["doc"], entry["corrections"]
    rec.update(source=entry["source"], verdict=entry.get("verdict"), corrections=c, note=d.get("note") or "",
               doc_ids=[d.get("doc_id")] + ([entry["variant_doc"].get("doc_id")] if entry.get("variant_doc") else []),
               judgment={"file": d.get("_file"), "file_sha256": d.get("_file_sha256"), "saved_at": d.get("saved_at"),
                         "blind": d.get("blind"), "plays_before_save": d.get("plays_before_save")})
    base = dict(entry["base"])
    A, B = ctx.analysis(tr["a"]), ctx.analysis(tr["b"])
    wrong_b = bool(c.get("wrong_b") or entry.get("verdict") == "da_rifare_altro_brano")
    dec = {"order": {"approved": not wrong_b}}
    if wrong_b:
        rec["status"] = "ripianifica"
        rec["changes"].append({"field": "wrong_b", "effect": f"coppia {tr['a']} → {tr['b']} vietata: ripianificare l'ordine"})
        rec["decisions"] = dec
        return rec
    if entry["source"] == "cue_tournament" and entry.get("verdict") == "nessuna":
        rec["status"] = "manuale"
        rec["needs_manual"].append("torneo: nessuna variante accettabile (provare un'altra tecnica o un altro brano B)")
        rec["decisions"] = dict(dec, exit={"approved": False, "shift_bars": None}, entry={"approved": False, "shift_bars": None},
                                technique={"approved": None})
        return rec
    a_bar, b_bar = base.get("a_out_bar"), base.get("b_in_bar")
    lead = int(base.get("b_lead_in_samples") or 0)
    tech = base.get("type_key")
    if A is None or B is None or a_bar is None or b_bar is None:
        rec["needs_manual"].append("analisi dei brani o cue mancanti: correzione non traducibile")
    else:
        a_bar, b_bar = int(a_bar), int(b_bar)
        # ---- uscita di A
        if c.get("stay_longer_a") and c.get("a_exit") in (None, "ok"):
            j, why = stay_longer_bar(A, a_bar)
            if j is None:
                rec["needs_manual"].append(why)
            else:
                rec["changes"].append({"field": "stay_longer_a", "from_bar": a_bar, "to_bar": j, "rule": why}); a_bar = j
        if c.get("a_exit") == "fine_frase":
            j, why = phrase_end_bar(A, a_bar, ctx.model(tr["a"]))
            if j is None:
                rec["needs_manual"].append(why)
            else:
                rec["changes"].append({"field": "a_exit", "value": "fine_frase", "from_bar": a_bar, "to_bar": j, "rule": why}); a_bar = j
        elif LC.bars_shift(c.get("a_exit")):
            k = LC.bars_shift(c["a_exit"])
            rec["changes"].append({"field": "a_exit", "value": c["a_exit"], "from_bar": a_bar, "to_bar": a_bar + k,
                                   "rule": f"griglia di A {k:+d} battute"}); a_bar += k
        # ---- ingresso di B
        if c.get("b_section"):
            j, why = section_bar(B, c["b_section"], b_bar)
            if j is None:
                rec["needs_manual"].append(why)
            else:
                rec["changes"].append({"field": "b_section", "value": c["b_section"], "from_bar": b_bar, "to_bar": j, "rule": why}); b_bar = j
        if LC.bars_shift(c.get("b_entry")):
            k = LC.bars_shift(c["b_entry"])
            rec["changes"].append({"field": "b_entry", "value": c["b_entry"], "from_bar": b_bar, "to_bar": b_bar + k,
                                   "rule": f"griglia di B {k:+d} battute"}); b_bar += k
        t_b = bar_time(B, b_bar)
        if c.get("b_pickup") and t_b is not None:
            n, why = pickup_samples(ctx.model(tr["b"]), B, t_b)
            if n > 0:
                rec["changes"].append({"field": "b_pickup", "lead_in_samples": n, "rule": why}); lead = n
            else:
                rec["needs_manual"].append("pickup di B richiesto ma " + why)
        if c.get("technique"):
            if c["technique"] == "altra":
                rec["needs_manual"].append("tecnica «altra»: vedi la nota del giudizio")
            else:
                rec["changes"].append({"field": "technique", "from": tech, "to": c["technique"]}); tech = c["technique"]
        for k_, msg in (("grid", "griglia {v}: rifare l'analisi del tempo prima di correggere i cue"),
                        ("grid_offbeat", "battuta sfasata: controllare la griglia (downbeat) di A e B")):
            if c.get(k_):
                rec["needs_manual"].append(msg.format(v=c.get("grid")))
        for side, track, bar in (("A", A, a_bar), ("B", B, b_bar)):
            if bar_time(track, bar) is None:
                rec["needs_manual"].append(f"battuta {bar} fuori dalla griglia di {side}")
        if not rec["needs_manual"] or all(bar_time(t_, b_) is not None for t_, b_ in ((A, a_bar), (B, b_bar))):
            rec["corrected"] = {"a_out_bar": a_bar, "b_in_bar": b_bar, "a_out_s": bar_time(A, a_bar),
                                "b_in_s": bar_time(B, b_bar), "type_key": tech, "b_lead_in_samples": lead}
    # ---- stato e decisioni (rispetto al piano del SISTEMA, non alla clip giudicata)
    changed = rec["corrected"] is not None and any(
        rec["corrected"][k] != orig[k] for k in ("a_out_bar", "b_in_bar", "type_key", "b_lead_in_samples"))
    grid_issue = bool(c.get("grid") or c.get("grid_offbeat"))
    if (rec["needs_manual"] and not changed) or grid_issue:
        rec["status"] = "manuale"  # griglia sospetta: le battute non sono affidabili finché l'analisi non è rifatta
    elif changed:
        rec["status"] = "corretta"
    elif entry["source"] == "gallery" and not LC.is_corrected(c):
        rec["status"] = "senza_correzioni"
    elif entry.get("verdict") == "da_correggere" and not LC.is_corrected(c):
        rec["status"] = "manuale"
        rec["needs_manual"].append("«da correggere» senza campi compilati: leggere la nota")
    else:
        rec["status"] = "approvata"
    cor = rec["corrected"] or {}

    def shift(k):
        if cor.get(k) is None or orig.get(k) is None:
            return None
        return int(cor[k]) - int(orig[k])

    explicit = entry.get("verdict") in ("approvata", "migliore")
    dec["exit"] = {"approved": shift("a_out_bar") in (0, None) and not any(ch["field"] in ("a_exit", "stay_longer_a") for ch in rec["changes"]
                                                                          if ch.get("from_bar") != ch.get("to_bar")),
                   "shift_bars": shift("a_out_bar"),
                   "explicit": explicit or c.get("a_exit") == "ok"}
    dec["entry"] = {"approved": shift("b_in_bar") in (0, None) and not c.get("b_pickup")
                    and (cor.get("b_lead_in_samples") or 0) == (orig.get("b_lead_in_samples") or 0),
                    "shift_bars": shift("b_in_bar"), "pickup": bool(c.get("b_pickup")),
                    "explicit": explicit or c.get("b_entry") == "ok"}
    dec["technique"] = {"approved": not c.get("technique") and (cor.get("type_key") in (None, orig["type_key"])),
                        "to": c.get("technique"), "explicit": explicit}
    rec["decisions"] = dec
    return rec


# --------------------------------------------------------------------------- congelate riutilizzabili
def frozen_decision(tr: dict) -> dict:
    """Decisione della transizione congelata (solo per il riuso del suo render: i parametri di render non servono)."""
    k = str(tr["type"])
    blend = k.startswith("blend") and k[5:].isdigit()
    tree = blend or k in ("cut", "echo_out", "filter_out", "filter_into_chorus", "backspin")
    render: dict = {}
    if not tree:  # parametri della proposta della policy registrata nel set (lead/overlap del runner v3 li leggono)
        props = ((tr.get("policy_books_v3") or {}).get("proposals") or []) + (tr.get("policy_proposals_books") or [])
        prop = next((x for x in props if isinstance(x, dict) and x.get("technique") == k and isinstance(x.get("params"), dict)), None)
        render = {"technique": k, "params": dict((prop or {}).get("params") or {})}
    if tree:  # parametri di render dell'albero (lead/overlap di dj_set li leggono), come nel set congelato
        render = dict((tree_decision(k) or {}).get("render") or {})
    return {"type": "blend" if blend else k, "type_key": k, "label": tr.get("label") or k, "rule": tr.get("rule") or "congelata",
            "rule_source": tr.get("rule_source"), "bars": int(k[5:]) if blend else int(tr.get("duration_bars") or 0),
            "render": render, "fallbacks": [], "creative": not tree, "trace": [] if tree else {}, "frozen": True}


def frozen_reuse(ctx: Context) -> list[dict]:
    """Transizioni accettate del set congelato (``transitions/tNN_accepted.{wav,json}`` + QA del tentativo)."""
    out = []
    tdir = ctx.dir / "transitions"
    for i, tr in sorted(ctx.transitions.items()):
        wav, meta = tdir / f"t{i:02d}_accepted.wav", tdir / f"t{i:02d}_accepted.json"
        if not (wav.is_file() and meta.is_file()):
            continue
        m = load_json(meta)
        expected = ((m.get("output") or {}).get("wav_sha256") or "").lower()
        actual = sha256_file(wav)
        if expected and expected != actual:
            continue
        att = tr.get("attempts") or []
        k = max((n for n, a in enumerate(att, 1) if a.get("verdict") not in (None, "fail", "render_error") and not a.get("veto")),
                default=None)
        qa_file = tdir / f"t{i:02d}_{k:02d}_{tr['type']}.json" if k else None
        out.append({"index": i, "a": tr["a"], "b": tr["b"], "a_out_bar": (tr.get("plan") or {}).get("a_out_bar"),
                    "b_in_bar": (tr.get("plan") or {}).get("b_in_bar"), "type_key": tr["type"],
                    "decision": frozen_decision(tr),
                    "wav": rel(wav), "wav_sha256": actual, "render_json": rel(meta),
                    "qa_json": rel(qa_file) if qa_file and qa_file.is_file() else None})
    return out


# --------------------------------------------------------------------------- piano
def build_plan(set_path: Path, judgment_paths: list[Path], run: Optional[str] = None, pin_unchanged: bool = True,
               reuse: bool = True) -> dict:
    ctx = Context(set_path)
    docs = LC.read_docs(judgment_paths)
    entries, ignored, set_doc = collect_entries(ctx, docs, run)
    records, superseded = [], []
    for i in sorted(ctx.transitions):
        items = entries.get(i) or []
        best, rest = choose_entry(items) if items else (None, [])
        superseded += [{"index": i, "doc_id": e["doc"].get("doc_id"), "source": e["source"]} for e in rest]
        records.append(resolve(ctx, i, best))
    pins, forbid, rerender = [], [], []
    for r in records:
        if r["status"] == "ripianifica":
            forbid.append([r["a"], r["b"]])
            continue
        if r["status"] == "corretta":
            rerender.append(r["index"])
            pins.append({"index": r["index"], "a": r["a"], "b": r["b"], **r["corrected"], "status": "corretta"})
        elif pin_unchanged:
            pins.append({"index": r["index"], "a": r["a"], "b": r["b"], **r["original"], "status": r["status"]})
    order = (ctx.doc.get("ordering") or {}).get("order") or [t["a"] for _, t in sorted(ctx.transitions.items())] + \
        ([ctx.transitions[max(ctx.transitions)]["b"]] if ctx.transitions else [])
    files = sorted({(d["_file"], d["_file_sha256"]) for d in docs})
    plan = {
        "schema_version": SCHEMA_PLAN,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "tools/apply_corrections.py",
        "set": {"path": rel(ctx.set_path), "sha256": ctx.sha, "run": ctx.run, "order": (ctx.doc.get("ordering") or {}).get("order")},
        "judgments": [{"file": rel(Path(f)), "sha256": s} for f, s in files],
        "set_judgment": {k: set_doc.get(k) for k in ("doc_id", "set_verdict", "flow", "energy", "variety", "repetitiveness",
                                                     "note", "saved_at")} if set_doc else None,
        "counts": {s: sum(r["status"] == s for r in records) for s in
                   ("approvata", "corretta", "ripianifica", "manuale", "senza_correzioni", "senza_giudizio")},
        "transitions": records,
        "ignored_docs": ignored,
        "superseded_docs": superseded,
        "runner": {
            "pins": pins, "forbid_pairs": forbid, "rerender": rerender,
            "pin_unchanged": bool(pin_unchanged),
            # ordine del set congelato bloccato (solo coppie consecutive ammissibili) se nessuna coppia è vietata
            "order_lock": order if (pin_unchanged and not forbid) else None,
            "reuse": {"frozen_run": rel(ctx.dir), "transitions": frozen_reuse(ctx) if reuse else []},
            "note": "pins = cue fissati (corretti o del set congelato); forbid_pairs = coppie da evitare nell'ordine; "
                    "order_lock = ordine congelato (nessuna coppia vietata); "
                    "rerender = transizioni corrette da rifare; reuse = render congelati usati solo se coppia, cue e tecnica coincidono",
        },
        "conventions": {"a_exit": "battute sulla griglia di A (switch); il contenuto di B allo switch non cambia",
                        "b_entry": "battute sulla griglia di B (punto d'ingresso nel brano B); l'uscita di A non cambia",
                        "rules": "fine frase, resta su A, sezione e pickup: regole [H] descritte in tools/apply_corrections.py"},
    }
    return plan


# --------------------------------------------------------------------------- riferimenti
def _words_at(model: Optional[dict], sample: int, role: str) -> dict:
    if not model:
        return {"status": "modello_dj_phrase_assente"}
    words = [w for w in model.get("words") or [] if not w.get("dropped")]
    if not words:
        return {"status": "nessuna_parola"}
    if role == "a":
        prev = [w for w in words if int(w["on"]) < sample]
        w = prev[-1] if prev else None
    else:
        nxt = [w for w in words if int(w["off"]) > sample]
        w = nxt[0] if nxt else None
    if w is None:
        return {"status": "nessuna_parola"}
    wi = (model.get("words") or []).index(w)
    line = next((ln for ln in model.get("lines") or [] if ln.get("first_word", 0) <= wi <= ln.get("last_word", -1)), None)
    return {"status": "ok", "word": w.get("text"), "word_on_sample": int(w["on"]), "word_off_sample": int(w["off"]),
            "truncated": bool(int(w["on"]) < sample < int(w["off"])),
            "line_excerpt": str((line or {}).get("text") or "")[:LINE_EXCERPT_CHARS], "line_p_end": (line or {}).get("p_end"),
            "line_start_sample": (line or {}).get("start_sample"), "line_end_sample": (line or {}).get("end_sample")}


def _track(ctx: Context, slug: str) -> dict:
    A = ctx.analysis(slug) or {}
    src = (ctx.doc.get("sources") or {}).get(slug) or {}
    return {"slug": slug, "title": src.get("title") or A.get("title"), "artist": src.get("artist") or A.get("artist"),
            "audio_sha256": A.get("track_id"), "analysis": A.get("_path"), "analysis_sha256": A.get("_sha256")}


def _section_at(A: dict, bar: int) -> Optional[str]:
    rows = A.get("phrases") or []
    return rows[bar].get("section_label") if 0 <= bar < len(rows) else None


def _grid(A: dict, bar: int) -> dict:
    tempo = A.get("tempo") or {}
    return {"bpm": tempo.get("bpm"), "period_s": tempo.get("period_s"), "provider": tempo.get("provider"),
            "confidence": tempo.get("confidence"), "bar_offset_beats": tempo.get("bar_offset_beats"),
            "n_bars": (A.get("grid") or {}).get("n_bars"), "bar": bar, "downbeat_s": bar_time(A, bar)}


def reference_records(plan: dict, verified_by: str = "utente") -> list[dict]:
    """Passaggi di riferimento dalle transizioni verificate all'ascolto (vedi docstring del modulo)."""
    ctx = Context(resolve_path(plan["set"]["path"]))
    if ctx.sha != plan["set"]["sha256"]:
        raise ValueError("il set.json è cambiato dopo la creazione del piano")
    refs = []
    prov = ctx.doc.get("corrections")  # set ri-renderizzato con un piano di correzioni (SetHooks.corrections)
    for r in plan["transitions"]:
        src, v = r.get("source"), r.get("verdict")
        if not ((src == "set_approval" and v == "approvata" and r["status"] == "approvata")
                or (src == "cue_tournament" and v == "migliore" and r["status"] in ("approvata", "corretta")
                    and not r["needs_manual"] and not LC.is_corrected(r.get("corrections") or {}))):
            continue
        cue = r["corrected"] or r["original"]
        A, B = ctx.analysis(r["a"]), ctx.analysis(r["b"])
        if A is None or B is None or cue.get("a_out_bar") is None:
            continue
        a_s, b_s = bar_time(A, int(cue["a_out_bar"])), bar_time(B, int(cue["b_in_bar"]))
        a_smp, b_smp = int(round(a_s * SR)), int(round(b_s * SR))
        lead = int(cue.get("b_lead_in_samples") or 0)
        from_corr = None
        if isinstance(prov, dict):
            from_corr = next((p for p in (prov.get("pins") or []) if p.get("a") == r["a"] and p.get("b") == r["b"]
                              and p.get("status") == "corretta"), None)
        rid = f"{r['a']}__{r['b']}__A{cue['a_out_bar']}_B{cue['b_in_bar']}" + (f"_L{lead}" if lead else "") + f"_{cue.get('type_key')}"
        refs.append({
            "schema_version": SCHEMA_REF, "id": rid,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "use": "verifica",
            "use_note": "passaggio verificato all'ascolto: dato di verifica, da non usare per tarare soglie o regole",
            "tracks": {"a": _track(ctx, r["a"]), "b": _track(ctx, r["b"])},
            "grid": {"a": _grid(A, int(cue["a_out_bar"])), "b": _grid(B, int(cue["b_in_bar"]))},
            "exit_a": {"bar": int(cue["a_out_bar"]), "seconds": a_s, "sample": a_smp, "sample_rate": SR,
                       "section": _section_at(A, int(cue["a_out_bar"])),
                       "words": _words_at(ctx.model(r["a"]), a_smp, "a")},
            "entry_b": {"bar": int(cue["b_in_bar"]), "seconds": b_s, "sample": b_smp, "sample_rate": SR,
                        "lead_in_samples": lead, "audible_sample": b_smp - lead,
                        "section": _section_at(B, int(cue["b_in_bar"])),
                        "words": _words_at(ctx.model(r["b"]), b_smp - lead, "b")},
            "technique": {"type_key": cue.get("type_key"), "label": dict(LC.TECHNIQUES).get(cue.get("type_key"), cue.get("type_key"))},
            "verified_by": verified_by,
            "verification": {"method": "set_approval" if src == "set_approval" else "cue_tournament",
                             "doc_ids": r["doc_ids"], **(r.get("judgment") or {}), "verdict": v},
            "derived_from": {"set": plan["set"], "transition_index": r["index"],
                             "corrections_plan_of_the_set": prov.get("plan") if isinstance(prov, dict) else None,
                             "correction_pin": from_corr,
                             "corrections_plan_entry": {"status": r["status"], "changes": r["changes"]}},
        })
    return refs


def write_references(refs: list[dict], out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ref in refs:
        body = {k: v for k, v in ref.items() if k != "created_at"}
        key = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:8]
        p = out_dir / f"{ref['id']}.json"
        if p.exists():
            old = load_json(p)
            if {k: v for k, v in old.items() if k != "created_at"} == body:
                continue
            p = out_dir / f"{ref['id']}__{key}.json"  # mai sovrascrivere un riferimento esistente
            if p.exists():
                continue
        p.write_text(json.dumps(ref, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(str(p))
    return written


# --------------------------------------------------------------------------- hook del runner
def _pair_pin(corr: dict, a: str, b: str) -> Optional[dict]:
    return next((p for p in (corr.get("runner") or {}).get("pins") or [] if p["a"] == a and p["b"] == b), None)


def is_forbidden(corr: dict, a: str, b: str) -> bool:
    return [a, b] in [list(x) for x in (corr.get("runner") or {}).get("forbid_pairs") or []]


def pin_pair(pc: dict, A: dict, B: dict, corr: dict, tree_params: Optional[dict] = None,
             order_params: Optional[dict] = None) -> dict:
    """Applica il piano alle candidate di una coppia (in place): coppia vietata → nessuna candidata ammissibile;
    cue fissato → solo quella candidata ammissibile (costruita con ``dj_select.plan_transition`` se manca)."""
    a, b = A["slug"], B["slug"]
    lock = (corr.get("runner") or {}).get("order_lock")
    if lock and (a, b) not in set(zip(lock, lock[1:])):
        for c in pc["candidates"]:
            c["admissible"] = False
        pc["corrections"] = {"status": "fuori_ordine"}  # ordine bloccato: si rifanno solo le transizioni corrette
        return pc
    if is_forbidden(corr, a, b):
        for c in pc["candidates"]:
            c["admissible"] = False
        pc.setdefault("notes", []).append("coppia vietata dalle correzioni (brano B sbagliato)")
        pc["corrections"] = {"status": "vietata"}
        return pc
    pin = _pair_pin(corr, a, b)
    if pin is None:
        return pc
    from musicmix import dj_select as djs
    from musicmix import dj_set as ds
    match = [c for c in pc["candidates"] if c["plan"]["a_out_bar"] == pin["a_out_bar"] and c["plan"]["b_in_bar"] == pin["b_in_bar"]]
    if match:
        keep = match[0]
        made = False
    else:
        out, inn = djs.bar_cue(A, int(pin["a_out_bar"])), djs.bar_cue(B, int(pin["b_in_bar"]))
        if out is None or inn is None:
            pc["corrections"] = {"status": "cue_fuori_griglia", "pin": pin}
            return pc
        plan = djs.plan_transition(A, B, out, inn, pc["compatibility"])
        ctx = ds.transition_context(A, B, plan, pc["compatibility"])
        dec = ds.decide_transition(ctx, tree_params)
        keep = {"plan": plan, "context": ctx, "decision": dec, "cost": ds.candidate_cost(plan, dec, order_params, ctx),
                "admissible": True, "notes": []}
        pc["candidates"].append(keep)
        made = True
    if int(pin.get("b_lead_in_samples") or 0) > 0:
        keep["plan"]["b_lead_in"] = {"status": "lead_in", "lead_in_samples": int(pin["b_lead_in_samples"]),
                                     "audible_sample": None, "evidence": "correzione dell'utente",
                                     "reasons": ["pickup di B richiesto dal giudizio"]}
    for c in pc["candidates"]:
        c["admissible"] = c is keep
    keep["admissible"] = True
    keep["notes"].append(f"cue fissato dalle correzioni ({pin.get('status')}): A battuta {pin['a_out_bar']}, B battuta {pin['b_in_bar']}"
                         + (f", tecnica {pin.get('type_key')}" if pin.get("type_key") else ""))
    keep["correction_pin"] = dict(pin)
    pc["candidates"].sort(key=lambda c: not c["admissible"])
    pc["corrections"] = {"status": "fissata", "constructed": made, "pin": pin}
    return pc


TREE_KINDS = ("cut", "echo_out", "filter_out", "filter_into_chorus", "backspin")


def tree_decision(type_key: str) -> Optional[dict]:
    """Decisione dell'albero di ``dj_set`` per una tecnica richiesta (blendN, cut, echo/filter out, backspin); le tecniche
    creative della policy (build, halving, loop...) hanno parametri che solo la policy calcola: None."""
    from musicmix import dj_set as ds
    p = ds.values(ds.TREE_PARAMS, None)
    if type_key.startswith("blend") and type_key[5:].isdigit():
        d = ds._mk("blend", p, bars=int(type_key[5:]))
    elif type_key in TREE_KINDS:
        d = ds._mk(type_key, p)
    else:
        return None
    return {**d, "type_key": type_key, "rule": "correzione_utente", "fallbacks": [], "trace": [],
            "rule_source": "[U] tecnica indicata dall'utente nel giudizio (tools/apply_corrections.py)"}


def prefer_technique(attempts: list, a: str, b: str, corr: dict) -> list:
    """Mette per primi i tentativi con la tecnica fissata (``type_key``) del piano; gli altri restano come riserva.

    Se nessun tentativo ha quella tecnica e la tecnica è dell'albero, la aggiunge in testa sul primo cue (il cue
    fissato); una tecnica creativa non proposta dalla policy su quel cue resta non disponibile (nota nel tentativo)."""
    pin = _pair_pin(corr, a, b)
    if not pin or not pin.get("type_key"):
        return attempts
    if pin.get("status") != "corretta" and attempts:  # transizione non corretta: prima il render congelato, se c'è
        fz = next((r for r in ((corr.get("runner") or {}).get("reuse") or {}).get("transitions") or []
                   if (r["a"], r["b"], r["a_out_bar"], r["b_in_bar"], r["type_key"]) ==
                   (a, b, pin["a_out_bar"], pin["b_in_bar"], pin["type_key"]) and r.get("decision")), None)
        if fz is not None:
            return [(attempts[0][0], dict(fz["decision"]), "render congelato riusato")] + list(attempts)
    first = [x for x in attempts if x[1].get("type_key") == pin["type_key"]]
    if first:
        return first + [x for x in attempts if x not in first]
    dec = tree_decision(pin["type_key"]) if attempts else None
    if dec is None:
        return attempts
    return [(attempts[0][0], dec, "tecnica richiesta dalla correzione")] + list(attempts)


def reuse_render(corr: dict, a: str, b: str, plan: dict, dec: dict):
    """``(res, qa)`` dal set congelato se coppia, cue e tecnica coincidono e la transizione non è da rifare; altrimenti None."""
    rn = corr.get("runner") or {}
    pin = _pair_pin(corr, a, b)
    if pin is not None and pin.get("status") == "corretta":
        return None
    for r in (rn.get("reuse") or {}).get("transitions") or []:
        if (r["a"], r["b"], r["a_out_bar"], r["b_in_bar"], r["type_key"]) != (a, b, plan.get("a_out_bar"), plan.get("b_in_bar"),
                                                                               dec.get("type_key")):
            continue
        if not r.get("qa_json"):
            return None
        import numpy as np
        import soundfile as sf
        from musicmix import dj_render as dj
        wav = resolve_path(r["wav"])
        if sha256_file(wav) != r["wav_sha256"]:
            return None
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        meta = load_json(resolve_path(r["render_json"]))
        qa = (load_json(resolve_path(r["qa_json"])) or {}).get("qa")
        if not qa:
            return None
        meta = dict(meta, reused_from={"wav": r["wav"], "sha256": r["wav_sha256"], "index": r["index"]})
        return dj.DJRenderResult(audio=np.ascontiguousarray(audio), sample_rate=int(sr), metadata=meta, decks=None), qa
    return None


def provenance(corr: dict) -> dict:
    rn = corr.get("runner") or {}
    return {"plan": {"path": corr.get("_path"), "sha256": corr.get("_sha256"), "schema_version": corr.get("schema_version")},
            "set": corr.get("set"), "pins": rn.get("pins"), "forbid_pairs": rn.get("forbid_pairs"), "rerender": rn.get("rerender")}


def load_plan(path: Path) -> dict:
    raw = Path(path).read_bytes()
    plan = json.loads(raw.decode("utf-8"))
    if plan.get("schema_version") != SCHEMA_PLAN:
        raise ValueError(f"atteso {SCHEMA_PLAN}")
    plan["_path"] = rel(Path(path))
    plan["_sha256"] = hashlib.sha256(raw).hexdigest()
    return plan


# --------------------------------------------------------------------------- ri-render (set v3)
def rerender(plan_path: Path, out: Path, plan_only: bool = False, overwrite: bool = False) -> int:
    """Set v3 ``ponte`` con il piano: cue fissati, coppie vietate, render congelati riusati, accettate scritte in ``out``."""
    import run_dj_set_v2_books as v2
    import run_dj_set_v3_bridge as v3
    corr = load_plan(plan_path)
    out = Path(out)
    if out.resolve() == v3.OUT.resolve():
        raise SystemExit("la run congelata non si riscrive: usare --out nuovo")
    orig_build = v2.build_set

    def patched(out_, **kw):
        hooks = kw.get("hooks")
        hooks.corrections = corr
        if hooks.on_accept is not None:
            inner = hooks.on_accept

            def on_accept(*args, **kwargs):  # on_accept_v3 scrive in v3.OUT: qui nella run nuova
                saved = v3.OUT
                v3.OUT = Path(out_)
                try:
                    return inner(*args, **kwargs)
                finally:
                    v3.OUT = saved
            hooks.on_accept = on_accept
        return orig_build(out_, **kw)

    v2.build_set = patched
    try:
        v3.run_set(overwrite, out, plan_only=plan_only)
    finally:
        v2.build_set = orig_build
    return 0


# --------------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="giudizi → corrections_plan.json (e passaggi di riferimento)")
    p.add_argument("--set", required=True, type=Path)
    p.add_argument("--judgments", required=True, type=Path, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--run", help="nome della run nei giudizi (default: run del set)")
    p.add_argument("--no-pin-unchanged", action="store_true", help="non fissare i cue delle transizioni non corrette")
    p.add_argument("--references", type=Path, help="cartella dei passaggi di riferimento (es. giudizi_cloud/reference_transitions)")
    p.add_argument("--verified-by", default="utente")
    r = sub.add_parser("rerender", help="set v3 con il piano (cue fissati, coppie vietate, riuso dei render congelati)")
    r.add_argument("--plan", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--plan-only", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan = build_plan(a.set, a.judgments, run=a.run, pin_unchanged=not a.no_pin_unchanged)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(plan["counts"]), "->", a.out)
        for t in plan["transitions"]:
            if t["status"] not in ("senza_giudizio",):
                print(f"  T{t['index']:02d} {t['a']} -> {t['b']}: {t['status']}"
                      + (f" | {'; '.join(LC.describe(t.get('corrections') or {}))}" if t.get("corrections") else "")
                      + (f" | manuale: {'; '.join(t['needs_manual'])}" if t["needs_manual"] else ""))
        if a.references:
            w = write_references(reference_records(plan, a.verified_by), a.references)
            print(f"passaggi di riferimento scritti: {len(w)}")
        return 0
    return rerender(a.plan, a.out, plan_only=a.plan_only, overwrite=a.overwrite)


if __name__ == "__main__":
    sys.exit(main())
