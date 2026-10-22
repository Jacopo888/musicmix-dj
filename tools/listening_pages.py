"""Pagine d'ascolto per la verità di riferimento: approvazione del set intero e torneo dei cue.

Chiamate da ``tools/build_listening_artifact.py --kind set-approval|cue-tournament`` (stesse convenzioni: pagina
autonoma, audio in ``audio/``, giudizi nella collezione ``judgments`` del database dell'Artifact con copia locale nel
browser e "Copia JSON"; manifest letti e mai modificati, percorsi registrati risolti con ``musicmix.paths.localize``).

**Approvazione del set** (``musicmix.set-approval.1``; ingresso: ``set.json`` ``musicmix.dj_set.1`` o
``musicmix.dj-set.1``). Tutto il set in un player con bande dei brani, regioni delle transizioni e linee di switch;
per ogni transizione: "ascolta la zona" (da 10 s prima dell'inizio a 10 s dopo la fine, con loop facoltativo),
verdetto ``approvata`` / ``da_correggere`` / ``da_rifare_altro_brano``, scelte rapide, correzioni strutturate
(:mod:`listening_corrections`) e nota → doc ``<run>__set_tNN``; giudizio complessivo (verdetto del set, flusso,
energia, varietà, ripetitività 1–5, nota) → doc ``<run>__set``. Ogni doc di transizione porta il piano del sistema
(battute e secondi di uscita/ingresso, tecnica) così che :mod:`apply_corrections` e :mod:`decision_agreement` non
dipendano dalla pagina. Audio: ``--set-mp3`` copiato, altrimenti il WAV del set (hash verificato) codificato in MP3
CBR al bitrate più alto tra 128/112/96/80/64 kbps che resta sotto ``SET_MP3_MAX_BYTES`` (limite dei file degli
Artifact).

**Torneo dei cue** (``musicmix.cue-tournament.1``; ingresso: manifest ``musicmix.cue-variants.1``). Per una coppia
A→B 4–6 varianti con la **stessa tecnica** e cue diversi (uscita di A e/o ingresso di B); ordine casuale con seme
registrato, alla cieca (cue e "scelta del sistema" compaiono dopo il salvataggio della scelta). Per torneo: la
migliore (o nessuna) e le accettabili → doc ``<run>__<torneo>__torneo``; per variante: scelte rapide, correzioni e
nota → doc ``<run>__<torneo>__<variante>``. Il manifest d'ascolto elenca i confronti espliciti (coppie che variano
solo il cue di B, solo il cue di A, entrambi, solo il pickup).

Formato del manifest delle varianti (``musicmix.cue-variants.1``)::

    {"schema_version": "musicmix.cue-variants.1", "run": "b_entry_fix_v1", "title": "...",
     "source_set": {"path": "runs/dj_set_v3_bridge/set.json", "sha256": "..."},          # facoltativo
     "tournaments": [{
        "id": "t03", "transition_index": 3,                                              # indice nel set (facolt.)
        "a": {"slug": "goosebumps", "title": "...", "artist": "...", "audio_sha256": "..."},
        "b": {"slug": "...", ...},
        "technique": "blend4",                     # type_key del runner; tutte le varianti devono averla
        "system": {"a_out_bar": 48, "b_in_bar": 0},  # cue scelto dal sistema (facoltativo)
        "variants": [{
           "id": "t03_bbbbd", "audio_path": "runs/b_entry_fix_v1/clips/t03_bbbbd.mp3",   # .wav o .mp3
           "sha256": "...",                        # del file indicato (se assente: hash_non_verificabile)
           "technique": "blend4",                  # facoltativo (default: quella del torneo)
           "a_out_bar": 48, "b_in_bar": 0, "a_out_s": 89.10, "b_in_s": 0.74,
           "b_lead_in_samples": 0,                 # pickup/lead-in di B a 48 kHz (facoltativo)
           "transition_start_s": 10.0, "switch_s": 13.69, "transition_end_s": 17.38, "duration_s": 27.38,
           "label": "...", "qa": {"verdict": "pass", "warn_reasons": []}}]}]}

Le clip seguono AGENTS.md (taglio: 20 s con lo switch a 10 s; sovrapposizione: 10 s + transizione + 10 s); una clip
fuori regola resta, con un avviso nel manifest d'ascolto.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import shutil

import build_listening_artifact as BLA
import listening_corrections as LC

SCHEMA_APPROVAL = "musicmix.set-approval.1"
SCHEMA_TOURNAMENT = "musicmix.cue-tournament.1"
SCHEMA_VARIANTS = "musicmix.cue-variants.1"
ZONE_LEAD_S = 10.0
SET_BITRATES = ("128k", "112k", "96k", "80k", "64k")
SIZE_MARGIN = 0.95  # stima CBR: si resta sotto il 95 % del limite
VERDICTS = (("approvata", "approvata"), ("da_correggere", "da correggere"),
            ("da_rifare_altro_brano", "da rifare con altro brano"))
SET_VERDICTS = (("approvato", "approvato dall'inizio alla fine"), ("da_correggere", "da correggere"),
                ("da_rifare", "da rifare"))
SET_SCALES = (("flow", "Flusso", "1 = a scatti · 5 = scorre"), ("energy", "Energia", "1 = piatta o sbagliata · 5 = giusta"),
              ("variety", "Varietà", "1 = monotono · 5 = vario"),
              ("repetitiveness", "Ripetitività", "1 = per nulla ripetitivo · 5 = molto ripetitivo"))
NO_BEST = "nessuna"


# --------------------------------------------------------------------------- ids
def approval_transition_doc_id(run: str, index: int) -> str:
    return f"{BLA.sanitize_segment(run)}__set_t{int(index):02d}"


def approval_set_doc_id(run: str) -> str:
    return f"{BLA.sanitize_segment(run)}__set"


def tournament_doc_id(run: str, tid: str) -> str:
    return f"{BLA.doc_id(run, tid)}__torneo"


def variant_doc_id(run: str, tid: str, vid: str) -> str:
    return f"{BLA.doc_id(run, tid)}__{BLA.sanitize_segment(vid)}"


def _css(template: str) -> str:
    return template.split("<style>", 1)[1].split("</style>", 1)[0]


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def _prepare_out(out: Path, overwrite: bool) -> Path:
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} esiste e non è vuota (usa --overwrite)")
    (out / "audio").mkdir(parents=True, exist_ok=True)
    return out


def choose_set_bitrate(duration_s: float, limit_bytes: int = BLA.SET_MP3_MAX_BYTES) -> str:
    """Bitrate CBR più alto (<= 128 kbps) che tiene l'MP3 del set sotto il limite."""
    for br in SET_BITRATES:
        if duration_s * int(br[:-1]) * 1000 / 8 <= SIZE_MARGIN * limit_bytes:
            return br
    return SET_BITRATES[-1]


# ======================================================================= set approval
def build_set_approval(manifest_path: Path, out: Path, run: str | None = None, encoder: str = "auto",
                       overwrite: bool = False, log=print, root: Path | None = None, set_mp3: Path | None = None,
                       blind: bool = False, bitrate: str | None = None) -> dict:
    manifest_path = Path(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf8"))
    if manifest.get("schema_version") not in BLA.SCHEMAS_SET:
        raise ValueError(f"approvazione del set: atteso set.json ({', '.join(BLA.SCHEMAS_SET)}), "
                         f"trovato {manifest.get('schema_version')!r}")
    manifest_dir = manifest_path.parent
    root = Path(root) if root is not None else BLA.repo_root()
    norm = BLA.convert_set(manifest, manifest_dir)
    run = run or norm.get("run") or manifest_path.resolve().parent.name
    norm["run"] = run
    tracks, transitions, excluded = BLA.validate_set(norm)
    if not tracks:
        raise ValueError("nessun brano valido nel set")
    for e in excluded:
        log(f"ESCLUSA {e['kind']} {e.get('id')}: {e['reason']} {e.get('detail', '')}".rstrip())
    out = _prepare_out(out, overwrite)
    dur = float(norm["duration_s"])
    audio_src, enc_info, dst = _set_audio(norm, manifest_dir, root, out, encoder, set_mp3, bitrate or choose_set_bitrate(dur), log)
    probe = BLA._probe(dst)
    size = dst.stat().st_size
    if size > BLA.SET_MP3_MAX_BYTES:
        log(f"AVVISO: {dst.name} pesa {size / 1e6:.1f} MB (> {BLA.SET_MP3_MAX_BYTES / 1e6:.0f} MB)")
    if probe.get("duration_s") is not None and abs(probe["duration_s"] - dur) > 1.0:
        log(f"AVVISO: audio di {probe['duration_s']:.2f} s, il set ne dichiara {dur:.2f}")

    raw_tr = {int(t["index"]): t for t in manifest.get("transitions") or []
              if isinstance(t, dict) and str(t.get("index", "")).isdigit()}
    by_slug = {t["slug"]: t for t in tracks}
    for t in transitions:
        t["doc_id"] = approval_transition_doc_id(run, t["index"])
        for side in ("a", "b"):
            tr = by_slug.get(t.get(side))
            t[f"{side}_name"] = (f"{tr['artist']} – {tr['title']}" if tr and tr["artist"]
                                 else (tr["title"] if tr else str(t.get(side) or "?")))
        plan = (raw_tr.get(t["index"]) or {}).get("plan") or {}
        t["plan"] = {k: plan.get(k) for k in ("a_out_bar", "b_in_bar", "a_out_s", "b_in_s", "type", "blend_bars")}
    audio = {"path": f"audio/{dst.name}", "sha256": BLA.sha256_file(dst), "bytes": size, "probe": probe,
             "source": audio_src}
    listing = {
        "schema_version": BLA.SCHEMA_OUT, "kind": "set-approval", "page_schema": SCHEMA_APPROVAL,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": run, "blind": bool(blind),
        "source_manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(raw).hexdigest(),
                            "schema_version": manifest.get("schema_version"), "title": norm.get("title")},
        "encoder": enc_info, "audio": audio, "duration_s": dur,
        "db": {"collection": BLA.COLLECTION, "transition_doc_id": "<run>__set_t<NN>",
               "set_doc_id": approval_set_doc_id(run)},
        "order": [t["slug"] for t in tracks], "tracks": tracks, "transitions": transitions,
        "included": [{"id": "set", "audio": audio["path"], "source_sha256": audio_src["sha256"]}],
        "excluded": excluded,
    }
    (out / "listening_manifest.json").write_text(json.dumps(listing, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    (out / "index.html").write_text(render_set_approval(run, norm, tracks, transitions, audio, dur, audio_src["sha256"],
                                                        blind), encoding="utf8")
    log(f"approvazione: {len(tracks)} brani, {len(transitions)} transizioni, audio {size / 1e6:.1f} MB -> {out}")
    return listing


def _set_audio(norm: dict, manifest_dir: Path, root: Path, out: Path, encoder: str, set_mp3: Path | None,
               bitrate: str, log):
    recorded = norm.get("audio") or {}
    if set_mp3 is not None:
        src = Path(set_mp3)
        if not src.is_file():
            raise FileNotFoundError(f"mp3 del set non trovato: {src}")
        dst = out / "audio" / "set.mp3"
        shutil.copyfile(src, dst)
        actual = BLA.sha256_file(src)
        rec_mp3, expected = recorded.get("mp3"), str(recorded.get("mp3_sha256") or "").lower()
        same = bool(rec_mp3) and BLA._basename(str(rec_mp3)) == src.name
        if same and expected and expected != actual:
            raise ValueError(f"sha256 dell'mp3 diverso: atteso {expected}, trovato {actual}")
        return ({"kind": "mp3_esistente", "path": str(src), "sha256": actual,
                 "hash_status": "verificato" if same and expected else "hash_non_verificabile"},
                {"kind": "copia", "bitrate": None}, dst)
    if not recorded.get("wav"):
        raise ValueError("il manifest non indica il WAV del set (output.wav): usa --set-mp3")
    src = BLA.resolve_review_audio(str(recorded["wav"]), manifest_dir, root)
    if not src.is_file() and (manifest_dir / BLA._basename(str(recorded["wav"]))).is_file():
        src = manifest_dir / BLA._basename(str(recorded["wav"]))
    if not src.is_file():
        raise FileNotFoundError(f"WAV del set non trovato: {src}")
    actual = BLA.sha256_file(src)
    expected = str(recorded.get("wav_sha256") or "").lower()
    if expected and expected != actual:
        raise ValueError(f"sha256 del WAV diverso: atteso {expected}, trovato {actual}")
    enc = BLA.choose_encoder(encoder)
    dst = out / "audio" / ("set.wav" if enc == "wav16" else "set.mp3")
    log(f"encoder: {enc}" + (f" ({bitrate} CBR)" if enc == "ffmpeg-mp3" else ""))
    BLA.encode(src, dst, enc, bitrate)
    return ({"kind": "wav", "path": str(src), "path_recorded": recorded["wav"], "sha256": actual,
             "hash_status": "verificato" if expected else "hash_non_verificabile"},
            {"kind": enc, "bitrate": bitrate if enc == "ffmpeg-mp3" else None}, dst)


def _chips(name: str, choices, cls: str = "") -> str:
    return "".join(f'<label class="chip{cls}"><input type="radio" name="{html.escape(name)}" value="{html.escape(k)}">'
                   f'<span>{html.escape(v)}</span></label>' for k, v in choices)


def _tags(prefix: str) -> str:
    return "".join(f'<label class="chip"><input type="checkbox" value="{k}"><span>{html.escape(v)}</span></label>'
                   for k, v in BLA.DJ_TAGS)


def _approval_transition(t: dict, blind: bool) -> str:
    idx = t["index"]
    hide = " data-reveal hidden" if blind else ""
    qa = BLA._qa_badge(t)
    if blind and qa:
        qa = qa.replace("<span ", "<span data-reveal hidden ", 1)
    span = (f"taglio a {BLA._mmss(t['switch_s'])}" if t["end_s"] - t["start_s"] < 0.05 else
            f"{BLA._mmss(t['start_s'])}–{BLA._mmss(t['end_s'])} · switch {BLA._mmss(t['switch_s'])} · "
            f"{t['end_s'] - t['start_s']:.1f} s")
    plan = t.get("plan") or {}
    cue = (f"A esce alla battuta {plan['a_out_bar']} · B entra dalla battuta {plan['b_in_bar']}"
           if plan.get("a_out_bar") is not None and plan.get("b_in_bar") is not None else "")
    rule = f' · regola <code>{html.escape(str(t["rule"]))}</code>' if t.get("rule") else ""
    return f"""<article class="tr" id="t{idx:02d}" data-doc="{html.escape(t['doc_id'])}" data-index="{idx}">
  <div class="row thead"><span class="tnum">T{idx:02d}</span><span class="badge f-{t['family']}"{hide}>{html.escape(t['label'])}</span>{qa}<span class="done small muted"></span></div>
  <p class="pair">{html.escape(t['a_name'])} → {html.escape(t['b_name'])}</p>
  <p class="muted small">{span}</p>
  <p class="muted small"{hide}>{html.escape(cue)}{rule}</p>
  <div class="row buttons">
    <button type="button" data-act="zone" aria-pressed="false">ascolta la zona (−10 s)</button>
    <label class="loop small"><input type="checkbox" class="loopbox"> loop</label>
  </div>
  <form class="judge">
    <fieldset><legend>Verdetto</legend><div class="chips">{_chips(f'verdict-t{idx}', VERDICTS)}</div></fieldset>
    <fieldset><legend>Scelte rapide</legend><div class="chips">{_tags(f't{idx}')}</div></fieldset>
    {LC.fieldset_html(f"t{idx}")}
    <label class="note-label">Nota<textarea rows="2" name="note" placeholder="cosa cambiare, cosa funziona…"></textarea></label>
    <div class="row between"><button type="submit" class="primary">Salva</button><span class="status small muted">non salvato</span></div>
    <p class="existing small muted"></p>
  </form>
</article>"""


def _approval_set_form(run: str) -> str:
    scales = "".join(
        f'<fieldset><legend>{label} <span class="small muted">({hint})</span></legend><div class="chips">'
        + "".join(f'<label class="chip"><input type="radio" name="set-{key}" value="{n}"><span>{n}</span></label>'
                  for n in range(1, 6)) + "</div></fieldset>" for key, label, hint in SET_SCALES)
    return f"""<section class="card" id="set-judge" data-doc="{html.escape(approval_set_doc_id(run))}">
  <h2>Giudizio complessivo del set</h2>
  <form class="judge setform">
    <fieldset><legend>Il set</legend><div class="chips">{_chips('set-verdict', SET_VERDICTS)}</div></fieldset>
    {scales}
    <label class="note-label">Nota<textarea rows="3" name="note" placeholder="flusso, energia, ordine, punti da rifare…"></textarea></label>
    <div class="row between"><button type="submit" class="primary">Salva giudizio del set</button><span class="status small muted">non salvato</span></div>
    <p class="existing small muted"></p>
  </form>
</section>"""


def render_set_approval(run: str, norm: dict, tracks: list[dict], transitions: list[dict], audio: dict, dur: float,
                        source_sha256: str, blind: bool = False) -> str:
    bar, nums = BLA._set_bar(tracks, transitions, dur)
    fams = [f for f in BLA.SET_FAMILY_LABELS if any(t["family"] == f for t in transitions)]
    legend = "" if blind else "".join(f'<span class="lg"><span class="legend-key f-{f}"></span>{BLA.SET_FAMILY_LABELS[f]}</span>'
                                      for f in fams)
    legend += ('<span class="lg"><span class="legend-key line"></span>switch</span>'
               '<span class="lg"><span class="legend-key bnd"></span>confine brano</span>')
    if blind:  # regions without the technique colour or title
        import re
        bar = re.sub(r'class="region f-[a-z_]+"', 'class="region f-other"', bar)
        bar = re.sub(r'title="T(\d+) [^"]*"', r'title="T\1"', bar)
    data = {
        "kind": "set-approval", "schema": SCHEMA_APPROVAL, "run": run, "collection": BLA.COLLECTION, "blind": bool(blind),
        "zone_lead": ZONE_LEAD_S, "duration_s": dur, "audio_sha256": audio["sha256"], "source_sha256": source_sha256,
        "set_doc_id": approval_set_doc_id(run), "order": [t["slug"] for t in tracks],
        "tags": [{"key": k, "label": v} for k, v in BLA.DJ_TAGS], "verdicts": dict(VERDICTS),
        "set_verdicts": dict(SET_VERDICTS), "scales": [{"key": k, "label": lab} for k, lab, _ in SET_SCALES],
        "tracks": [{"position": t["position"], "start_s": t["start_s"], "end_s": t["end_s"],
                    "name": (f"{t['artist']} – {t['title']}" if t["artist"] else t["title"])} for t in tracks],
        "transitions": [{k: t.get(k) for k in ("index", "doc_id", "type", "family", "label", "a", "b", "start_s",
                                                 "switch_s", "end_s", "qa_verdict", "plan")} for t in transitions],
    }
    heading = norm.get("title")
    subtitle = f'<p class="small">{html.escape(heading)}</p>' if heading else ""
    if blind:
        subtitle += f'<p class="small muted">{html.escape(BLA.BLIND_NOTE)}</p>'
    rows = "\n".join(_approval_transition(t, blind) for t in transitions) or '<p class="muted">Nessuna transizione valida.</p>'
    css = _css(BLA.SET_HTML_TEMPLATE) + LC.CSS + APPROVAL_CSS
    return (APPROVAL_TEMPLATE
            .replace("__CSS__", css)
            .replace("__SUBTITLE__", subtitle)
            .replace("__RUN__", html.escape(run))
            .replace("__NTRACKS__", str(len(tracks)))
            .replace("__NTRANS__", str(len(transitions)))
            .replace("__DUR__", BLA._mmss(dur))
            .replace("__LEGEND__", legend)
            .replace("__AUDIO__", html.escape(audio["path"]))
            .replace("__DURS__", f"{dur:.2f}")
            .replace("__BAR__", bar)
            .replace("__NUMS__", nums)
            .replace("__TRACKS__", "\n".join(BLA._set_track_row(t) for t in tracks))
            .replace("__TRANSITIONS__", rows)
            .replace("__SETFORM__", _approval_set_form(run))
            .replace("__COMMON_JS__", COMMON_JS)
            .replace("__CORR_JS__", LC.js())
            .replace("__DATA__", _json(data)))


APPROVAL_CSS = """[hidden]{display:none!important}
.progress{font-weight:600}
.done.ok{color:var(--ok)}
.tr.verdict-approvata{border-left:4px solid var(--ok)}
.tr.verdict-da_correggere{border-left:4px solid var(--warn)}
.tr.verdict-da_rifare_altro_brano{border-left:4px solid var(--fail)}
details.tracklist summary{cursor:pointer;font-weight:600;min-height:32px;display:flex;align-items:center}
"""

#: JS comune: stato, salvataggio con coda, copia locale, "Copia JSON", database con onSnapshot.
#: Ogni pagina registra i suoi moduli con ``register(docId, form, {fill, make, describe, after})``.
COMMON_JS = r"""
const LS_KEY = "musicmix:judgments:" + CFG.run + ":" + CFG.kind;
const state = {db: null, saved: {}, local: {}};
const REG = new Map();
const $ = (s, el) => (el || document).querySelector(s);
const fmt = s => { s = Math.max(0, s || 0); const m = Math.floor(s / 60); const r = Math.floor(s % 60); return m + ":" + String(r).padStart(2, "0"); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
function lsRead(){ try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}") || {}; } catch (e) { return {}; } }
function lsWrite(obj){ try { localStorage.setItem(LS_KEY, JSON.stringify(obj)); return true; } catch (e) { return false; } }
function banner(msg){ const b = $("#banner"); b.textContent = msg; b.classList.toggle("show", !!msg); }
function setStatus(form, text, cls){ const s = $(".status", form); s.textContent = text; s.className = "status small " + (cls || "muted"); }
function register(docId, form, spec){
  REG.set(docId, Object.assign({form: form}, spec));
  form.addEventListener("input", () => { form._dirty = true; setStatus(form, "modifiche non salvate"); });
  form.addEventListener("submit", ev => { ev.preventDefault(); save(docId); });
}
function showExisting(docId, j, where){
  const r = REG.get(docId); const p = $(".existing", r.form);
  if (!j) { p.textContent = ""; return; }
  const when = j.saved_at ? new Date(j.saved_at).toLocaleString("it-IT") : "";
  p.textContent = "Giudizio " + where + ": " + r.describe(j) + (when ? " · " + when : "");
}
function applyJudgment(docId, j, where){
  const r = REG.get(docId);
  if (!r) return;
  showExisting(docId, j, where);
  if (j && !r.form._dirty && !r.form._saving) { r.fill(j); setStatus(r.form, "salvato", "ok"); }
  if (j && r.after) r.after(j);
}
const queue = {};
async function save(docId){
  const r = REG.get(docId), form = r.form;
  const q = queue[docId] || (queue[docId] = {busy: false, pending: false});
  if (q.busy) { q.pending = true; setStatus(form, "in coda…"); return; }
  const first = r.make();
  if (first === null) return;
  q.busy = true; form._saving = true;
  const btn = $("button[type=submit]", form); btn.disabled = true;
  try {
    let body = first;
    do {
      q.pending = false;
      body = body || r.make();
      if (!body) break;
      body = Object.assign(body, {run: CFG.run, doc_id: docId, page_schema: CFG.schema, saved_at: new Date().toISOString()});
      form._dirty = false;
      setStatus(form, "salvataggio…");
      if (state.db) {
        try {
          await state.db.collection(CFG.collection).doc(docId).set(body);
          state.saved[docId] = body;
          setStatus(form, "salvato nel database", "ok");
        } catch (e) {
          const code = e && e.code || "errore";
          state.local[docId] = body; lsWrite(state.local);
          setStatus(form, "non salvato nel database (" + code + "): copia locale", "err");
          if (code === "invalid_argument") banner("Non hai permesso di scrittura sul database: i giudizi restano in questo browser. Usa “Copia JSON”.");
        }
      } else {
        state.local[docId] = body;
        const ok = lsWrite(state.local);
        setStatus(form, ok ? "salvato in questo browser" : "non salvato (storage bloccato): usa Copia JSON", ok ? "ok" : "err");
      }
      const j = state.saved[docId] || state.local[docId];
      showExisting(docId, j, state.saved[docId] ? "salvato" : "locale");
      if (j && r.after) r.after(j);
      body = null;
    } while (q.pending);
  } finally {
    q.busy = false; form._saving = false; btn.disabled = false;
  }
}
$("#copy").addEventListener("click", async () => {
  const all = Object.assign({}, state.local, state.saved);
  const text = JSON.stringify({run: CFG.run, collection: CFG.collection, judgments: all}, null, 2);
  try { await navigator.clipboard.writeText(text); $("#copy").textContent = "Copiato"; }
  catch (e) {
    const ta = document.createElement("textarea"); ta.value = text; ta.rows = 8;
    $("header").appendChild(ta); ta.select();
    $("#copy").textContent = "Seleziona e copia il testo qui sotto";
  }
  setTimeout(() => { $("#copy").textContent = "Copia JSON"; }, 3000);
});
async function getDb(){
  const deadline = Date.now() + 10000;
  while (!(window.claude && typeof window.claude.use === "function") && Date.now() < deadline) await sleep(100);
  if (!(window.claude && typeof window.claude.use === "function")) return null;
  try { return await Promise.race([window.claude.use("db"), sleep(12000).then(() => null)]); }
  catch (e) { return null; }
}
function startDb(){
  state.local = lsRead();
  Object.entries(state.local).forEach(([id, j]) => applyJudgment(id, j, "locale"));
  getDb().then(db => {
    if (!db) {
      $("#dbstate").textContent = "Database non disponibile: salvataggio in questo browser.";
      banner("Database dell'Artifact non disponibile in questa vista: i giudizi vengono salvati solo in questo browser. Usa “Copia JSON” per conservarli.");
      return;
    }
    state.db = db;
    $("#dbstate").textContent = "Giudizi salvati nel database dell'Artifact (" + CFG.collection + ").";
    try {
      db.collection(CFG.collection).onSnapshot(snap => {
        snap.docs.forEach(d => {
          const j = d.data();
          if (!j || j.run !== CFG.run || !REG.has(d.id)) return;
          state.saved[d.id] = j;
          applyJudgment(d.id, j, "salvato");
        });
      }, err => { $("#dbstate").textContent = "Lettura dal database interrotta (" + (err && err.code || "errore") + ")."; });
    } catch (e) { $("#dbstate").textContent = "Lettura dal database non riuscita."; }
  });
}
"""

APPROVAL_TEMPLATE = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Approvazione del set</title>
<style>__CSS__</style>
</head>
<body>
<main>
<header>
  <h1>Approvazione del set</h1>
  __SUBTITLE__
  <p class="muted small">Run <code>__RUN__</code> · __NTRACKS__ brani · __NTRANS__ transizioni · __DUR__ · <span class="progress" id="progress"></span></p>
  <p class="small muted">Per ogni transizione: ascolta la zona, poi «approvata», «da correggere» (con le correzioni) o «da rifare con altro brano». Le correzioni diventano modifiche del piano e, una volta approvate al nuovo ascolto, passaggi di riferimento.</p>
  <div class="legend small muted">__LEGEND__</div>
  <div class="banner" id="banner"></div>
  <div class="row"><span class="small muted" id="dbstate">Collegamento al database in corso…</span><button type="button" id="copy" class="small">Copia JSON</button></div>
</header>
<section class="player" aria-label="Player del set">
  <audio id="audio" preload="metadata" src="__AUDIO__"></audio>
  <div class="setbar" id="bar" role="slider" tabindex="0" aria-label="Posizione nel set" aria-valuemin="0" aria-valuemax="__DURS__" aria-valuenow="0">__BAR__</div>
  <div class="nums" aria-hidden="true">__NUMS__</div>
  <div class="timerow small"><span class="time" id="time">0:00 / __DUR__</span><span class="now muted" id="now"></span></div>
  <div class="row buttons">
    <button type="button" id="back">−10 s</button>
    <button type="button" id="play">▶︎ / ❚❚</button>
    <button type="button" id="fwd">+10 s</button>
  </div>
</section>
<section class="card">
  <details class="tracklist"><summary>Tracklist</summary>
  <ol class="tracks">
__TRACKS__
  </ol>
  </details>
</section>
<h2>Transizioni</h2>
__TRANSITIONS__
__SETFORM__
</main>
<script id="page-data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
const CFG = JSON.parse(document.getElementById("page-data").textContent);
__COMMON_JS__
__CORR_JS__
// ---------------------------------------------------------------- player
const audio = $("#audio"), bar = $("#bar"), fill = $(".fill", bar), head = $(".head", bar);
const timeEl = $("#time"), nowEl = $("#now");
const trkBtns = Array.from(document.querySelectorAll(".trk"));
const trCards = Array.from(document.querySelectorAll(".tr"));
const dur = () => (isFinite(audio.duration) && audio.duration > 0) ? audio.duration : CFG.duration_s;
let win = null;  // {from, to, loop, idx}
let lastTrack = -1, lastTr = -1;
function setWin(w){
  win = w;
  trCards.forEach(c => { $('[data-act="zone"]', c).setAttribute("aria-pressed", String(!!w && w.idx === Number(c.dataset.index))); });
}
function seek(t, keepWin){
  if (!keepWin) setWin(null);
  audio.currentTime = Math.min(Math.max(0, t), Math.max(0, dur() - 0.05));
  paint();
}
function play(){ audio.play().catch(() => {}); }
function paint(){
  const t = audio.currentTime, d = dur();
  const p = Math.min(100, 100 * t / d);
  fill.style.width = p + "%"; head.style.left = p + "%";
  timeEl.textContent = fmt(t) + " / " + fmt(d);
  bar.setAttribute("aria-valuenow", t.toFixed(1));
  bar.setAttribute("aria-valuetext", fmt(t));
  let ti = CFG.tracks.findIndex(x => t >= x.start_s && t < x.end_s);
  if (ti < 0 && CFG.tracks.length && t >= CFG.tracks[CFG.tracks.length - 1].start_s) ti = CFG.tracks.length - 1;
  if (ti !== lastTrack) {
    lastTrack = ti;
    trkBtns.forEach((b, i) => b.classList.toggle("current", i === ti));
    nowEl.textContent = ti >= 0 ? CFG.tracks[ti].position + ". " + CFG.tracks[ti].name : "";
  }
  const tr = CFG.transitions.findIndex(x => t >= x.start_s - CFG.zone_lead && t <= x.end_s + CFG.zone_lead);
  if (tr !== lastTr) { lastTr = tr; trCards.forEach((c, i) => c.classList.toggle("current", i === tr)); }
}
function guard(){
  if (win && audio.currentTime >= win.to) {
    if (win.loop) { audio.currentTime = win.from; const c = trCards.find(x => Number(x.dataset.index) === win.idx); if (c) c._plays += 1; }
    else { audio.pause(); setWin(null); }
  }
}
function tick(){ guard(); paint(); if (!audio.paused) requestAnimationFrame(tick); }
audio.addEventListener("play", () => requestAnimationFrame(tick));
["timeupdate", "seeked", "loadedmetadata", "pause"].forEach(ev => audio.addEventListener(ev, () => { guard(); paint(); }));
const fromPointer = ev => { const r = bar.getBoundingClientRect(); seek(((ev.clientX - r.left) / r.width) * dur()); };
bar.addEventListener("pointerdown", ev => { bar.setPointerCapture(ev.pointerId); fromPointer(ev); });
bar.addEventListener("pointermove", ev => { if (bar.hasPointerCapture(ev.pointerId)) fromPointer(ev); });
bar.addEventListener("keydown", ev => {
  if (ev.key === "ArrowLeft") { seek(audio.currentTime - 5); ev.preventDefault(); }
  if (ev.key === "ArrowRight") { seek(audio.currentTime + 5); ev.preventDefault(); }
  if (ev.key === " ") { audio.paused ? play() : audio.pause(); ev.preventDefault(); }
});
$("#back").addEventListener("click", () => seek(audio.currentTime - 10));
$("#fwd").addEventListener("click", () => seek(audio.currentTime + 10));
$("#play").addEventListener("click", () => { audio.paused ? play() : audio.pause(); });
trkBtns.forEach(b => b.addEventListener("click", () => { seek(Number(b.dataset.startS)); play(); }));

// ---------------------------------------------------------------- transitions
const trByDoc = new Map(CFG.transitions.map(t => [t.doc_id, t]));
function reveal(card){ card.querySelectorAll("[data-reveal]").forEach(e => { e.hidden = false; }); }
function progress(){
  const n = CFG.transitions.filter(t => state.saved[t.doc_id] || state.local[t.doc_id]).length;
  const ok = CFG.transitions.filter(t => { const j = state.saved[t.doc_id] || state.local[t.doc_id]; return j && j.verdict === "approvata"; }).length;
  $("#progress").textContent = n + "/" + CFG.transitions.length + " giudicate · " + ok + " approvate";
}
trCards.forEach(card => {
  const t = trByDoc.get(card.dataset.doc);
  const f = $("form", card), loopBox = $(".loopbox", card);
  card._plays = 0;
  $('[data-act="zone"]', card).addEventListener("click", () => {
    if (win && win.idx === t.index) { audio.pause(); setWin(null); return; }
    const from = Math.max(0, t.start_s - CFG.zone_lead), to = Math.min(dur(), t.end_s + CFG.zone_lead);
    seek(from); setWin({from: from, to: to, loop: loopBox.checked, idx: t.index}); card._plays += 1; play();
  });
  loopBox.addEventListener("change", () => { if (win && win.idx === t.index) win.loop = loopBox.checked; });
  f.addEventListener("change", ev => {
    if (ev.target.name === "verdict-t" + t.index && ev.target.value === "da_correggere") $("details.corr", f).open = true;
  });
  register(t.doc_id, f, {
    fill: j => {
      f.querySelectorAll('input[name="verdict-t' + t.index + '"]').forEach(i => { i.checked = i.value === j.verdict; });
      f.querySelectorAll("input[type=checkbox][value]").forEach(i => { i.checked = (j.tags || []).includes(i.value); });
      fillCorr(f, j.corrections); f.elements.note.value = j.note || "";
    },
    make: () => {
      const v = $('input[name="verdict-t' + t.index + '"]:checked', f);
      if (!v) { setStatus(f, "scegli un verdetto", "err"); return null; }
      const corr = readCorr(f);
      if (v.value === "da_rifare_altro_brano") corr.wrong_b = true;
      return {type: "set_approval_transizione", schema_version: CFG.schema, transition_index: t.index, a: t.a, b: t.b,
              technique: t.type, label: t.label, plan: t.plan, set_start_s: t.start_s, set_switch_s: t.switch_s,
              set_end_s: t.end_s, qa_verdict: t.qa_verdict, verdict: v.value,
              tags: Array.from(f.querySelectorAll("input[type=checkbox][value]:checked")).map(i => i.value),
              corrections: corr, note: f.elements.note.value.trim(), plays_before_save: card._plays, blind: !!CFG.blind,
              position_s: Math.round(audio.currentTime * 1000) / 1000, source_sha256: CFG.source_sha256,
              audio_sha256: CFG.audio_sha256};
    },
    describe: j => {
      const c = corrText(j.corrections);
      return (CFG.verdicts[j.verdict] || j.verdict || "—") + (c ? " · " + c : "") + (j.note ? " · “" + j.note + "”" : "");
    },
    after: j => {
      reveal(card);
      ["approvata", "da_correggere", "da_rifare_altro_brano"].forEach(v => card.classList.toggle("verdict-" + v, j.verdict === v));
      const d = $(".done", card); d.textContent = "✓ " + (CFG.verdicts[j.verdict] || ""); d.className = "done small ok";
      progress();
    }
  });
});
(function(){
  const sec = $("#set-judge"), f = $("form", sec);
  const val = n => { const c = $('input[name="' + n + '"]:checked', f); return c ? c.value : null; };
  register(CFG.set_doc_id, f, {
    fill: j => {
      f.querySelectorAll('input[name="set-verdict"]').forEach(i => { i.checked = i.value === j.set_verdict; });
      CFG.scales.forEach(s => f.querySelectorAll('input[name="set-' + s.key + '"]').forEach(i => { i.checked = Number(i.value) === j[s.key]; }));
      f.elements.note.value = j.note || "";
    },
    make: () => {
      const body = {type: "set_approval", schema_version: CFG.schema, set_verdict: val("set-verdict"), order: CFG.order,
                    n_transitions: CFG.transitions.length, note: f.elements.note.value.trim(), source_sha256: CFG.source_sha256,
                    audio_sha256: CFG.audio_sha256, blind: !!CFG.blind,
                    n_judged: CFG.transitions.filter(t => state.saved[t.doc_id] || state.local[t.doc_id]).length};
      CFG.scales.forEach(s => { const v = val("set-" + s.key); body[s.key] = v ? Number(v) : null; });
      return body;
    },
    describe: j => (CFG.set_verdicts[j.set_verdict] || "—") + " · " +
      CFG.scales.map(s => s.label.toLowerCase() + " " + (j[s.key] != null ? j[s.key] + "/5" : "—")).join(" · ") +
      (j.note ? " · “" + j.note + "”" : "")
  });
})();
progress();
paint();
startDb();
})();
</script>
</body>
</html>
"""


# ======================================================================= cue tournament
def _variant_audio(v: dict, manifest_dir: Path, root: Path):
    recorded = v.get("audio_path")
    if not recorded:
        return None, ("audio_path_mancante", "")
    src = BLA.resolve_review_audio(str(recorded), manifest_dir, root)
    if not src.is_file():
        return None, ("audio_mancante", str(src))
    actual = BLA.sha256_file(src)
    expected = str(v.get("sha256") or "").lower()
    if expected and expected != actual:
        return None, ("sha256_diverso", f"atteso {expected}, trovato {actual}")
    if src.suffix.lower() == ".wav":
        try:
            rate, frames = BLA.wav_info(src)
        except (OSError, ValueError) as exc:
            return None, ("wav_non_leggibile", str(exc))
        length = frames / rate if rate else 0.0
    else:
        length = BLA._float(v.get("duration_s")) or BLA._probe(src).get("duration_s")
    if not length or length <= 0:
        return None, ("durata_sconosciuta", str(src))
    return {"src": src, "sha256": actual, "hash_status": "verificato" if expected else "hash_non_verificabile",
            "duration_s": float(length)}, None


def _cue(v: dict) -> dict:
    return {"a_out_bar": v.get("a_out_bar"), "b_in_bar": v.get("b_in_bar"), "a_out_s": BLA._float(v.get("a_out_s")),
            "b_in_s": BLA._float(v.get("b_in_s")), "b_lead_in_samples": int(v.get("b_lead_in_samples") or 0)}


def comparisons(variants: list[dict]) -> list[dict]:
    """Confronti espliciti tra coppie di varianti: che cosa cambia (solo cue di B, solo cue di A, entrambi, pickup)."""
    out = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            x, y = variants[i]["cue"], variants[j]["cue"]
            same_a = x["a_out_bar"] == y["a_out_bar"]
            same_b = x["b_in_bar"] == y["b_in_bar"]
            same_p = x["b_lead_in_samples"] == y["b_lead_in_samples"]
            if same_a and same_b:
                varies = "identico" if same_p else "solo_pickup_B"
            elif same_a:
                varies = "solo_cue_B"
            elif same_b:
                varies = "solo_cue_A"
            else:
                varies = "cue_A_e_B"
            row = {"x": variants[i]["id"], "y": variants[j]["id"], "varies": varies, "pickup_differs": not same_p}
            if not same_a and x["a_out_bar"] is not None and y["a_out_bar"] is not None:
                row["delta_a_bars"] = int(y["a_out_bar"]) - int(x["a_out_bar"])
            if not same_b and x["b_in_bar"] is not None and y["b_in_bar"] is not None:
                row["delta_b_bars"] = int(y["b_in_bar"]) - int(x["b_in_bar"])
            out.append(row)
    return out


def display_order(run: str, tid: str, ids: list[str], seed: int = 0) -> tuple[list[str], int]:
    """Ordine casuale riproducibile (seme = sha256 di run, torneo e ``seed``)."""
    s = int(hashlib.sha256(f"{run}|{tid}|{seed}".encode()).hexdigest()[:16], 16)
    order = list(ids)
    random.Random(s).shuffle(order)
    return order, s


def build_cue_tournament(manifest_path: Path, out: Path, run: str | None = None, encoder: str = "auto",
                         overwrite: bool = False, log=print, root: Path | None = None, seed: int = 0) -> dict:
    manifest_path = Path(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf8"))
    if manifest.get("schema_version") != SCHEMA_VARIANTS:
        raise ValueError(f"torneo dei cue: atteso {SCHEMA_VARIANTS}, trovato {manifest.get('schema_version')!r}")
    root = Path(root) if root is not None else BLA.repo_root()
    run = run or manifest.get("run") or manifest_path.resolve().parent.name
    out = _prepare_out(out, overwrite)
    enc = None
    tournaments, excluded = [], []
    for tm in manifest.get("tournaments") or []:
        if not isinstance(tm, dict) or not tm.get("id"):
            excluded.append({"kind": "torneo", "id": None, "reason": "id_mancante"})
            continue
        tid = str(tm["id"])
        tech = str(tm.get("technique") or "")
        system = tm.get("system") if isinstance(tm.get("system"), dict) else {}
        vars_in, seen = [], set()
        for v in tm.get("variants") or []:
            v = v if isinstance(v, dict) else {}
            vid = str(v.get("id") or "")

            def skip(reason, detail=""):
                excluded.append({"kind": "variante", "tournament": tid, "id": vid, "reason": reason, "detail": detail})
                log(f"ESCLUSA {tid}/{vid or '?'}: {reason} {detail}".rstrip())

            if not vid or vid in seen:
                skip("id_mancante_o_duplicato"); continue
            vtech = str(v.get("technique") or tech)
            if not tech:
                tech = vtech
            if vtech != tech:
                skip("tecnica_diversa", f"{vtech} invece di {tech}"); continue
            prep, problem = _variant_audio(v, manifest_path.parent, root)
            if problem:
                skip(*problem); continue
            t0, sw, t1 = (BLA._float(v.get(k)) for k in ("transition_start_s", "switch_s", "transition_end_s"))
            length, tol = prep["duration_s"], 0.05
            if None in (t0, sw, t1) or not (-tol <= t0 <= sw + tol and sw <= t1 + tol and t1 <= length + tol):
                skip("marcatori_non_validi", f"{t0}–{t1} s, switch {sw} s in {length:.2f} s"); continue
            src = prep.pop("src")
            fname = BLA.sanitize_segment(f"{tid}__{vid}", 150)
            if src.suffix.lower() == ".mp3":
                dst = out / "audio" / f"{fname}.mp3"
                shutil.copyfile(src, dst)
                how = "copia"
            else:
                enc = enc or BLA.choose_encoder(encoder)
                dst = out / "audio" / f"{fname}.{'wav' if enc == 'wav16' else 'mp3'}"
                BLA.encode(src, dst, enc)
                how = enc
            cut = (t1 - t0) < 0.05
            rule_ok = (abs(length - 2 * BLA.CLIP_SIDE_S) <= tol and abs(sw - BLA.CLIP_SIDE_S) <= tol) if cut else \
                (abs(t0 - BLA.CLIP_SIDE_S) <= tol and abs(length - (t1 + BLA.CLIP_SIDE_S)) <= tol)
            if not rule_ok:
                log(f"AVVISO {tid}/{vid}: clip fuori regola AGENTS.md ({length:.2f} s, transizione {t0:.2f}–{t1:.2f} s)")
            cue = _cue(v)
            is_system = bool(system) and cue["a_out_bar"] == system.get("a_out_bar") and cue["b_in_bar"] == system.get("b_in_bar") \
                and cue["b_lead_in_samples"] == int(system.get("b_lead_in_samples") or 0)
            shifts = {}
            if system.get("a_out_bar") is not None and cue["a_out_bar"] is not None:
                shifts["a_shift_bars"] = int(cue["a_out_bar"]) - int(system["a_out_bar"])
            if system.get("b_in_bar") is not None and cue["b_in_bar"] is not None:
                shifts["b_shift_bars"] = int(cue["b_in_bar"]) - int(system["b_in_bar"])
            qa = v.get("qa") if isinstance(v.get("qa"), dict) else {}
            seen.add(vid)
            vars_in.append({"id": vid, "doc_id": variant_doc_id(run, tid, vid), "audio": f"audio/{dst.name}",
                            "audio_sha256": BLA.sha256_file(dst), "encoding": how, "path_recorded": v.get("audio_path"),
                            "path_resolved": str(src), **prep, "transition_start_s": t0, "switch_s": sw,
                            "transition_end_s": t1, "clip_rule_ok": bool(rule_ok), "technique": tech, "cue": cue,
                            "is_system": is_system, **shifts, "label": str(v.get("label") or ""),
                            "qa_verdict": str(qa.get("verdict") or "non_registrato")})
        if len(vars_in) < 2:
            excluded.append({"kind": "torneo", "id": tid, "reason": "meno_di_due_varianti", "detail": str(len(vars_in))})
            log(f"ESCLUSO torneo {tid}: {len(vars_in)} varianti valide")
            continue
        if not 4 <= len(vars_in) <= 6:
            log(f"AVVISO torneo {tid}: {len(vars_in)} varianti (consigliate 4–6)")
        order, seed_used = display_order(run, tid, [x["id"] for x in vars_in], seed)
        by_id = {x["id"]: x for x in vars_in}
        shown = [by_id[i] for i in order]
        for k, x in enumerate(shown):
            x["shown_number"] = k + 1
        side = lambda k: tm.get(k) if isinstance(tm.get(k), dict) else {"slug": tm.get(k)}  # noqa: E731
        tournaments.append({"id": tid, "doc_id": tournament_doc_id(run, tid), "title": tm.get("title"),
                            "transition_index": tm.get("transition_index"), "a": side("a"), "b": side("b"),
                            "technique": tech, "technique_label": dict(LC.TECHNIQUES).get(tech, tech),
                            "system": system or None, "order": order, "order_seed": seed_used,
                            "variants": shown, "comparisons": comparisons(vars_in)})
    listing = {
        "schema_version": BLA.SCHEMA_OUT, "kind": "cue-tournament", "page_schema": SCHEMA_TOURNAMENT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": run, "blind": True,
        "source_manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(raw).hexdigest(),
                            "schema_version": SCHEMA_VARIANTS, "title": manifest.get("title"),
                            "source_set": manifest.get("source_set")},
        "db": {"collection": BLA.COLLECTION, "tournament_doc_id": "<run>__<torneo>__torneo",
               "variant_doc_id": "<run>__<torneo>__<variante>"},
        "tournaments": tournaments,
        "included": [{"id": x["id"], "tournament": t["id"], "audio": x["audio"], "source_sha256": x["sha256"]}
                     for t in tournaments for x in t["variants"]],
        "excluded": excluded,
    }
    (out / "listening_manifest.json").write_text(json.dumps(listing, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    (out / "index.html").write_text(render_tournament(run, tournaments, manifest.get("title")), encoding="utf8")
    log(f"torneo: {len(tournaments)} tornei, {len(listing['included'])} varianti, {len(excluded)} escluse -> {out}")
    return listing


def _pair_name(side: dict) -> str:
    if not side:
        return "?"
    name = side.get("title") or side.get("slug") or "?"
    return f"{side['artist']} – {name}" if side.get("artist") else str(name)


def _variant_card(t: dict, v: dict) -> str:
    dur = v["duration_s"] or 1.0
    t0, sw, t1 = v["transition_start_s"], v["switch_s"], v["transition_end_s"]
    left, width, sw_pct = 100.0 * t0 / dur, 100.0 * max(0.0, t1 - t0) / dur, 100.0 * sw / dur
    cue = v["cue"]
    info = []
    if cue["a_out_bar"] is not None:
        info.append(f"A esce alla battuta {cue['a_out_bar']}" + (f" ({v['a_shift_bars']:+d})" if "a_shift_bars" in v else ""))
    if cue["b_in_bar"] is not None:
        info.append(f"B dalla battuta {cue['b_in_bar']}" + (f" ({v['b_shift_bars']:+d})" if "b_shift_bars" in v else ""))
    if cue["b_lead_in_samples"]:
        info.append(f"pickup di B {cue['b_lead_in_samples'] / 48000:.2f} s")
    if v["is_system"]:
        info.append("scelta del sistema")
    if v.get("label"):
        info.append(v["label"])
    pre = f"v-{t['id']}-{v['id']}"
    return f"""<article class="version" data-doc="{html.escape(v['doc_id'])}" data-variant="{html.escape(v['id'])}">
  <div class="row vhead"><span class="badge kind">variante {v['shown_number']}</span></div>
  <p class="small muted" data-reveal hidden>{html.escape(' · '.join(info))} · <code>{html.escape(v['id'])}</code></p>
  <audio preload="metadata" controls src="{html.escape(v['audio'])}"></audio>
  <div class="track" role="slider" tabindex="0" aria-label="Posizione nella variante {v['shown_number']}" aria-valuemin="0" aria-valuemax="{dur:.2f}" aria-valuenow="0">
    <div class="region" style="left:{left:.4f}%;width:{width:.4f}%" title="regione di transizione"></div>
    <div class="fill"></div>
    <div class="switch-marker" data-switch-s="{sw:.6f}" style="left:{sw_pct:.4f}%" title="switch"></div>
  </div>
  <div class="row between small muted"><span class="time">0:00</span><span>cambio attorno a {sw:.1f} s</span><span class="dur">{dur:.1f} s</span></div>
  <div class="row buttons">
    <button type="button" data-act="back">−5 s</button>
    <button type="button" data-act="play">▶︎ / ❚❚</button>
    <button type="button" data-act="fwd">+5 s</button>
    <button type="button" data-act="transition">vai alla transizione</button>
  </div>
  <form class="judge">
    <fieldset><legend>Scelte rapide</legend><div class="chips">{_tags(pre)}</div></fieldset>
    {LC.fieldset_html(pre)}
    <label class="note-label">Nota<textarea rows="2" name="note" placeholder="cosa cambia rispetto alle altre…"></textarea></label>
    <div class="row between"><button type="submit" class="primary">Salva nota</button><span class="status small muted">non salvato</span></div>
    <p class="existing small muted"></p>
  </form>
</article>"""


def _tournament_block(t: dict) -> str:
    tid = html.escape(t["id"])
    rows = "".join(
        f'<div class="vrow"><span class="vname">variante {v["shown_number"]}</span>'
        f'<label class="chip"><input type="radio" name="best-{tid}" value="{html.escape(v["id"])}"><span>migliore</span></label>'
        f'<label class="chip"><input type="checkbox" data-acceptable value="{html.escape(v["id"])}"><span>accettabile</span></label></div>'
        for v in t["variants"])
    rows += (f'<div class="vrow"><span class="vname">nessuna</span><label class="chip"><input type="radio" name="best-{tid}" '
             f'value="{NO_BEST}"><span>nessuna va bene</span></label></div>')
    title = t.get("title") or f"{_pair_name(t['a'])} → {_pair_name(t['b'])}"
    idx = f" · transizione {t['transition_index']} del set" if t.get("transition_index") else ""
    cards = "\n".join(_variant_card(t, v) for v in t["variants"])
    return f"""<section class="group" data-doc="{html.escape(t['doc_id'])}" data-tournament="{tid}">
  <h2>{html.escape(str(title))}</h2>
  <p class="muted small"><code>{tid}</code>{idx} · {len(t['variants'])} varianti · tecnica fissa: {html.escape(t['technique_label'])} · cambiano solo i cue</p>
  <div class="versions">
{cards}
  </div>
  <form class="pref">
    <fieldset><legend>Quale variante è la migliore? Quali sono accettabili?</legend><div class="vtable">{rows}</div></fieldset>
    <label class="note-label">Perché (facoltativo)<textarea rows="2" name="note" placeholder="cosa decide la scelta…"></textarea></label>
    <div class="row between"><button type="submit" class="primary">Salva scelta</button><span class="status small muted">non salvato</span></div>
    <p class="existing small muted"></p>
  </form>
</section>"""


def render_tournament(run: str, tournaments: list[dict], heading: str | None = None) -> str:
    data = {
        "kind": "cue-tournament", "schema": SCHEMA_TOURNAMENT, "run": run, "collection": BLA.COLLECTION, "blind": True,
        "lead": BLA.GO_TO_JOIN_LEAD_S, "none": NO_BEST, "tags": [{"key": k, "label": v} for k, v in BLA.DJ_TAGS],
        "tournaments": [{"id": t["id"], "doc_id": t["doc_id"], "technique": t["technique"], "order": t["order"],
                         "transition_index": t.get("transition_index"),
                         "pair": {"a": (t["a"] or {}).get("slug"), "b": (t["b"] or {}).get("slug")},
                         "system": t.get("system"),
                         "variants": [{k: v[k] for k in ("id", "doc_id", "shown_number", "transition_start_s", "switch_s",
                                                         "transition_end_s", "duration_s", "cue", "is_system", "sha256")}
                                      for v in t["variants"]]} for t in tournaments],
    }
    subtitle = f'<p class="small">{html.escape(heading)}</p>' if heading else ""
    blocks = "\n".join(_tournament_block(t) for t in tournaments) or '<p class="muted">Nessun torneo valido.</p>'
    css = _css(BLA.DJ_HTML_TEMPLATE).replace("__CORR_CSS__", "") + LC.CSS + TOURNAMENT_CSS
    return (TOURNAMENT_TEMPLATE
            .replace("__CSS__", css)
            .replace("__SUBTITLE__", subtitle)
            .replace("__RUN__", html.escape(run))
            .replace("__N__", str(len(tournaments)))
            .replace("__GROUPS__", blocks)
            .replace("__COMMON_JS__", COMMON_JS)
            .replace("__CORR_JS__", LC.js())
            .replace("__DATA__", _json(data)))


TOURNAMENT_CSS = """.vtable{display:grid;gap:6px}
.vrow{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.vname{min-width:6.5rem;font-weight:600;font-size:.9rem}
.version.best{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
"""

TOURNAMENT_TEMPLATE = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Torneo dei cue</title>
<style>__CSS__</style>
</head>
<body>
<main>
<header>
  <h1>Torneo dei cue</h1>
  __SUBTITLE__
  <p class="muted small">Run <code>__RUN__</code> · __N__ tornei · stessa tecnica, cambiano solo l'uscita di A e/o l'ingresso di B.</p>
  <p class="small muted">Ordine casuale e alla cieca: i cue di ogni variante compaiono dopo aver salvato la scelta. Scegli la migliore, segna quelle accettabili; le correzioni di una variante sono facoltative.</p>
  <div class="banner" id="banner"></div>
  <div class="row"><span class="small muted" id="dbstate">Collegamento al database in corso…</span><button type="button" id="copy" class="small">Copia JSON</button></div>
</header>
__GROUPS__
</main>
<script id="page-data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
const CFG = JSON.parse(document.getElementById("page-data").textContent);
__COMMON_JS__
__CORR_JS__
const tByDoc = new Map(CFG.tournaments.map(t => [t.doc_id, t]));
// ---------------------------------------------------------------- players
document.querySelectorAll(".version").forEach(card => {
  const sec = card.closest(".group"), t = tByDoc.get(sec.dataset.doc);
  const v = t.variants.find(x => x.doc_id === card.dataset.doc);
  const audio = $("audio", card), track = $(".track", card), fill = $(".fill", card), time = $(".time", card);
  const dur = () => (isFinite(audio.duration) && audio.duration > 0) ? audio.duration : v.duration_s;
  const seek = s => { audio.currentTime = Math.min(Math.max(0, s), Math.max(0, dur() - 0.05)); };
  const paint = () => {
    const p = Math.min(1, audio.currentTime / dur());
    fill.style.width = (p * 100) + "%";
    time.textContent = fmt(audio.currentTime) + " / " + fmt(dur());
    track.setAttribute("aria-valuenow", audio.currentTime.toFixed(1));
  };
  ["timeupdate", "seeked", "loadedmetadata"].forEach(ev => audio.addEventListener(ev, paint));
  audio.addEventListener("play", () => document.querySelectorAll("audio").forEach(a => { if (a !== audio) a.pause(); }));
  const fromPointer = ev => { const r = track.getBoundingClientRect(); seek(((ev.clientX - r.left) / r.width) * dur()); };
  track.addEventListener("pointerdown", ev => { track.setPointerCapture(ev.pointerId); fromPointer(ev); });
  track.addEventListener("pointermove", ev => { if (track.hasPointerCapture(ev.pointerId)) fromPointer(ev); });
  track.addEventListener("keydown", ev => {
    if (ev.key === "ArrowLeft") { seek(audio.currentTime - 1); ev.preventDefault(); }
    if (ev.key === "ArrowRight") { seek(audio.currentTime + 1); ev.preventDefault(); }
  });
  card.querySelectorAll("button[data-act]").forEach(btn => btn.addEventListener("click", () => {
    const act = btn.dataset.act;
    if (act === "back") seek(audio.currentTime - 5);
    else if (act === "fwd") seek(audio.currentTime + 5);
    else if (act === "transition") { seek(v.transition_start_s - CFG.lead); audio.play().catch(() => {}); }
    else if (act === "play") { audio.paused ? audio.play().catch(() => {}) : audio.pause(); }
  }));
  card._plays = 0;
  audio.addEventListener("play", () => { card._plays += 1; });
  const f = $("form", card);
  register(v.doc_id, f, {
    fill: j => {
      f.querySelectorAll("input[type=checkbox][value]").forEach(i => { i.checked = (j.tags || []).includes(i.value); });
      fillCorr(f, j.corrections); f.elements.note.value = j.note || "";
    },
    make: () => ({type: "torneo_variante", schema_version: CFG.schema, tournament_id: t.id, variant_id: v.id,
                  shown_number: v.shown_number, technique: t.technique, pair: t.pair, transition_index: t.transition_index,
                  cue: v.cue, is_system: v.is_system, source_sha256: v.sha256,
                  tags: Array.from(f.querySelectorAll("input[type=checkbox][value]:checked")).map(i => i.value),
                  corrections: readCorr(f), note: f.elements.note.value.trim(), plays_before_save: card._plays, blind: true,
                  position_s: Math.round(audio.currentTime * 1000) / 1000}),
    describe: j => { const c = corrText(j.corrections); const tags = (j.tags || []).map(k => (CFG.tags.find(x => x.key === k) || {label: k}).label);
      return (tags.length ? tags.join(", ") : "nessuna scelta rapida") + (c ? " · correzioni: " + c : "") + (j.note ? " · “" + j.note + "”" : ""); }
  });
});
// ---------------------------------------------------------------- tournament choice
document.querySelectorAll(".group").forEach(sec => {
  const t = tByDoc.get(sec.dataset.doc), f = $("form.pref", sec);
  const cards = Array.from(sec.querySelectorAll(".version"));
  register(t.doc_id, f, {
    fill: j => {
      f.querySelectorAll('input[type=radio]').forEach(i => { i.checked = i.value === j.best; });
      f.querySelectorAll('input[data-acceptable]').forEach(i => { i.checked = (j.acceptable || []).includes(i.value); });
      f.elements.note.value = j.note || "";
    },
    make: () => {
      const b = $('input[type=radio]:checked', f);
      if (!b) { setStatus(f, "scegli la migliore o «nessuna»", "err"); return null; }
      const acc = Array.from(f.querySelectorAll('input[data-acceptable]:checked')).map(i => i.value);
      if (b.value !== CFG.none && !acc.includes(b.value)) acc.push(b.value);
      const best = t.variants.find(x => x.id === b.value);
      const plays = {}; cards.forEach(c => { plays[c.dataset.variant] = c._plays || 0; });
      return {type: "torneo", schema_version: CFG.schema, tournament_id: t.id, technique: t.technique, pair: t.pair,
              transition_index: t.transition_index, system: t.system, order_shown: t.order, best: b.value,
              best_cue: best ? best.cue : null, best_is_system: best ? best.is_system : null, acceptable: acc,
              options: t.variants.map(x => x.id).concat([CFG.none]), note: f.elements.note.value.trim(), plays_before_save: plays,
              blind: true};
    },
    describe: j => {
      const n = id => { const x = t.variants.find(v => v.id === id); return x ? "variante " + x.shown_number : id; };
      return "migliore: " + (j.best === CFG.none ? "nessuna" : n(j.best)) + " · accettabili: " +
        ((j.acceptable || []).map(n).join(", ") || "nessuna") + (j.note ? " · “" + j.note + "”" : "");
    },
    after: j => {
      sec.querySelectorAll("[data-reveal]").forEach(e => { e.hidden = false; });
      cards.forEach(c => c.classList.toggle("best", c.dataset.variant === j.best));
    }
  });
});
startDb();
})();
</script>
</body>
</html>
"""
