"""Canali v4 del giudice automatico, ricavati dai libri (proposte D1-D4, Q-1...Q-4).

Modulo separato da :mod:`musicmix.dj_critic` (il veto v3 resta invariato e chiamabile
come prima). Ogni canale è una misura separata e documentata; nessuno è un voto di
gusto. Fonti (pagine stampate, rilette nel testo con ``tools/libri.py``):

* **D1** ``metric_surprise`` (sorpresa sul *quando*): scarto dei battiti di B dalla
  griglia di A estrapolata oltre la giunta + peso ipermetrico del downbeat di arrivo
  (``dj_phrase.hypermetric_weight``). Huron 2006 p. 116 [T] ("independence of the what
  and when of surprise"), pp. 303-304 [T] ("Syncopation only challenges metric
  perceptions; it never annihilates meter"), pp. 179-180 [A] (ipermetro forte-debole-
  medio-debole), pp. 349-350. ``content_surprise`` (sul *cosa*: NLL di MusicGen alla
  giunta) è riportata come valutabile solo se il metro è rispettato.
* **D2** incertezza prima della giunta (entropia predittiva di MusicGen) separata dalla
  sorpresa realizzata (-log p): Huron p. 55 [T] ("One can be confidently wrong as well as
  confidently right"). Solo diagnostica.
* **D3** finestre di 1 s (riconoscimento dello stile, Huron pp. 207-208 [E]) e 3-5 s
  (memoria a breve termine, p. 228 [E]) attorno alla giunta, con la stabilità dei
  punteggi tra finestre (:func:`window_stability`).
* **D4** il controllo ``dropout`` di ``dj_qa`` (Huron p. 52 [E citata]) quando presente.
* **Q-1** tessitura sovrapposta "busy" (Butler 2006 p. 244 [P riportata]): solo diagnostica.
* **Q-2** backbeat: rullanti di A e B sui battiti 2 e 4 nella sovrapposizione (Webber
  2008 p. 129 [P]: "you start listening for the snare, and matching the snare up").
* **Q-3** downbeat dopo l'ingresso della cassa ("turning the beat around", Butler 2006
  pp. 127, 142 [A]): fase della pulsazione nell'intro senza cassa contro le prime
  battute con cassa.
* **Q-4** due blocchi separati e nominati nell'output: ``technical`` (veto/integrità) e
  ``musical`` (diagnostica, **mai** usato come veto). ``pass`` del QA non è un proxy di
  qualità (Brewster & Broughton cap. 50; Katz p. 177; AGENTS.md).
* **Tecniche volute** (``dj_creative``): ``metadata["creative"]["technique"]``,
  ``params.gap_beats``, ``creative.repetition_checks``: silenzi, loop e build dichiarati
  non vengono penalizzati (stessa logica di ``dj_qa.intended_gap``).

Soglie: tutte [H] salvo dove indicato; nessuna è stata stimata sui giudizi umani.
Quando manca l'ingresso necessario (stem, deck separati, griglia, modello) il canale
restituisce ``None`` con il motivo, mai un valore inventato.

Importa solo numpy a livello di modulo: gira nel venv principale e in ``venv-np1``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from musicmix import dj_critic as C
from musicmix import dj_phrase as P
from musicmix import dj_qa

DJ_CRITIC_BOOKS_VERSION = "musicmix.dj_critic_books.3"  # .3 (22/10/2026, palestra v2): la fase dichiarata non sopprime la misura affidabile del QA (dj_qa.7, ingresso pulito); dropout fail del QA = veto. .2 (21/10/2026): fase dichiarata dal render e tonalità solo con sovrapposizione armonica nel blocco technical

# ------------------------------------------------------------------ costanti
#: D3: finestre dai libri (dj_phrase.BOOK_RULES), accanto a SEAM_SIDE_S / AESTHETIC_WINDOW_S di dj_critic.
STYLE_WINDOW_S = float(P.BOOK_RULES["style_schema_seconds"].value[1])  # 1 s (Huron pp. 207-208)
STM_WINDOWS_S = tuple(float(v) for v in P.BOOK_RULES["stm_seconds"].value[:2])  # 3 e 5 s (Huron p. 228)
WINDOWS_S = (STYLE_WINDOW_S,) + STM_WINDOWS_S
WINDOW_GUARD_S = 0.12  # come CRAFT_GUARD_S

PARAMS: dict[str, tuple[Any, str]] = {
    "metric_dev_ref_ms": (80.0, "[H] grid_term = scarto/80 ms, saturato a 1: 40 ms (flag del mestiere v3) vale 0,5"),
    "metric_ok_grid_term": (0.5, "[H] metro rispettato se lo scarto mediano dei battiti di B è sotto 40 ms ..."),
    "metric_ok_hyper_weight": (0.5, "[H] ... e l'arrivo cade su un downbeat ipermetrico forte o medio (peso >= 0,5, Huron pp. 179-180)"),
    "metric_side_s": (6.0, "[H] lato A (prima) e lato B (dopo) per Beat This!; ~3-5 s di contesto (Huron p. 228) più margine"),
    "backbeat_warn_ms": (15.0, "[P] Webber p. 129 (si allineano i rullanti) + [H] stessa soglia di dj_qa per i downbeat dei deck (15 ms), "
                               "applicata alla mediana con segno (scarto sistematico)"),
    "backbeat_window_ms": (70.0, "[H] ricerca dell'attacco del rullante entro +-70 ms dal battito della griglia"),
    "backbeat_band_hz": ((180.0, 3000.0), "[H] corpo e schiocco del rullante, sopra la cassa (40-120 Hz, dj_form.BANDS) e sotto gli hi-hat"),
    "kick_band_hz": ((40.0, 120.0), "[H] dj_form.BANDS['kick']"),
    "min_pairs": (3, "[H] come dj_qa.MIN_PAIRS"),
    "downbeat_flip_warn_beats": (0.25, "[H] intro fuori fase di almeno un sedicesimo... avviso da un quarto di battito (un ottavo = 0,5)"),
    "downbeat_flip_min_exposed_bars": (1, "[H] l'intro senza cassa deve essere udibile da solo (A già uscito) per almeno una battuta"),
}


def _p(name: str) -> Any:
    return PARAMS[name][0]


# ======================================================== tecniche dichiarate
#: Veto tecnico (raccomandazione v3, runs/dj_critic_v1/REPORT.md §10-11): fase, loudness, tonalità con sovrapposizione.
#: ``vad_at_cut`` resta un flag indicativo (Silero non distingue la parola tagliata sul rap con base).
VETO_FLAGS = ("beat_phase_ms", "loud_jump_db", "chroma_dist")
TEMPO_BREAK_TECHNIQUES = ("tape_stop", "backspin")
LOOP_BUILD_TECHNIQUES = ("halving_build", "loop_extend", "build_into_chorus", "echo_freeze")


# Regola in uso nel set v3 (orchestratore, 21/10/2026; runs/dj_critic_v4/REPORT.md §4.2), ora nel blocco technical:
# * la fase misurata con Beat This! sulla clip è diagnostica quando il render dichiara il downbeat di B sulla stessa
#   griglia con residuo < PHASE_DECLARED_MAX_MS (blend: downbeat_alignment.max_abs_error_ms; ingresso senza
#   sovrapposizione: downbeat di B posto per costruzione sul downbeat di A). Resta vero solo quanto le griglie delle
#   analisi: confidenza "low" della griglia di A o B = avviso [H];
# * la distanza tonale > 1 è un veto solo con sovrapposizione armonica udibile (B beatmatchato sopra A); la coda d'eco
#   sotto B è riportata come nota con il suo livello [H].
PHASE_DECLARED_MAX_MS = 5.0
HARMONIC_OVERLAP_KINDS = ("blend", "loop_extend", "acapella_over", "vocal_swap")


def declared_phase(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Fase di B rispetto alla griglia di A dichiarata dal render (non misurata sulla clip) e confidenza delle griglie."""
    meta = dict(metadata or {})
    prov = meta.get("provenance") or {}
    conf = {k: ((prov.get(k) or {}).get("grid") or {}).get("confidence") for k in ("a", "b")}
    da = meta.get("downbeat_alignment") or {}
    if da.get("max_abs_error_ms") is not None:
        ms, src = float(da["max_abs_error_ms"]), "downbeat_alignment.max_abs_error_ms (beatmatched overlap)"
    elif "b_entry_output_sample" in da or meta.get("kind") == "cut" or meta.get("base_kind") == "cut" or meta.get("creative") \
            or meta.get("phrase"):
        ms, src = 0.0, "B's downbeat placed on A's downbeat by construction (no rhythmic overlap; true only as far as the analysis grids)"
    else:
        ms, src = None, "not declared"
    return {"ms": ms, "source": src, "grid_confidence": conf, "threshold_ms": PHASE_DECLARED_MAX_MS}


def harmonic_overlap(metadata: Optional[Mapping[str, Any]], is_cut: bool) -> dict[str, Any]:
    """Due brani suonano insieme in modo udibile (conta la tonalità)?"""
    meta = dict(metadata or {})
    kind = (meta.get("creative") or {}).get("technique") or meta.get("kind")
    st = meta.get("stretch") or {}
    eff = meta.get("effect") or {}
    tail = eff.get("tail_vs_b_rms_db_first_4s") if eff.get("type") == "echo_out" and not eff.get("tail_stopped_on_b") else None
    if is_cut:
        return {"overlap": False, "why": "cut", "echo_tail_vs_b_db": None}
    if kind in HARMONIC_OVERLAP_KINDS or (st.get("applied") and kind not in TEMPO_BREAK_TECHNIQUES):
        return {"overlap": True, "why": f"{kind}: B beatmatched over A", "echo_tail_vs_b_db": tail}
    return {"overlap": False, "why": f"{kind}: no beatmatched overlap" + (f" (echo tail under B at {tail:+.1f} dB, note only)" if tail is not None else ""),
            "echo_tail_vs_b_db": tail}


def declared_intent(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Cosa il render dichiara come voluto (silenzi, loop, build, rottura del tempo).

    Stessa logica di ``dj_qa.intended_gap`` (tecniche con pausa o ``gap_beats``/``gap_bars`` > 0);
    in più loop/build (ripetizione voluta: Margulis 2014 p. 23 [A]) e i controlli di ripetizione già
    calcolati da ``dj_creative`` (``creative.repetition_checks``, Huron p. 286, Margulis pp. 173-174),
    riportati e non ricalcolati.
    """
    meta = dict(metadata or {})
    cre = meta.get("creative") or {}
    tech = cre.get("technique") or meta.get("kind")
    params = cre.get("params") or {}
    sr = float(meta.get("sample_rate") or 48000)
    gap = dj_qa.intended_gap(meta)
    beat_s = dj_qa._beat_s(meta)
    gap_beats = float(params.get("gap_beats") or 0.0)
    regions = []
    for r in cre.get("regions") or []:
        out = r.get("output") or []
        if len(out) == 2 and out[0] is not None and out[1] is not None:
            regions.append({"name": r.get("name"), "s": [float(out[0]) / sr, float(out[1]) / sr]})
    gap_region = None
    for r in regions:
        nm = str(r["name"] or "").lower()
        if any(k in nm for k in ("stop", "gap", "break", "silence", "mute")) and r["s"][1] > r["s"][0]:
            gap_region = r["s"]
    sw = (meta.get("switch") or {}).get("output_sample")
    if gap_region is None and gap_beats > 0 and beat_s and sw is not None:
        gap_region = [float(sw) / sr - gap_beats * beat_s, float(sw) / sr]
    reps = []
    for chk in cre.get("repetition_checks") or []:
        for w in chk.get("warnings") or []:
            reps.append({"unit_bars": chk.get("unit_bars"), "repeats": chk.get("repeats"), "warning": w})
    return {
        "technique": tech,
        "intended_gap": gap,
        "gap_beats": gap_beats or None,
        "gap_region_s": gap_region,
        "loop_or_build": tech in LOOP_BUILD_TECHNIQUES,
        "tempo_break": tech in TEMPO_BREAK_TECHNIQUES,
        "repetition_warnings": reps,
        "regions_s": regions,
        "source": "metadata (dj_creative/dj_render/dj_effects)" if meta else "none",
    }


# ================================================================ griglia
def _beats_from_tracker(x: np.ndarray, sr: int, t0: float, t1: float, tracker) -> Optional[dict]:
    """Beat This! su [t0, t1] (secondi della clip): battiti, downbeat e retta dei battiti."""
    if tracker is None:
        return None
    t0, t1 = max(0.0, t0), min(len(x) / sr, t1)
    if t1 - t0 < 2.0:
        return None
    seg = np.asarray(x[int(t0 * sr): int(t1 * sr)], dtype=np.float32)
    bt = tracker(seg, sr)
    beats = np.asarray(bt["beats"], float) + t0
    downs = np.asarray(bt["downbeats"], float) + t0
    line = dj_qa._fit_beat_line(beats)
    return {"beats": beats, "downbeats": downs, "line": line}


def output_grid(metadata: Optional[Mapping[str, Any]], n_samples: int, sr: int) -> Optional[dict]:
    """Griglia (secondi della clip) dichiarata dal render: il primo campione della sovrapposizione
    (o lo switch) è un downbeat di A e la battuta dura ``stretch.a_bar_s`` (B è stirato sulla griglia
    di A nei blend). [H] griglia regolare: non segue le micro-variazioni di tempo di A."""
    meta = metadata or {}
    bar = (meta.get("stretch") or {}).get("a_bar_s")
    if not bar:
        return None
    ov = meta.get("overlap") or {}
    sw = meta.get("switch") or {}
    t_ref = ov.get("output_start_sample", sw.get("output_sample"))
    if t_ref is None:
        return None
    msr = float(meta.get("sample_rate") or sr)
    t0 = float(t_ref) / msr
    beat = float(bar) / 4.0
    dur = n_samples / sr
    k0 = -int(math.floor(t0 / beat))
    k1 = int(math.ceil((dur - t0) / beat))
    beats = t0 + beat * np.arange(k0, k1 + 1)
    beats = beats[(beats >= 0) & (beats < dur)]
    idx = np.round((beats - t0) / beat).astype(int)
    return {"beats": beats, "downbeats": beats[idx % 4 == 0], "beat_s": beat, "source": "metadata (overlap start + stretch.a_bar_s)"}


# ===================================================================== D1
def hypermetric_entry(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Posizione ipermetrica (0-3 nel gruppo di 4 battute) dello switch nella griglia di A.

    Riferimento: il cue di A pianificato (``provenance.plan.a_out_s``, scelto su un downbeat di inizio
    frase da ``dj_select``/``dj_set``), altrimenti ``cue_out_a_s``. [H] il cue pianificato è la battuta 0
    del gruppo. Senza questi metadati: ``None`` (nessun valore inventato).
    """
    meta = metadata or {}
    sr = float(meta.get("sample_rate") or 48000)
    sw = meta.get("switch") or {}
    a_sw = sw.get("a_source_sample")
    bar = (meta.get("stretch") or {}).get("a_bar_s")
    plan = (meta.get("provenance") or {}).get("plan") or {}
    ref, src = plan.get("a_out_s"), "provenance.plan.a_out_s (cue on a phrase-start downbeat)"
    if ref is None:
        ref, src = meta.get("cue_out_a_s"), "cue_out_a_s"
    if a_sw is None or not bar or ref is None:
        return {"status": "unknown", "bar_in_group": None, "weight": None,
                "reason": "metadati senza switch.a_source_sample, stretch.a_bar_s o cue di A"}
    bars = (float(a_sw) / sr - float(ref)) / float(bar)
    k = int(round(bars))
    return {"status": "ok", "bars_from_cue": bars, "bar_in_group": k % 4, "weight": P.hypermetric_weight(k % 4),
            "reference": src, "off_bar": abs(bars - k) > 0.1,
            "basis": "[A] Huron 2006 Fig. 10.4 pp. 179-180 (forte-debole-medio-debole); [T] p. 184; [H] cue = battuta 0"}


def metric_surprise(x: np.ndarray, sr: int, start_s: float, switch_s: float, end_s: float, *,
                    tracker: Any = "auto", metadata: Optional[Mapping[str, Any]] = None,
                    intent: Optional[Mapping[str, Any]] = None, joint_tracking: bool = True) -> dict[str, Any]:
    """D1: sorpresa sul *quando* alla giunta.

    Griglia dei battiti di A (retta di Beat This! sul lato A-solo prima della transizione) estrapolata
    oltre la giunta; battiti di B (Beat This! sul lato B-solo dopo la fine) confrontati con il battito
    atteso più vicino (tempo doppio/metà ripiegato). ``grid_dev_ms`` = mediana dello scarto assoluto.
    ``grid_term`` = min(1, scarto / ``metric_dev_ref_ms``); ``hyper_term`` = 1 - peso ipermetrico dello
    switch (:func:`hypermetric_entry`). ``metric_surprise`` = max dei due termini [H] (il peggiore dei
    due modi di arrivare "fuori tempo"). ``metric_ok`` decide se la sorpresa di contenuto è valutabile.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    intent = intent or declared_intent(metadata)
    hyper = hypermetric_entry(metadata)
    out: dict[str, Any] = {"hypermeter": hyper, "basis": "Huron 2006 p. 116, pp. 303-304 [T]; pp. 179-180 [A]"}
    # diagnostica: allineamento dichiarato dal render (griglie delle analisi, deck separati). Sui blend reali di
    # runs/dj_gallery_v2 è < 1 ms mentre Beat This! sulla sola clip può sbagliare di mezzo battito (decks_real.json).
    al = (metadata or {}).get("downbeat_alignment") or {}
    out["render_declared_downbeat_error_ms"] = al.get("median_abs_error_ms")
    if intent.get("tempo_break"):
        out.update(status="not_applicable", reason=f"tecnica {intent['technique']}: la griglia di A è interrotta di proposito",
                   metric_surprise=None, metric_ok=None)
        return out
    tr = dj_qa._get_tracker(tracker) if (tracker is None or isinstance(tracker, str)) else tracker
    if tr is None:
        out.update(status="unavailable", reason="nessun beat tracker", metric_surprise=None, metric_ok=None)
        return out
    side = float(_p("metric_side_s"))
    a_end = start_s - 0.12
    if intent.get("gap_region_s"):
        a_end = min(a_end, float(intent["gap_region_s"][0]) - 0.12)
    if joint_tracking:
        # v4.1: una sola passata di Beat This! sul tratto continuo A-lato -> B-lato (come la fase v3 di dj_critic),
        # poi rette separate sui battiti dei due lati. Con due passate brevi e separate (v4.0, joint_tracking=False)
        # il tracker partiva "a freddo" su 6 s e le rette fallivano su 16 di 28 item giudicati.
        g = _beats_from_tracker(x, sr, a_end - side, end_s + 0.12 + side, tr)
        ga = gb = None
        if g is not None:
            ba = g["beats"][g["beats"] <= a_end]
            bb_ = g["beats"][g["beats"] >= end_s + 0.12]
            ga = {"beats": ba, "line": dj_qa._fit_beat_line(ba)}
            gb = {"beats": bb_, "line": dj_qa._fit_beat_line(bb_)}
    else:
        ga = _beats_from_tracker(x, sr, a_end - side, a_end, tr)
        gb = _beats_from_tracker(x, sr, end_s + 0.12, end_s + 0.12 + side, tr)
    out["tracking"] = "joint (v4.1)" if joint_tracking else "separate sides (v4.0)"
    if not ga or not gb or not ga["line"] or gb["line"] is None:
        out.update(status="unreliable", reason="Beat This!: rette dei battiti non stimabili ai due lati",
                   metric_surprise=None, metric_ok=None)
        return out
    ka, ca = ga["line"]["period_s"], ga["line"]["intercept_s"]
    kb = gb["line"]["period_s"]
    r = kb / ka
    period = ka
    folded = None
    if abs(r / 2 - 1) < 0.08:
        folded = "B a metà tempo"
    elif abs(r * 2 - 1) < 0.08:
        period, folded = ka / 2, "B a tempo doppio"
    bb = gb["beats"][(gb["beats"] >= end_s + 0.12)]
    dev = ((bb - ca) / period + 0.5) % 1.0 - 0.5
    dev_ms = np.abs(dev) * period * 1000.0
    med = float(np.median(dev_ms)) if len(dev_ms) else float("nan")
    gt = float(min(1.0, med / float(_p("metric_dev_ref_ms")))) if np.isfinite(med) else None
    ht = None if hyper.get("weight") is None else float(1.0 - hyper["weight"])
    terms = [v for v in (gt, ht) if v is not None]
    ms = float(max(terms)) if terms else None
    ok = None
    if gt is not None:
        ok = bool(gt < _p("metric_ok_grid_term") and (hyper.get("weight") is None or hyper["weight"] >= _p("metric_ok_hyper_weight")))
    out.update(status="ok", grid_dev_ms=med, grid_dev_beats=float(np.median(np.abs(dev))) if len(dev) else None,
               first_beats_dev_ms=dev_ms[:4].round(1).tolist(), n_b_beats=int(len(bb)),
               tempo_change_pct=float(100.0 * (ka / kb - 1.0)) if folded is None else None, folded=folded,
               a_line_rms_ms=ga["line"]["rms_resid_ms"], b_line_rms_ms=gb["line"]["rms_resid_ms"],
               grid_term=gt, hyper_term=ht, metric_surprise=ms, metric_ok=ok,
               rule="metric_surprise = max(min(1, grid_dev_ms/80), 1 - hypermetric_weight) [H]")
    return out


def content_surprise(nll: Optional[np.ndarray], *, fps: float = C.MUSICGEN_FPS, offset_s: float = 0.0,
                     start_s: float, switch_s: float, end_s: float, metric: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """D1 (*cosa*): picco di NLL di MusicGen alla giunta (``dj_critic.surprisal_seam_stats``).

    Riportata come valutabile (``gated=False``) solo se il metro è rispettato (``metric_ok``): con un
    arrivo fuori griglia la sorpresa del contenuto non è separabile da quella del tempo (Huron
    pp. 303-304: la sincope presuppone il metro). Mai un veto (Huron pp. 24-25: la sorpresa può piacere).
    """
    if nll is None:
        return {"status": "unavailable", "reason": "NLL di MusicGen non calcolata", "value": None}
    nll = np.asarray(nll, dtype=np.float64)
    if nll.ndim == 2:
        nll = nll.sum(0)
    st = C.surprisal_seam_stats(nll, fps=fps, start_s=start_s, switch_s=switch_s, end_s=end_s, offset_s=offset_s)
    ok = (metric or {}).get("metric_ok")
    gated = ok is not True
    val = st.get("peak_z") if st.get("status") == "ok" else None
    return {"status": st.get("status"), "peak_z": val, "excess_post2s": st.get("excess_post2s"),
            "value": None if gated else val, "gated": gated,
            "gate_reason": None if not gated else ("metro non rispettato" if ok is False else "metro non misurato"),
            "stats": st}


# ===================================================================== D2
class MusicGenEntropy(C.MusicGenSurprisal):
    """MusicGen-small non condizionato (stessi pesi e caricamento di ``dj_critic.MusicGenSurprisal``):
    NLL del token realizzato **e** entropia della distribuzione predittiva per codebook e frame."""

    def frame_nll_entropy(self, mono32k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.load()
        torch, dec = self.torch, self.model
        with torch.inference_mode():
            enc = self.encodec.encode(torch.from_numpy(np.ascontiguousarray(mono32k, dtype=np.float32))[None, None, :])
            codes = enc.audio_codes[0, 0].cpu().numpy()
            pad = int(self.cfg.decoder.pad_token_id)
            bos = int(self.cfg.decoder.bos_token_id)
            inp, delayed = C.musicgen_delay_targets(codes, pad, bos)
            enc_h = torch.zeros(1, 1, self.cfg.decoder.hidden_size, dtype=next(dec.parameters()).dtype)
            mask = torch.zeros(1, 1, dtype=torch.long)
            o = dec(input_ids=torch.from_numpy(inp), encoder_hidden_states=enc_h, encoder_attention_mask=mask)
            logp = torch.log_softmax(o.logits.float(), dim=-1)
            ent = -(logp.exp() * logp).sum(-1).cpu().numpy()  # (K, S)
            logp = logp.cpu().numpy()
        nll = C.frame_nll_from_logprobs(logp, delayed, codes.shape[1])
        return nll, entropy_by_frame(ent, codes.shape[1])


def entropy_by_frame(ent_ks: np.ndarray, n_frames: int) -> np.ndarray:
    """Entropia (K, S) in posizioni del pattern "delay" -> (K, F): il codebook k del frame f sta in f+k."""
    k_ = ent_ks.shape[0]
    out = np.zeros((k_, n_frames))
    for k in range(k_):
        out[k] = ent_ks[k, np.arange(n_frames) + k]
    return out


def predictive_uncertainty(nll: Optional[np.ndarray], entropy: Optional[np.ndarray], *, fps: float = C.MUSICGEN_FPS,
                           offset_s: float = 0.0, start_s: float, switch_s: float, end_s: float,
                           windows: Sequence[float] = WINDOWS_S) -> dict[str, Any]:
    """D2 (Huron p. 55): incertezza *prima* della giunta (entropia media della distribuzione
    predittiva negli ultimi ``w`` s prima della transizione) e sorpresa *realizzata* (-log p media nei
    primi ``w`` s dopo lo switch), in campi separati; ``realized_minus_expected`` > 0 = "sicuro e
    sbagliato". Nats, somma sui 4 codebook; z rispetto ai tratti A-solo e B-solo come la sorpresa v1.
    Solo diagnostica: nessuna soglia."""
    if nll is None or entropy is None:
        return {"status": "unavailable", "reason": "entropia di MusicGen non calcolata"}
    nll = np.asarray(nll, dtype=np.float64)
    ent = np.asarray(entropy, dtype=np.float64)
    if nll.ndim == 2:
        nll = nll.sum(0)
    if ent.ndim == 2:
        ent = ent.sum(0)
    t = offset_s + np.arange(len(nll)) / fps
    base = ((t >= offset_s + C.SURPRISAL_BASE_SKIP_S) & (t < start_s - C.SURPRISAL_BASE_GUARD_S)) | (t >= end_s + C.SURPRISAL_BASE_GUARD_S)
    if base.sum() < int(2 * fps):
        return {"status": "insufficient_context"}
    out: dict[str, Any] = {"status": "ok", "base_entropy_median": float(np.median(ent[base])),
                           "base_nll_median": float(np.median(nll[base])), "windows": {}}
    for w in windows:
        pre = (t >= start_s - w) & (t < start_s)
        post = (t >= switch_s) & (t < switch_s + w)
        if not pre.any() or not post.any():
            continue
        e_pre = float(ent[pre].mean())
        r_post = float(nll[post].mean())
        out["windows"][f"{w:g}s"] = {
            "entropy_before": e_pre, "entropy_before_z": float(C.robust_z(e_pre, ent[base])),
            "nll_after": r_post, "nll_after_z": float(C.robust_z(r_post, nll[base])),
            "realized_minus_expected": r_post - float(ent[post].mean()),
        }
    return out


# ===================================================================== D3
def _mfcc_mean(seg: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    m = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    return m[1:].mean(axis=1)


def window_profile(x: np.ndarray, sr: int, start_s: float, end_s: float, *, windows: Sequence[float] = WINDOWS_S,
                   guard_s: float = WINDOW_GUARD_S, nll: Optional[np.ndarray] = None, nll_fps: float = C.MUSICGEN_FPS,
                   nll_offset_s: float = 0.0, switch_s: Optional[float] = None,
                   skip_region_s: Optional[Sequence[float]] = None) -> dict[str, Any]:
    """D3: le stesse misure su finestre di 1, 3 e 5 s ai due lati della transizione.

    Lato prima = [start - w, start - guard] (solo A), lato dopo = [end + guard, end + w] (solo B).
    Misure: salto di loudness (BS.1770), distanza timbrica (MFCC 1-12 medi, euclidea / sqrt(12)),
    distanza dei profili tonali (Krumhansl-Kessler, come ``dj_critic``); con la NLL di MusicGen anche
    l'eccesso di sorpresa nei primi ``w`` s dopo lo switch. ``skip_region_s``: pausa dichiarata da
    escludere dal lato prima (tecniche volute).
    """
    import librosa

    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = C.resample(x, sr, C.SR_CRAFT) if sr != C.SR_CRAFT else x
    srr = C.SR_CRAFT
    a_end = start_s - guard_s
    if skip_region_s:
        a_end = min(a_end, float(skip_region_s[0]) - guard_s)
    out: dict[str, Any] = {"windows": {}, "basis": "Huron 2006 pp. 207-208 [E] (1 s), p. 228 [E] (3-5 s)"}
    base = None
    if nll is not None:
        nl = np.asarray(nll, dtype=np.float64)
        nl = nl.sum(0) if nl.ndim == 2 else nl
        tn = nll_offset_s + np.arange(len(nl)) / nll_fps
        bm = ((tn >= nll_offset_s + C.SURPRISAL_BASE_SKIP_S) & (tn < start_s - C.SURPRISAL_BASE_GUARD_S)) | (tn >= end_s + C.SURPRISAL_BASE_GUARD_S)
        base = float(np.median(nl[bm])) if bm.sum() >= 2 * nll_fps else None
    for w in windows:
        a0, b1 = a_end - w + guard_s, end_s + w
        if a0 < 0 or b1 * srr > len(y) or w < 0.5:
            out["windows"][f"{w:g}s"] = {"status": "outside_clip"}
            continue
        pre = y[int(a0 * srr): int(a_end * srr)]
        post = y[int((end_s + guard_s) * srr): int(b1 * srr)]
        if len(pre) < int(0.4 * srr) or len(post) < int(0.4 * srr):
            out["windows"][f"{w:g}s"] = {"status": "too_short"}
            continue
        row: dict[str, Any] = {"status": "ok"}
        lp, lq = C._st_loudness(pre, srr), C._st_loudness(post, srr)
        row["loud_jump_db"] = float(abs(lq - lp))
        row["timbre_dist"] = float(np.linalg.norm(_mfcc_mean(pre, srr) - _mfcc_mean(post, srr)) / math.sqrt(12))
        c_pre = librosa.feature.chroma_stft(y=pre, sr=srr, n_fft=4096, hop_length=1024).mean(axis=1)
        c_post = librosa.feature.chroma_stft(y=post, sr=srr, n_fft=4096, hop_length=1024).mean(axis=1)
        row["chroma_dist"] = float(1.0 - np.corrcoef(C._key_profile_corr(c_pre), C._key_profile_corr(c_post))[0, 1])
        if base is not None:
            s0 = switch_s if switch_s is not None else end_s
            m = (tn >= s0) & (tn < s0 + w)
            row["nll_excess"] = float(nl[m].mean() - base) if m.any() else None
        out["windows"][f"{w:g}s"] = row
    return out


WINDOW_MEASURES = ("loud_jump_db", "timbre_dist", "chroma_dist", "nll_excess")


def window_stability(profiles: Mapping[str, Mapping[str, Any]], measures: Sequence[str] = WINDOW_MEASURES,
                     windows: Sequence[float] = WINDOWS_S) -> dict[str, Any]:
    """D3: stabilità dei punteggi tra finestre sulle stesse clip (Spearman tra finestre, n clip)."""
    from scipy.stats import spearmanr

    keys = [f"{w:g}s" for w in windows]
    out: dict[str, Any] = {}
    for m in measures:
        res = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = [], []
                for p in profiles.values():
                    wa = ((p or {}).get("windows") or {}).get(keys[i]) or {}
                    wb = ((p or {}).get("windows") or {}).get(keys[j]) or {}
                    va, vb = wa.get(m), wb.get(m)
                    if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):
                        a.append(va)
                        b.append(vb)
                if len(a) >= 4:
                    rho = float(spearmanr(a, b).statistic)
                    res[f"{keys[i]}~{keys[j]}"] = {"n": len(a), "spearman": rho}
                else:
                    res[f"{keys[i]}~{keys[j]}"] = {"n": len(a), "spearman": None}
        out[m] = res
    return out


# ===================================================================== D4
def qa_dropout(qa: Optional[Mapping[str, Any]], intent: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """D4: legge ``checks["dropout"]`` di ``dj_qa`` (presente solo con ``QAConfig(dropout_check=True)``)."""
    chk = ((qa or {}).get("checks") or {}).get("dropout")
    if not chk:
        return {"status": "absent", "reason": "controllo dropout non eseguito dal QA (default spento)"}
    st = chk.get("status")
    declared = chk.get("intended") or (intent or {}).get("intended_gap")
    # dj_qa.7: "fail" = silence longer than the declared pause by >= dropout_min_beats (the QA already discounts it)
    return {"status": st, "longest_s": chk.get("longest_s"), "intended": declared,
            "warning": st == "warn" and not declared, "veto": st == "fail", "basis": "Huron 2006 p. 52 [E citata]"}


def qa_entry_phase(qa: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """dj_qa.7: fase di B a un ingresso pulito misurata in modo affidabile (due rivelatori concordi). Se è ``fail`` la fase
    dichiarata dal render non la sopprime: la dichiarazione vale quanto la griglia d'analisi che il render ha usato."""
    rh = ((qa or {}).get("checks") or {}).get("rhythm") or {}
    en = (rh.get("methods") or {}).get("beat_this_entry") or {}
    if not en:
        return {"status": "absent"}
    return {"status": en.get("status"), "reliable": en.get("reliable"), "offset_ms": en.get("beat_phase_offset_at_switch_ms"),
            "threshold_ms": en.get("threshold_ms"), "veto": en.get("status") == "fail" and bool(en.get("reliable"))}


# ===================================================================== Q-2
def _band_env(x: np.ndarray, sr: int, lo: float, hi: float, smooth_ms: float = 2.0) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    sos = butter(4, [lo, min(hi, 0.45 * sr)], btype="band", fs=sr, output="sos") if lo > 0 else \
        butter(4, hi, btype="low", fs=sr, output="sos")
    e = np.abs(sosfiltfilt(sos, x))
    n = max(1, int(smooth_ms * 1e-3 * sr))
    return np.convolve(e, np.ones(n) / n, mode="same")


def _attack_time(env: np.ndarray, sr: int, t: float, win_s: float) -> Optional[tuple[float, float]]:
    """Istante di massima salita dell'inviluppo entro ``t ± win_s`` (secondi) e la sua forza."""
    i0, i1 = max(1, int((t - win_s) * sr)), min(len(env) - 1, int((t + win_s) * sr))
    if i1 - i0 < 8:
        return None
    d = np.diff(env[i0 - 1: i1 + 1])
    k = int(np.argmax(d))
    if d[k] <= 0:
        return None
    if 0 < k < len(d) - 1:
        a, b, c = d[k - 1], d[k], d[k + 1]
        den = a - 2 * b + c
        k = k + (0.5 * (a - c) / den if den != 0 else 0.0)
    return (i0 - 1 + k + 0.5) / sr, float(d[int(round(min(max(k, 0), len(d) - 1)))])


def backbeat_alignment(deck_a: Optional[np.ndarray], deck_b: Optional[np.ndarray], sr: int, beats_s: Sequence[float],
                       downbeats_s: Sequence[float], start_s: float, end_s: float, *, source: str = "decks") -> dict[str, Any]:
    """Q-2: scarto dei rullanti di B rispetto a quelli di A sui battiti 2 e 4 della sovrapposizione.

    Per ogni battito 2 e 4 (griglia di A) si cerca in ciascun deck l'attacco più ripido della banda del
    rullante (180-3000 Hz) entro ±70 ms; lo scarto è B - A (gli errori della griglia si annullano).
    Come contrasto, lo stesso sui battiti 1 e 3 nella banda della cassa. ``warn`` se la mediana dello
    scarto assoluto supera 15 ms con almeno 3 coppie. Senza deck separati: ``None`` (sul mix A e B non
    sono separabili). Preferire stem drums allineati (``source="drums_stems"``), poi i deck grezzi.
    """
    out: dict[str, Any] = {"basis": "[P] Webber 2008 p. 129 (matching the snare up); [H] 15 ms come dj_qa", "source": source}
    if deck_a is None or deck_b is None:
        out.update(status="unavailable", reason="deck A e B separati (o stem drums) non forniti", backbeat_median_abs_ms=None)
        return out
    beats = np.asarray(beats_s, float)
    downs = np.asarray(downbeats_s, float)
    sel = beats[(beats >= start_s) & (beats <= end_s)]
    if len(sel) < 4 or len(downs) == 0:
        out.update(status="unavailable", reason="sovrapposizione troppo corta o griglia assente", backbeat_median_abs_ms=None)
        return out
    period = float(np.median(np.diff(beats))) if len(beats) > 2 else 0.5
    # posizione nella battuta: indice del battito dal downbeat precedente
    pos = np.array([int(round((b - downs[downs <= b + 1e-3].max()) / period)) % 4 if (downs <= b + 1e-3).any() else -1 for b in sel])
    win = float(_p("backbeat_window_ms")) / 1000.0
    lo, hi = _p("backbeat_band_hz")
    ea, eb = _band_env(deck_a, sr, lo, hi), _band_env(deck_b, sr, lo, hi)
    ka_lo, ka_hi = _p("kick_band_hz")
    ka, kb = _band_env(deck_a, sr, 0.0, ka_hi), _band_env(deck_b, sr, 0.0, ka_hi)

    def pairs(mask, env_a, env_b):
        res, strengths = [], []
        for t in sel[mask]:
            ra, rb = _attack_time(env_a, sr, t, win), _attack_time(env_b, sr, t, win)
            if ra and rb:
                res.append((t, 1000.0 * (rb[0] - ra[0])))
                strengths.append((ra[1], rb[1]))
        if not res:
            return [], []
        s = np.array(strengths)
        # attacchi deboli (sotto il 25 % del più forte del deck) non contano: probabilmente nessun colpo
        keep = (s[:, 0] >= 0.25 * s[:, 0].max()) & (s[:, 1] >= 0.25 * s[:, 1].max())
        return [r for r, k in zip(res, keep) if k], s[keep].tolist()

    back, _ = pairs((pos == 1) | (pos == 3), ea, eb)
    front, _ = pairs((pos == 0) | (pos == 2), ka, kb)
    errs = np.array([e for _, e in back])
    ferrs = np.array([e for _, e in front])
    out.update(n_backbeats=int(len(errs)), backbeat_errors_ms=errs.round(2).tolist(),
               backbeat_median_abs_ms=float(np.median(np.abs(errs))) if len(errs) else None,
               backbeat_median_signed_ms=float(np.median(errs)) if len(errs) else None,
               n_kick_beats=int(len(ferrs)), kick_median_abs_ms=float(np.median(np.abs(ferrs))) if len(ferrs) else None,
               threshold_ms=float(_p("backbeat_warn_ms")))
    if len(errs) < int(_p("min_pairs")):
        out.update(status="unreliable", reason=f"solo {len(errs)} rullanti accoppiati su 2 e 4")
        return out
    # Regola (emendamento 21/10/2026, dopo la verifica su 5 blend reali con allineamento noto < 1 ms): conta lo
    # scarto *sistematico* (mediana con segno), non la mediana del valore assoluto: nei brani reali i colpi vicini al
    # 2 e al 4 non sono sempre lo stesso rullante nei due deck (ghost notes, clap, swing) e danno scarti sparsi di
    # +-50-80 ms a media nulla, che con |scarto| producevano falsi avvisi.
    bad = abs(out["backbeat_median_signed_ms"]) > float(_p("backbeat_warn_ms"))
    out["dispersion_iqr_ms"] = float(np.subtract(*np.percentile(errs, [75, 25])))
    out.update(status="warn" if bad else "pass",
               reason=(f"rullanti di B sfasati di {out['backbeat_median_signed_ms']:+.1f} ms (mediana) sui battiti 2 e 4"
                       if bad else None))
    return out


# ===================================================================== Q-3
def _circ_phase(env: np.ndarray, fps: float, t0: float, period: float, beats_ref: float) -> tuple[float, float]:
    """Fase (in battiti, rispetto alla griglia che ha un battito in ``beats_ref``) e coerenza del pettine."""
    t = t0 + np.arange(len(env)) / fps
    e = np.maximum(env - np.median(env), 0)
    z = np.sum(e * np.exp(-2j * np.pi * (t - beats_ref) / period))
    return float(-np.angle(z) / (2 * np.pi)), float(np.abs(z) / (e.sum() + 1e-12))  # + = attacchi in ritardo


def downbeat_after_kick(signal_b: Optional[np.ndarray], sr: int, beats_s: Sequence[float], downbeats_s: Sequence[float], *,
                        region_s: Sequence[float], a_audible_until_s: float, source: str = "deck_b") -> dict[str, Any]:
    """Q-3 ("turning the beat around", Butler pp. 127, 142 [A]).

    Nella regione di B (``region_s``) trova le battute con cassa (``dj_form.bar_layers``); confronta la
    fase della pulsazione (pettine sull'inviluppo d'attacco a tutto spettro, al periodo del battito)
    nelle battute **senza** cassa prima del suo ingresso con quella delle prime 4 battute **con** cassa.
    ``warn`` se differiscono di almeno 0,25 battiti **e** l'intro senza cassa si sente da sola (A non
    più udibile) per almeno una battuta: con la cassa di A sotto, il metro resta ancorato ad A [H].
    """
    out: dict[str, Any] = {"basis": "[A] Butler 2006 pp. 127, 142; [H] soglie", "source": source}
    if signal_b is None:
        out.update(status="unavailable", reason="segnale di B non disponibile")
        return out
    from musicmix import dj_form

    x = np.asarray(signal_b, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    r0, r1 = float(region_s[0]), float(region_s[1])
    downs = np.asarray(downbeats_s, float)
    beats = np.asarray(beats_s, float)
    downs = downs[(downs >= r0 - 1e-3) & (downs < r1)]
    if len(downs) < 3 or len(beats) < 4:
        out.update(status="unavailable", reason="griglia insufficiente nella regione di B")
        return out
    period = float(np.median(np.diff(beats)))
    layers = dj_form.bar_layers(x, sr, downs.tolist(), beats.tolist())
    kick_on = float(dj_form.PARAMS["kick_on"][0])
    kick = np.asarray(layers["kick"], float)[: len(downs)]
    has = kick >= kick_on
    first = next((i for i in range(len(has) - 1) if has[i] and has[i + 1]), None)
    out["kick_by_bar"] = kick.round(2).tolist()
    if first is None:
        out.update(status="not_applicable", reason="nessun ingresso stabile della cassa nella regione")
        return out
    if first == 0:
        out.update(status="not_applicable", reason="la cassa c'è già alla prima battuta di B: nessun intro senza cassa")
        return out
    bar_s = 4 * period
    intro = (float(downs[0]), float(downs[first]))
    kick_end = float(min(r1, downs[first] + 4 * bar_s))
    hop = max(1, int(0.004 * sr))
    fps = sr / hop

    def ph(t0, t1):
        # inviluppo d'attacco a tutto spettro (salita di |x| lisciato 3 ms): stessa latenza per cassa e clap,
        # a differenza del flusso mel (finestre di ~90 ms, ritardo diverso per le basse frequenze)
        seg = x[int(t0 * sr): int(t1 * sr)]
        n = max(1, int(0.003 * sr))
        e = np.convolve(np.abs(seg), np.ones(n) / n, mode="same")
        d = np.maximum(np.diff(e, prepend=e[:1]), 0.0)
        m = len(d) // hop
        env = d[: m * hop].reshape(m, hop).sum(axis=1)
        return _circ_phase(env, fps, t0 + 0.5 * hop / sr, period, float(beats[0]))

    p_intro, c_intro = ph(*intro)
    p_kick, c_kick = ph(downs[first], kick_end)
    d = (p_intro - p_kick + 0.5) % 1.0 - 0.5
    exposed = max(0.0, intro[1] - max(intro[0], a_audible_until_s))
    exposed_bars = exposed / bar_s
    out.update(kick_entry_s=float(downs[first]), intro_bars=int(first), exposed_intro_bars=float(exposed_bars),
               phase_intro_beats=p_intro, phase_kick_beats=p_kick, phase_diff_beats=float(d),
               coherence_intro=c_intro, coherence_kick=c_kick, threshold_beats=float(_p("downbeat_flip_warn_beats")))
    off = abs(d) >= float(_p("downbeat_flip_warn_beats"))
    if off and exposed_bars >= float(_p("downbeat_flip_min_exposed_bars")):
        out.update(status="warn", reason=(f"intro di B senza cassa sfasata di {d:+.2f} battiti rispetto alla cassa che entra "
                                          f"(udibile da sola per {exposed_bars:.1f} battute): rischio di 'beat girato'"))
    elif off:
        out.update(status="pass", reason="intro sfasata ma coperta dal metro di A (la cassa di A è ancora udibile)")
    else:
        out.update(status="pass", reason=None)
    return out


# ===================================================================== Q-1
def texture_overlap(deck_a: Optional[np.ndarray], deck_b: Optional[np.ndarray], sr: int, beats_s: Sequence[float],
                    downbeats_s: Sequence[float], start_s: float, end_s: float) -> dict[str, Any]:
    """Q-1 (Butler p. 244 [P riportata], "busy"): solo diagnostica. Attacchi per battito
    dell'accompagnamento di ciascun deck (``dj_form.bar_layers``, bande) nelle battute della
    sovrapposizione; ``both_busy_share`` = quota di battute "busy" in entrambi. Misura debole
    (correlazione 0,0-0,5 tra rappresentazioni, ``dj_form.PARAMS['busy_onsets_per_beat']``)."""
    if deck_a is None or deck_b is None:
        return {"status": "unavailable", "reason": "deck separati non forniti"}
    from musicmix import dj_form

    downs = np.asarray(downbeats_s, float)
    downs = downs[(downs >= start_s - 1e-3) & (downs < end_s - 0.5)]
    if len(downs) < 1 or end_s - start_s < 1.0:
        return {"status": "not_applicable", "reason": "nessuna sovrapposizione"}
    edges = np.append(downs, end_s)
    beats = np.asarray(beats_s, float)
    thr = float(dj_form.PARAMS["busy_onsets_per_beat"][0])
    res = {}
    for name, d in (("a", deck_a), ("b", deck_b)):
        L = dj_form.bar_layers(d, sr, edges.tolist(), beats.tolist())
        res[name] = np.asarray(L["onsets_per_beat"], float)[: len(downs)]
    m = min(len(res["a"]), len(res["b"]))
    both = (res["a"][:m] >= thr) & (res["b"][:m] >= thr)
    return {"status": "ok", "bars": int(m), "onsets_per_beat_a": res["a"][:m].round(2).tolist(),
            "onsets_per_beat_b": res["b"][:m].round(2).tolist(), "both_busy_share": float(both.mean()) if m else None,
            "threshold": thr, "diagnostic_only": True, "basis": "[P] Butler 2006 p. 244; misura debole (dj_form)"}


# ================================================================== v4 score
def _technical_loudness(x: np.ndarray, sr: int, start_s: float, end_s: float, intent: Mapping[str, Any],
                        craft: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Salto di loudness come nel mestiere v3 (3 s ai due lati), con il lato prima che finisce prima
    di una pausa dichiarata (un silenzio voluto non è un calo di livello di A)."""
    gap = intent.get("gap_region_s")
    if not gap:
        return None if craft is None else craft.get("loud_jump_db")
    y = np.asarray(x, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    a1 = min(start_s, float(gap[0])) - C.CRAFT_GUARD_S
    a0 = max(0.0, a1 - 3.0)
    b0 = end_s + C.CRAFT_GUARD_S
    pre = y[int(a0 * sr): int(a1 * sr)]
    post = y[int(b0 * sr): int((b0 + 3.0) * sr)]
    if len(pre) < int(0.4 * sr) or len(post) < int(0.4 * sr):
        return None
    return float(abs(C._st_loudness(post, sr) - C._st_loudness(pre, sr)))


def score_v4(x: np.ndarray, sr: int, metadata: Optional[Mapping[str, Any]], *, start_s: float, switch_s: float, end_s: float,
             craft: Optional[Mapping[str, Any]] = None, qa: Optional[Mapping[str, Any]] = None,
             decks: Optional[Mapping[str, Any]] = None, nll: Optional[np.ndarray] = None,
             entropy: Optional[np.ndarray] = None, nll_offset_s: float = 0.0, tracker: Any = "auto",
             windows: bool = True) -> dict[str, Any]:
    """Tutti i canali v4 di una clip, in due blocchi separati (Q-4).

    ``x``: clip (mono o canali) a ``sr``; ``craft``: feature del mestiere v3 già calcolate
    (``dj_critic.craft_features``), se assenti i veto v3 non vengono ripetuti; ``qa``: report ``dj_qa``;
    ``decks``: deck separati in coordinate della clip (``a``/``a_raw``, ``b``/``b_stretched_raw``,
    facoltativi ``drums_a``/``drums_b``) alla stessa ``sr``; ``nll``/``entropy``: MusicGen (K, F) con
    inizio a ``nll_offset_s``.

    ``technical``: veto del mestiere v3 (con la pausa voluta esclusa dal salto di loudness, e il flag di
    fase non applicato alle rotture di tempo dichiarate) + avvisi nuovi (dropout non dichiarato,
    backbeat, beat girato). ``musical``: solo diagnostica, mai veto.
    """
    x = np.asarray(x, dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim > 1 else x
    meta = dict(metadata or {})
    intent = declared_intent(meta)
    tr = dj_qa._get_tracker(tracker) if (tracker is None or isinstance(tracker, str)) else tracker
    is_cut = (end_s - start_s) < 0.5
    # ------------------------------------------------------------ technical
    veto, warnings, notes, indicative = [], [], [], []
    dphase = declared_phase(meta)
    hov = harmonic_overlap(meta, is_cut)
    if craft is not None:
        feats = dict(craft)
        lj = _technical_loudness(mono, sr, start_s, end_s, intent, craft)
        if intent.get("gap_region_s") and lj is not None:
            notes.append(f"salto di loudness misurato prima della pausa dichiarata ({intent['intended_gap']}): "
                         f"{feats.get('loud_jump_db', float('nan')):.2f} -> {lj:.2f} dB")
            feats["loud_jump_db"] = lj
        for f in C.craft_flags(feats, is_cut=is_cut):
            if f["flag"] == "beat_phase_ms" and intent.get("tempo_break"):
                notes.append(f"flag di fase non applicato: {intent['technique']} interrompe la griglia di proposito")
                continue
            if f["flag"] not in VETO_FLAGS:
                indicative.append(f)  # vad_at_cut: Silero non informativo nella dose-risposta (REPORT v3 §10)
                continue
            if f["flag"] == "beat_phase_ms" and dphase["ms"] is not None and dphase["ms"] < PHASE_DECLARED_MAX_MS:
                indicative.append(dict(f, diagnostic_only=True,
                                       why=f"render declares {dphase['ms']:.2f} ms ({dphase['source']}); Beat This! on the mixed clip "
                                           "gives false positives (runs/dj_critic_v4/REPORT.md §4.2)"))
                low = [k.upper() for k, v in dphase["grid_confidence"].items() if v == "low"]
                if low:
                    warnings.append({"check": "grid_confidence", "status": "warn", "value": dphase["grid_confidence"],
                                     "message": f"phase guaranteed only by the analysis grids, and the grid of {'/'.join(low)} has low confidence"})
                continue
            if f["flag"] == "chroma_dist" and not hov["overlap"]:
                indicative.append(dict(f, diagnostic_only=True, why=f"no audible harmonic overlap ({hov['why']})"))
                continue
            veto.append(f)
    else:
        notes.append("feature del mestiere v3 non fornite: veto v3 non ricalcolati")
    drop = qa_dropout(qa, intent)
    entry = qa_entry_phase(qa)
    if entry.get("veto"):
        veto.append({"flag": "qa_entry_phase_ms", "value": abs(float(entry["offset_ms"])), "threshold": entry.get("threshold_ms"),
                     "message": f"B's beat {entry['offset_ms']:+.0f} ms from A's pulse at a clean entry (dj_qa.7, reliable: two detectors "
                                "agree); the phase declared by the render does not suppress it"})
    if drop.get("veto"):
        veto.append({"flag": "dropout", "value": drop.get("longest_s"), "threshold": None,
                     "message": "undeclared silence in the seam (dj_qa.7 dropout fail; Huron p. 52)"})
    elif drop.get("warning"):
        warnings.append({"check": "dropout", "status": "warn", "value": drop.get("longest_s"),
                         "message": "silenzio nella giunta non dichiarato dal render (Huron p. 52)"})
    d = dict(decks or {})
    grid = output_grid(meta, len(mono), sr)
    if grid is None and tr is not None:
        g = _beats_from_tracker(mono, sr, max(0.0, start_s - 8.0), start_s, tr)
        if g and g["line"]:
            k, c = g["line"]["period_s"], g["line"]["intercept_s"]
            dur = len(mono) / sr
            bts = c + k * np.arange(-int(c / k) - 1, int((dur - c) / k) + 2)
            bts = bts[(bts >= 0) & (bts < dur)]
            dbs = np.asarray(g["downbeats"], float)
            if len(dbs):
                ph = np.round((dbs - c) / k).astype(int) % 4
                ref = int(np.bincount(ph, minlength=4).argmax())
                idx = np.round((bts - c) / k).astype(int)
                grid = {"beats": bts, "downbeats": bts[(idx - ref) % 4 == 0], "beat_s": k, "source": "Beat This! sul lato A"}
    da = d.get("drums_a")
    db = d.get("drums_b")
    bsrc = "drums_stems"
    if da is None or db is None:
        da = d.get("a_raw", d.get("a"))
        db = d.get("b_stretched_raw", d.get("b"))
        bsrc = "decks (mix per deck, banda del rullante)"
    if grid is not None and not is_cut:
        bb = backbeat_alignment(da, db, sr, grid["beats"], grid["downbeats"], start_s, end_s, source=bsrc)
        if bsrc != "drums_stems":
            # Proxy non validato: su 5 blend reali (runs/dj_critic_v4/decks_real.json) il backbeat sul mix di ciascun
            # deck non concorda con quello sugli stem drums (g02: stem +5 ms, proxy +32 ms; allineamento del render
            # < 1 ms). Riportato come diagnostica, mai come avviso; il canale tecnico resta None senza stem drums.
            bb = {"status": "unavailable", "backbeat_median_abs_ms": None, "proxy_diagnostic": bb,
                  "reason": "stem drums non forniti: il proxy dei deck non è affidabile (diagnostica in proxy_diagnostic)"}
    else:
        bb = {"status": "not_applicable" if is_cut else "unavailable", "backbeat_median_abs_ms": None,
              "reason": "cut: nessuna sovrapposizione" if is_cut else "griglia non disponibile"}
    if bb.get("status") == "warn":
        warnings.append({"check": "backbeat", "status": "warn", "value": bb["backbeat_median_abs_ms"],
                         "threshold": bb["threshold_ms"], "message": bb["reason"]})
    # Q-3: B dal deck (blend) oppure dalla parte B-solo della clip
    sig_b = d.get("b_stretched_raw", d.get("b"))
    if sig_b is not None and grid is not None:
        q3 = downbeat_after_kick(sig_b, sr, grid["beats"], grid["downbeats"], region_s=(start_s, len(mono) / sr),
                                 a_audible_until_s=end_s, source="deck_b")
    elif tr is not None and len(mono) / sr - end_s > 4.0:
        gb = _beats_from_tracker(mono, sr, end_s, len(mono) / sr, tr)
        if gb and len(gb["downbeats"]) >= 3:
            q3 = downbeat_after_kick(mono, sr, gb["beats"], gb["downbeats"], region_s=(end_s, len(mono) / sr),
                                     a_audible_until_s=end_s, source="clip B-solo + Beat This!")
        else:
            q3 = {"status": "unavailable", "reason": "Beat This! senza downbeat sul lato B"}
    else:
        q3 = {"status": "unavailable", "reason": "né deck B né beat tracker"}
    if q3.get("status") == "warn":
        warnings.append({"check": "downbeat_after_kick", "status": "warn", "value": q3.get("phase_diff_beats"),
                         "threshold": q3.get("threshold_beats"), "message": q3["reason"]})
    status = "veto" if veto else ("warn" if warnings else "ok")
    technical = {"status": status, "veto_flags": veto, "warnings": warnings, "notes": notes, "indicative_flags": indicative,
                 "n_veto": len(veto), "n_warn": len(warnings),
                 "qa": {"verdict": (qa or {}).get("verdict"), "dropout": drop, "entry_phase": entry,
                        "note": "pass del QA = integrità tecnica, non qualità musicale (Q-4)"},
                 "backbeat": bb, "downbeat_after_kick": q3, "declared_phase": dphase, "harmonic_overlap": hov,
                 "rule": "veto = flag del mestiere v3: fase > 40 ms solo se il render non dichiara un residuo < 5 ms; loudness > 3 dB "
                         "(non con la pausa dichiarata); tonalità > 1 solo con sovrapposizione armonica udibile (B beatmatchato su A); "
                         "warn = dropout non dichiarato, backbeat > 15 ms, beat girato, griglia a bassa confidenza quando la fase è "
                         "garantita solo dalle griglie"}
    # -------------------------------------------------------------- musical
    ms = metric_surprise(mono, sr, start_s, switch_s, end_s, tracker=tr, metadata=meta, intent=intent)
    cs = content_surprise(nll, offset_s=nll_offset_s, start_s=start_s, switch_s=switch_s, end_s=end_s, metric=ms)
    unc = predictive_uncertainty(nll, entropy, offset_s=nll_offset_s, start_s=start_s, switch_s=switch_s, end_s=end_s)
    wp = window_profile(mono, sr, start_s, end_s, nll=nll, nll_offset_s=nll_offset_s, switch_s=switch_s,
                        skip_region_s=intent.get("gap_region_s")) if windows else {"status": "skipped"}
    tex = texture_overlap(d.get("a_raw", d.get("a")), d.get("b_stretched_raw", d.get("b")), sr,
                          grid["beats"] if grid else [], grid["downbeats"] if grid else [], start_s, end_s) \
        if grid is not None and not is_cut else {"status": "not_applicable" if is_cut else "unavailable"}
    musical = {"diagnostic_only": True, "metric_surprise": ms, "content_surprise": cs, "uncertainty": unc,
               "windows": wp, "texture": tex, "repetition_warnings": intent["repetition_warnings"],
               "note": "diagnostica e gusto: mai usata come veto; nessuna soglia stimata sui giudizi"}
    return {"version": DJ_CRITIC_BOOKS_VERSION, "intent": intent, "grid_source": (grid or {}).get("source"),
            "technical": technical, "musical": musical}


def summary_v4(v4: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """Scalari per la validazione (dichiarati in PREREGISTRATION di runs/dj_critic_v4)."""
    t, m = v4.get("technical") or {}, v4.get("musical") or {}
    ms = m.get("metric_surprise") or {}
    cs = m.get("content_surprise") or {}
    w = ((m.get("windows") or {}).get("windows") or {})
    unc = ((m.get("uncertainty") or {}).get("windows") or {})
    out = {
        "v4_n_veto": float(t.get("n_veto", 0)),
        "v4_n_issues": float(t.get("n_veto", 0) + t.get("n_warn", 0)),
        "metric_surprise": ms.get("metric_surprise"),
        "metric_grid_dev_ms": ms.get("grid_dev_ms"),
        "hyper_weight": (ms.get("hypermeter") or {}).get("weight"),
        "content_surprise_gated": cs.get("value"),
        "content_surprise_raw": cs.get("peak_z"),
        "backbeat_ms": (t.get("backbeat") or {}).get("backbeat_median_abs_ms"),
    }
    for k in ("1s", "3s", "5s"):
        for f in ("loud_jump_db", "timbre_dist", "chroma_dist", "nll_excess"):
            out[f"win{k}_{f}"] = (w.get(k) or {}).get(f)
        u = unc.get(k) or {}
        out[f"ent_before_{k}"] = u.get("entropy_before_z")
        out[f"nll_after_{k}"] = u.get("nll_after_z")
        out[f"confident_wrong_{k}"] = u.get("realized_minus_expected")
    return out
