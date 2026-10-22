"""Controllo automatico pass/fail di un render DJ (dj_render / dj_effects).

Dato un render (clip audio + metadati; facoltativi i deck separati, gli stem
vocali e le parole ASR di A), misura sul PCM, in modo indipendente dalle
griglie usate per renderizzare, sette difetti tecnici del mestiere DJ e
restituisce un dict JSON-serializzabile con esito ``pass`` / ``warn`` /
``fail`` e motivi per ciascun controllo.

Controlli (``checks``):

* ``rhythm``: rianalisi con Beat This! (checkpoint del progetto, logit
  50 fps con interpolazione parabolica del picco) dei deck A e B nella
  sovrapposizione, più cross-correlazione del flusso spettrale tra i deck
  (``decks``); senza deck, Beat This! sulla sola clip (regioni A-sola e
  B-sola, rette di beat estrapolate allo switch: misura indicativa).
  Ambiguità di fase del tracker: se l'errore (mediana dei downbeat sui deck,
  o scarto di fase allo switch sulla clip) dista da mezzo battito meno della
  soglia del ritmo e, per entrambi i deck, allin1 o madmom concordano con la
  griglia usata per il render (``grid_support``: blocco ``tempo`` di
  dj_select), il fail diventa ``warn`` "possibile ambiguità di fase del
  tracker": Beat This! può agganciarsi ai controtempi (MAFIA, galleria v1).
* ``double_bass``: energia < 200 Hz dei due deck (post EQ/fader) entrambi
  sopra -6 dB dal proprio picco per più di 100 ms consecutivi.
* ``loudness_jump``: LUFS short-term (3 s, EBU Tech 3341); cut: prima vs dopo
  lo switch; effetti e blend: A senza effetto prima dell'inizio della
  transizione vs B dopo la fine (effetti: dopo lo switch), più nessun picco
  momentary dentro l'effetto/sovrapposizione sopra A prima.
* ``true_peak``: true peak 4x (BS.1770-4) e campioni a fondo scala.
* ``vocal_clash``: voce attiva e udibile in entrambi i deck (dopo fader e
  ducking) per più di 1 s in totale.
* ``verse_cut``: per un'uscita brusca di A senza effetto, parola di A
  attiva entro +-150 ms dal punto in cui A esce del tutto; per un'uscita con
  effetto (echo, filter, build, loop, stutter, spin; dal .5) parola che
  attraversa il punto in cui finisce la riproduzione secca e continua di A
  (``a_dry_end``: ``effect.dry_a_cut_output_sample`` o la catena iniziale di
  ``a_map``), regola [H] in ``EFFECT_WORD_EPS_S``; uscita a fader graduale:
  ``not_applicable``.
* ``b_entry`` (dal .6, 21/10/2026, simmetrico di ``verse_cut``): nel punto in cui B diventa udibile (lead-in
  dichiarato in ``meta['b_entry']``, altrimenti deck B misurato o metadati dell'ingresso) nessuna parola di B
  è troncata (inizia prima e finisce dopo di oltre ``b_entry_eps_s``) e la prima frase udibile di B non è
  iniziata prima (parola precedente di B a meno di ``b_rest_beats`` battiti dalla prima udibile); B che sale
  col fader (blend, sovrapposizione) è ``not_applicable``. Regole [H] in ``THRESHOLD_ORIGIN``.
* ``duration``: durata della clip secondo AGENTS.md.
* ``phrase`` (dal .7): il cue di B cade su un inizio frase di B (tabella ``phrases`` dell'analisi usata dal render).
* ``click`` (dal .7): discontinuità alle giunte (energia sopra 20,5 kHz nei 5 ms attorno a switch, bordi delle
  mappe, fine del dry, ingresso udibile di B, rispetto al resto della clip attorno al punto).
* ``echo_tempo`` (dal .7): ripetizioni di A dopo la sua uscita (copie dell'ultimo battito di A trovate con un filtro
  adattato) con passo che non è un multiplo di 1/4 o 1/3 di battito.
* ``dropout`` (dal .7 acceso di default, ``fail``): silenzio non dichiarato nella giunta.

Changelog .7 (22/10/2026, palestra dei difetti v1 -> v2, ``conoscenza_mix/_meta/PALESTRA_DIFETTI_2026-10-21.md`` §6):

1. Ritmo a un ingresso pulito (cut, echo/filter out, build, halving, break: B non suona sotto A): nuova misura
   ``beat_this_entry`` (rette di beat su deck A prima dell'uscita e su deck B dopo il downbeat, fase allo switch letta
   nel battito dichiarato di ciascun lato). Conta solo se affidabile (abbastanza battiti, rette diritte, consenso tra
   le due metà della finestra, rivelatore di flusso d'onset concorde entro 0,2 battiti, Beat This! non già fuori
   griglia nell'analisi): allora sopra ``ENTRY_PHASE_FAIL_MS`` è ``fail`` qualunque fase dichiari il render; se non è
   affidabile il controllo è ``unavailable`` (warn, mai pass). Sostituisce la via mix-only (15 ms nel cut: falsi
   allarmi sulle continuazioni naturali; "warn al massimo" negli effetti: sfasamenti non rifiutati). Soglia [E]
   calibrata sul rumore delle giunte naturali (``tools/calibrate_qa_noise.py``). L'ambiguità di fase del tracker
   richiede ora anche che Beat This! sia fuori griglia nell'analisi di uno dei due brani (``grid_phase_support``:
   all'ottava diversa la coincidenza 0,5 non è "fuori griglia").
2. ``phrase``: B fuori frase (±1 battuta) non era misurato.
3. ``dropout``: ``fail`` (era ``warn``) se il silenzio supera di ``dropout_min_beats`` la pausa dichiarata
   (``gap_beats``/``gap_bars``; prima una pausa dichiarata esentava anche un buco più lungo); acceso di default.
4. ``click`` ed ``echo_tempo``: nuovi.
5. ``dj_creative.qa_inputs`` passa sempre il deck B (doppio basso misurabile anche in halving build / break).

Copertura (dal .5): ogni controllo ha ``coverage`` in ``measured`` /
``unavailable`` / ``not_applicable`` / ``intended_gap``; il report ha
``coverage`` con i controlli richiesti non misurati, e un controllo richiesto
``unavailable`` trasforma un ``pass`` in ``warn`` (``QAConfig.require_coverage``).
Un controllo non applicabile ha ``status: not_applicable`` (fino al .4 era
``pass``, cioè un pass non misurato).

Integrità tecnica non è qualità musicale (AGENTS.md): un ``pass`` dice solo
che nessuno di questi difetti espliciti è stato misurato.

Coordinate: tutti i tempi sono nella clip in uscita (secondi o campioni);
``stems_to_output`` e ``words_to_output`` convertono dalle sorgenti usando i
metadati del render; B stirato passa per ``stretch.time_map`` (mappa a tratti
del render) quando c'è, altrimenti per la retta dichiarata in
``inputs.b_projection``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .paths import models_dir

DJ_QA_VERSION = "musicmix.dj_qa.7"  # .7 (22/10/2026, palestra dei difetti v2): vedi il changelog nel docstring. .6 (21/10/2026, INGRESSO_B): controllo b_entry (parola o frase di B troncata dove B diventa udibile), richiesto. .5 (21/10/2026, P0.6): stato di copertura per controllo e "pass" solo con i controlli richiesti misurati; parola tagliata quando A esce con un effetto; B proiettato col time-map del render. # (+ opzione rhythm_on_raw_a, spenta, 21/10/2026: versione invariata perché i default non cambiano) .2: tracker phase ambiguity warn (20/10/2026); .3: loudness windows outside effects/overlap (21/10/2026); .4: controllo opzionale dropout (libri, 21/10/2026, spento)
ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- thresholds
# [H] = ipotesi ingegneristica (nostra o dell'utente), non validata da ascolti.
# [S] = standard / letteratura, citato per la definizione della misura, che
#       non convalida automaticamente la soglia applicata a una transizione.

# [H] Soglia dell'utente (20/10/2026). Ordine di grandezza delle asincronie di
# attacchi percussivi che diventano "flam" udibili (decine di ms); nessuna
# fonte della biblioteca fissa 15 ms per il beatmatching.
DOWNBEAT_MEDIAN_MAX_MS = 15.0
# [H] Deriva lineare dell'errore tra inizio e fine della sovrapposizione
# (pendenza Theil-Sen x durata); fail solo se l'IC 95% esclude 0, altrimenti warn.
DRIFT_MAX_MS = 15.0
# [H] Coppie minime di downbeat/finestre per considerare la misura valida.
MIN_PAIRS = 3
# [H] Correlazione minima del flusso spettrale tra deck perché il ritardo
# stimato sia affidabile (stessa soglia diagnostica di dj_render).
XCORR_MIN = 0.5
# [H] Modalità senza deck: la retta dei beat di Beat This! in A-sola/B-sola deve
# avere residuo RMS <= 20 ms (un frame), altrimenti la misura è inaffidabile.
MIX_LINE_MAX_RMS_MS = 20.0

# [H] Ambiguità di fase del tracker: un errore a mezzo battito (+-soglia del
# ritmo) con la griglia del render confermata da un provider indipendente
# (allin1 o madmom, coincidenza dei beat >= 0,8 entro 70 ms come in
# dj_select.choose_grid) è trattato come Beat This! sui controtempi: warn.
PHASE_AMBIGUITY_PROVIDERS = ("allin1", "madmom")
PHASE_AGREE_MIN = 0.8

# [H] Doppio basso: soglia dell'utente. 200 Hz = taglio del bass swap del
# renderer (convenzione di mixer DJ a 3 bande, non un risultato sperimentale).
BASS_HZ = 200.0
BASS_REL_DB = -6.0
BASS_MAX_S = 0.100
# [H] Finestra RMS dell'inviluppo dei bassi: 150 ms liscia i singoli colpi di
# kick/basso; uno swap istantaneo produce al massimo 0.5 x 150 = 75 ms di
# sovrapposizione sopra -6 dB (copertura della finestra tra 25% e 75%).
BASS_RMS_WINDOW_S = 0.150

# [S] Finestra short-term 3 s: EBU Tech 3341 / EBU R128 (K-weighting BS.1770).
# [H] Soglie di salto 3 dB (fail) e 2 dB (warn): ipotesi dell'utente.
LOUDNESS_WINDOW_S = 3.0
LOUDNESS_FAIL_DB = 3.0
LOUDNESS_WARN_DB = 2.0
# [H] Dove si misura il salto (ipotesi ingegneristica, correzione 21/10/2026).
# * cut: 3 s prima vs 3 s dopo lo switch (invariato).
# * effetti (echo_out, filter_out, filter verso il ritornello) e blend: A SENZA
#   effetto (3 s subito prima dell'inizio della transizione) vs B dopo la fine
#   della transizione (effetti: dopo lo switch, dove B entra pulito). Le
#   finestre attorno allo switch cadevano dentro l'effetto, più basso per
#   costruzione (filtro passa-alto, eco attenuata) o dentro la dissolvenza di
#   A: misuravano il buco voluto e bocciavano proprio le transizioni votate
#   9/10 (set v1: 42 tentativi su 49 falliti, soprattutto per questo). La
#   finestra attorno allo switch resta nel report solo come informazione.
# * dentro la regione dell'effetto/sovrapposizione un buco di livello è atteso,
#   un picco no: il massimo della loudness momentary (400 ms, BS.1770 / EBU
#   Tech 3341, passo 100 ms) nella regione non deve superare il massimo
#   momentary di A prima (per i blend: il maggiore tra A prima e B dopo, perché
#   la sovrapposizione sale legittimamente verso B). Confronto picco con picco
#   sulla stessa misura; stesse soglie 3/2 dB del salto (non validate). Nei
#   blend il picco dà al più warn: la somma dei deck (es. bassi senza swap) ha
#   già il suo controllo (double_bass) e la regola richiesta è per gli effetti.
LOUDNESS_MOMENTARY_S = 0.4
LOUDNESS_MOMENTARY_HOP_S = 0.1

# [S] -1 dBTP: livello massimo di true peak di EBU R128 (misura BS.1770-4,
# sovracampionamento 4x); qui applicato come tetto della clip d'ascolto.
TRUE_PEAK_MAX_DBTP = -1.0
FULL_SCALE = 0.999  # |x| >= -0.009 dBFS conta come campione a fondo scala

# [H] Scontro di voci: soglia dell'utente (il renderer fa ducking da 2 s, MosaikBox).
VOCAL_CLASH_MAX_S = 1.0
# [H] Attività vocale da stem Demucs: RMS 20 ms > max(floor, p95 - 24 dB),
# lacune < 150 ms chiuse (stessa euristica acustica di dj_render; non ASR).
VOCAL_HOP_S = 0.02
VOCAL_REL_DB = -24.0
VOCAL_ABS_FLOOR = 0.002
VOCAL_CLOSE_S = 0.15

# [H] Verso tagliato: soglia dell'utente. Uscita "brusca" se A passa da -6 dB
# al silenzio in meno di 100 ms (un cut con micro-crossfade lo è sempre).
WORD_GUARD_S = 0.150
ABRUPT_EXIT_S = 0.100
EXEMPT_EXIT_EFFECTS = ("echo", "filter", "delay", "reverb")

# Buco nella giunta (proposta D4 dei libri): un battito mancante dove era atteso non è "assenza neutra" ma un evento
# (Huron 2006 p. 52 [E citata]: mismatch negativity per l'omissione di un evento atteso). Soglie [H].
DROPOUT_REL_DB = -30.0
DROPOUT_MIN_BEATS = 0.5
# .7 [H]: with an exit effect or technique the silence is searched from 0.1 s before B's entry on (inside the effect a dip
# is part of it: natural echo outs dip below -30 dB between repeats for > 0.5 beat, calibration of 22/10/2026).
DROPOUT_PRE_S = 0.1
# .7 [E]/[H]: inside the exit effect (A side, before B's entry) the silence counts only from DROPOUT_EFFECT_BEATS on
# (natural echo/filter outs dip there by themselves; calibrated on natural junctions, calibration.json).
DROPOUT_EFFECT_BEATS = 1.5
INTENDED_GAP_TECHNIQUES = ("break_before_entry", "tape_stop", "backspin")

# Parola tagliata quando A esce con un effetto (P0.6, 21/10/2026) [H]: con echo/filter/build/loop/stutter il segnale
# *secco* e continuo di A si interrompe in un punto preciso (fine del dry nell'echo, taglio dopo il filtro, inizio del
# loop o dello stutter); una coda (eco, ripetizione) non completa la parola. La parola è tagliata se inizia prima e
# finisce dopo quel punto di almeno EFFECT_WORD_EPS_S (incertezza dei confini ASR). Se nella banda della voce
# (1-4 kHz, ultimi EFFECT_FADED_WIN_S prima del punto) A è già sotto EFFECT_FADED_DB rispetto ai 3 s prima della
# transizione, il controllo non si applica.
EFFECT_WORD_EPS_S = 0.05
EFFECT_FADED_DB = -20.0
EFFECT_STEM_WIN_S = 0.06
EFFECT_FADED_WIN_S = 0.5  # several beats: between two transients a short window of an instrumental looks silent

# Copertura (P0.6) [H]: stati per controllo. Un controllo richiesto e "unavailable" impedisce il verdetto "pass"
# (diventa "warn": nessun controllo mancante sembra superato). Richiesti: sempre ritmo, salto di loudness, true
# peak, durata, verso tagliato; con due brani sovrapposti (B beatmatchato) anche doppio basso e scontro di voci.
COVERAGE_STATES = ("measured", "unavailable", "not_applicable", "intended_gap")
REQUIRED_ALWAYS = ("rhythm", "loudness_jump", "true_peak", "duration", "verse_cut", "b_entry")

# Ingresso di B (.6, 21/10/2026) [H]: simmetrico della parola tagliata di A. Una parola di B è troncata se inizia prima del
# punto in cui B diventa udibile e finisce dopo, di almeno B_ENTRY_EPS_S (confini ASR); la prima frase udibile di B è
# iniziata prima (verso troncato) se la parola precedente di B finisce meno di B_REST_BEATS battiti prima della prima
# udibile: nessun respiro (la pausa notata minima di dj_phrase, che con p >= 0,5 fa un confine in dj_phrase.b_lead_in). Un ingresso con fader più lungo di ABRUPT_EXIT_S non tronca.
B_ENTRY_EPS_S = 0.05
B_REST_BEATS = 0.1875  # a noted rest (3/4 of a sixteenth, dj_phrase rest_min_16th): less is no breath at all
B_DEFAULT_BEAT_S = 0.5
REQUIRED_WITH_OVERLAP = ("double_bass", "vocal_clash")

# Fase di B a un ingresso pulito (.7, 22/10/2026). Senza sovrapposizione ritmica (cut, echo/filter out, build, halving,
# break) il .6 misurava la fase solo sul mix con soglia 15 ms per il cut (falsi allarmi sulle continuazioni naturali:
# rumore di Beat This! 21-54 ms) e "warn al massimo" per gli effetti (sfasamenti di 1/2 e 1/4 di battito non rifiutati).
# Ora: rette di beat su deck A (prima dell'uscita) e deck B (dopo il downbeat), confronto allo switch, misura valida solo
# se affidabile (vedi ``_rhythm_entry``). Soglie: ENTRY_PHASE_FAIL_MS = [E] calibrata sul rumore delle giunte naturali
# (tools/calibrate_qa_noise.py, runs/defect_gym_v2/calibration.json: brani e punti disgiunti dai riferimenti della
# palestra), le altre [H].
ENTRY_WINDOW_S = 10.0
ENTRY_MIN_BEATS = 6
ENTRY_PHASE_FAIL_MS = 60.0  # [E] 1.5 x 39.7 ms, max |offset| affidabile su 75 giunte naturali pulite (calibration.json)
ENTRY_CONSENSUS_MS = 25.0
# [H] the onset-flux phase depends on each track's drum pattern (bias up to ~80 ms between two tracks on the gym
# references): it can only confirm on which side of the beat B falls, so the two detectors "agree" within 0.2 beat
# (a Beat This! jump to the off-beats, 0.5 beat, is caught; a tracker error of a few tens of ms is not).
ENTRY_AGREE_BEATS = 0.2
ENTRY_FLUX_MIN_CONCENTRATION = 0.02
ENTRY_TEMPO_TOL = 0.06
# [E] rumore di Beat This! sui deck delle sovrapposizioni naturali (blend di un brano con sé stesso): la soglia della
# mediana dei blend è max(15 ms dell'utente, questo valore). Calibrazione: runs/defect_gym_v2/calibration.json.
OVERLAP_BT_NOISE_MS = 15.0  # xcorr naturale max 6.7 ms: resta la soglia dell'utente; Beat This! deck-vs-deck p95 94 ms: solo warn
# Clic (.7): banda sopra le sorgenti codificate, finestra attorno alle giunte; CLICK_RATIO [E] calibrata (giunte naturali).
CLICK_BAND_HZ = (20500.0, 23500.0)
CLICK_WIN_S = 0.005
CLICK_RATIO = 6.7  # [E] 1.5 x 4.45, massimo sulle 168 giunte naturali (calibration.json)
CLICK_WARN_RATIO = 4.4  # [E] massimo naturale
# Eco fuori tempo (.7) [H]: progressione di almeno 3 copie (z robusto >= 6) nei 3 s dopo l'uscita, tolleranza 8 ms; il passo
# deve stare entro 10 ms da un multiplo di 1/4 o 1/3 di battito (proposta §4.6 della palestra v1).
ECHO_TEMPLATE_S = 0.06
ECHO_WINDOW_S = 3.0
ECHO_MIN_Z = 6.0
ECHO_MIN_REPEATS = 3
ECHO_TOL_S = 0.008
ECHO_MIN_NCC = 0.2  # [H] a copy of A's transient must explain at least this normalized correlation
ECHO_GRID_TOL_MS = 10.0

# Regole AGENTS.md (utente, 20/10/2026): cut 10 s + 10 s; blend/effetti 10 s
# + transizione intera + 10 s. Tolleranza: 2 ms (arrotondamento dei campioni).
CLIP_SIDE_S = 10.0
CLIP_TOLERANCE_S = 0.002


@dataclass(frozen=True)
class QAConfig:
    downbeat_median_max_ms: float = DOWNBEAT_MEDIAN_MAX_MS
    drift_max_ms: float = DRIFT_MAX_MS
    min_pairs: int = MIN_PAIRS
    xcorr_min: float = XCORR_MIN
    bass_hz: float = BASS_HZ
    bass_rel_db: float = BASS_REL_DB
    bass_max_s: float = BASS_MAX_S
    loudness_window_s: float = LOUDNESS_WINDOW_S
    loudness_fail_db: float = LOUDNESS_FAIL_DB
    loudness_warn_db: float = LOUDNESS_WARN_DB
    true_peak_max_dbtp: float = TRUE_PEAK_MAX_DBTP
    vocal_clash_max_s: float = VOCAL_CLASH_MAX_S
    word_guard_s: float = WORD_GUARD_S
    abrupt_exit_s: float = ABRUPT_EXIT_S
    clip_side_s: float = CLIP_SIDE_S
    clip_tolerance_s: float = CLIP_TOLERANCE_S
    # libri (21/10/2026), proposta D4: buco non voluto nella giunta. Spento di default (nessun controllo in più).
    dropout_check: bool = True  # .7: on by default (was off; the set runners already turned it on)
    dropout_rel_db: float = DROPOUT_REL_DB
    dropout_min_beats: float = DROPOUT_MIN_BEATS
    dropout_severity: str = "fail"  # .7: "warn" = behaviour of .4-.6
    dropout_effect_beats: float = DROPOUT_EFFECT_BEATS
    # 21/10/2026 (verifica dei libri su audio reale): dj_render passa anche il deck A grezzo (``a_raw``, prima di EQ e
    # fader) marcato ``a_raw_opt_in``. Spento: il ritmo dei blend di dj_render si misura su A dopo l'EQ, come prima.
    # Acceso: si misura su A grezzo (le opzioni EQ dei libri non spostano più la misura). I deck grezzi passati da
    # dj_creative/dj_effects (senza il marcatore) sono usati come prima in entrambi i casi.
    rhythm_on_raw_a: bool = False
    # P0.6 (21/10/2026): parola tagliata all'uscita con effetto ("fail" come il verso tagliato dei cut; "warn"; "off" =
    # comportamento .4, in cui l'uscita con effetto era "pass" senza misura) e copertura obbligatoria del verdetto.
    effect_word_cut: str = "fail"
    effect_word_eps_s: float = EFFECT_WORD_EPS_S
    require_coverage: bool = True
    # .6 (21/10/2026): ingresso di B ("fail" | "warn" | "off" = non misurato, comportamento .5)
    b_entry_check: str = "fail"
    b_entry_eps_s: float = B_ENTRY_EPS_S
    b_rest_beats: float = B_REST_BEATS
    # .7 (22/10/2026, palestra dei difetti): fase di B a un ingresso pulito (soglie calibrate sul rumore delle giunte
    # naturali, vedi ENTRY_*), soglia del ritmo dei blend con il rumore misurato, controlli nuovi.
    entry_window_s: float = ENTRY_WINDOW_S
    entry_min_beats: int = ENTRY_MIN_BEATS
    entry_phase_fail_ms: float = ENTRY_PHASE_FAIL_MS
    entry_consensus_ms: float = ENTRY_CONSENSUS_MS
    entry_agree_beats: float = ENTRY_AGREE_BEATS
    entry_flux_min_concentration: float = ENTRY_FLUX_MIN_CONCENTRATION
    entry_tempo_tol: float = ENTRY_TEMPO_TOL
    overlap_bt_noise_ms: float = OVERLAP_BT_NOISE_MS
    phrase_check: str = "fail"
    click_check: str = "on"
    click_ratio: float = CLICK_RATIO
    click_severity: str = "fail"
    click_warn_ratio: float = CLICK_WARN_RATIO
    echo_check: str = "on"
    echo_grid_tol_ms: float = ECHO_GRID_TOL_MS
    echo_severity: str = "warn"  # .7 [E]: natural continuations show off-grid progressions (10/196, calibration.json): warn only


THRESHOLD_ORIGIN = {
    "downbeat_median_max_ms": "[H] user threshold; order of magnitude of audible percussive flams, no library source fixes 15 ms",
    "drift_max_ms": "[H] engineering hypothesis",
    "xcorr_min": "[H] same diagnostic threshold as dj_render",
    "bass_rel_db / bass_max_s / bass_hz": "[H] user threshold; 200 Hz = 3-band DJ mixer convention used by the renderer",
    "loudness_window_s": "[S] EBU Tech 3341 short-term loudness (3 s, BS.1770 K-weighting, ungated)",
    "loudness_fail_db / loudness_warn_db": "[H] user thresholds",
    "loudness windows": "[H] cut: 3 s before/after the switch; effects and blends: A before the transition start (no effect) vs B after the transition end "
                        "(effects: after the switch); inside the effect/overlap region the max momentary loudness (400 ms) must not exceed A's before "
                        "(blends: max of A before and B after, at most warn): a dip is expected, a bump is not",
    "true_peak_max_dbtp": "[S] EBU R128 maximum true-peak level, ITU-R BS.1770-4 4x oversampling",
    "vocal_clash_max_s": "[H] user threshold (renderer ducks from 2 s, after MosaikBox)",
    "word_guard_s / abrupt_exit_s": "[H] user threshold / engineering hypothesis",
    "phase ambiguity": "[H] half-beat error within downbeat_median_max_ms + allin1/madmom beat coincidence >= 0.8 with the render grid on both decks -> warn",
    "clip durations": "AGENTS.md (user rule, 20/10/2026)",
    "rhythm_on_raw_a (optional)": "[H] engineering choice (books integration check, 21/10/2026): measure A's beats before the EQ so EQ options "
                                   "(bass_gap_beats, mid_duck_b_bars) do not move the rhythm verdict; off by default",
    "effect_word_cut / effect_word_eps_s": "[H] P0.6 (21/10/2026): with an exit effect the word is cut if A's dry, continuous playback "
                                          "stops inside it (start < stop - 50 ms and end > stop + 50 ms); a tail (echo, repeats) does not complete it; "
                                          "not applicable if A's voice band (1-4 kHz) is already 20 dB down",
    "b_entry / b_entry_eps_s / b_rest_beats": "[H] .6 (21/10/2026, INGRESSO_B): where B becomes audible no word of B may start "
                                                 "before and end after it by > 50 ms (ASR boundaries), and B's first heard word may not "
                                                 "follow B's previous word by less than a noted rest, 3/16 of a beat (no breath: the phrase "
                                                 "began earlier; B&B 2003 cap. 10 "
                                                 "'start halfway through a phrase'); fader entries (blend, overlap) not applicable",
    "coverage": "[H] P0.6: per-check state measured / unavailable / not_applicable / intended_gap; a required check that is unavailable "
                "turns a pass into warn (required: rhythm, loudness_jump, true_peak, duration, verse_cut; + double_bass, vocal_clash "
                "when B is beatmatched over A)",
    "dropout (optional)": "[E citata] Huron 2006 p. 52 (an omitted expected beat is an event); [H] -30 dB vs the 3 s before the transition, "
                          ">= 0.5 beat, inside the transition +- 1 s; intended gaps (gap_beats, break/stop/spin techniques) pass",
}


# ------------------------------------------------------------------ Beat This!


def default_beat_this_checkpoint() -> Path:
    """Checkpoint small0 del progetto (`env/RUNTIME_MAP.md`); env ``MUSICMIX_BEAT_THIS_CKPT``."""
    env = os.environ.get("MUSICMIX_BEAT_THIS_CKPT")
    if env:
        return Path(env)
    candidates = [ROOT / ".cache" / "structure_runtime" / "models" / "beat_this-small0.ckpt"]
    candidates.append(models_dir() / "beat_this-small0.ckpt")
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


class BeatThisTracker:
    """Beat This! (Foscarin et al., ISMIR 2024; pacchetto ufficiale ``beat_this``).

    Post-processing "minimal" ufficiale (picchi dei logit a 50 fps), poi ogni
    evento viene raffinato con interpolazione parabolica del logit attorno al
    frame: risoluzione sotto i 20 ms del frame, stesso modello per entrambi i deck.
    """

    fps = 50

    def __init__(self, checkpoint_path: Optional[os.PathLike | str] = None, device: str = "cpu"):
        from beat_this.inference import Audio2Frames
        from beat_this.model.postprocessor import Postprocessor

        self.checkpoint = Path(checkpoint_path) if checkpoint_path else default_beat_this_checkpoint()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Beat This checkpoint not found: {self.checkpoint}")
        self._frames = Audio2Frames(checkpoint_path=str(self.checkpoint), device=device, float16=False)
        self._post = Postprocessor(type="minimal", fps=self.fps)

    def info(self) -> dict:
        import hashlib

        h = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        return {"model": "Beat This! small0, minimal postprocessing + parabolic logit refinement", "checkpoint": str(self.checkpoint), "checkpoint_sha256": h}

    def __call__(self, audio: np.ndarray, sr: int) -> dict:
        x = _as_stereo(audio).astype(np.float32)
        if len(x) < sr:
            return {"beats": np.zeros(0), "downbeats": np.zeros(0)}
        beat_logits, db_logits = self._frames(x, int(sr))
        beats, downbeats = self._post(beat_logits, db_logits)
        bl = beat_logits.detach().cpu().numpy().astype(float)
        dl = db_logits.detach().cpu().numpy().astype(float)
        return {"beats": _parabolic(np.asarray(beats, float), bl, self.fps), "downbeats": _parabolic(np.asarray(downbeats, float), dl, self.fps)}


def _parabolic(times: np.ndarray, logits: np.ndarray, fps: int) -> np.ndarray:
    out = []
    for t in times:
        i = int(round(t * fps))
        if 0 < i < len(logits) - 1:
            a, b, c = logits[i - 1], logits[i], logits[i + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if den < 0 else 0.0
            out.append((i + float(np.clip(d, -0.5, 0.5))) / fps)
        else:
            out.append(float(t))
    return np.asarray(out, dtype=float)


_TRACKER_CACHE: dict = {}


def _get_tracker(tracker):
    if tracker is None or callable(tracker):
        return tracker
    if tracker == "auto":
        if "auto" not in _TRACKER_CACHE:
            try:
                _TRACKER_CACHE["auto"] = BeatThisTracker()
            except Exception as exc:  # optional runtime
                _TRACKER_CACHE["auto"] = exc
        t = _TRACKER_CACHE["auto"]
        return None if isinstance(t, Exception) else t
    raise ValueError("tracker must be 'auto', None or a callable(audio, sr) -> {'beats','downbeats'}")


# ------------------------------------------------------------------ geometry


def transition_geometry(meta: dict, n_samples: int) -> dict:
    """Inizio, switch, fine della transizione (campioni della clip) dai metadati.

    Legge ``overlap.output_start_sample/output_end_sample`` e
    ``switch.output_sample`` (schema dj_render); in alternativa
    ``transition.{output_start_sample, output_end_sample, switch_sample}``.
    """
    sr = int(meta.get("sample_rate", 48000))
    ov = meta.get("overlap") or meta.get("transition") or {}
    start = ov.get("output_start_sample")
    end = ov.get("output_end_sample")
    sw = (meta.get("switch") or {}).get("output_sample", ov.get("switch_sample"))
    if start is None or end is None:
        raise ValueError("metadata lacks overlap/transition output_start_sample/output_end_sample")
    if sw is None:
        sw = (int(start) + int(end)) // 2
    kind = str(meta.get("kind", "unknown"))
    effect = _exit_effect(meta)
    # dj_effects.drop_in is a dj_render cut (``base_kind: "cut"``): before the
    # 20/10/2026 correction it was treated as a blend (wrong clip rule, no cut checks).
    is_cut = kind == "cut" or meta.get("base_kind") == "cut" or (
        effect is None and int(end) - int(start) <= int(0.05 * sr) and "cut" in kind)
    return {"kind": kind, "is_cut": bool(is_cut), "sr": sr, "start": int(start), "switch": int(sw), "end": int(end), "n": int(n_samples), "exit_effect": effect}


def _exit_effect(meta: dict) -> Optional[str]:
    blobs = []
    for key in ("effect", "exit_effect", "effects", "fx", "kind"):
        if key in meta:
            blobs.append(json.dumps(meta[key]).lower() if not isinstance(meta[key], str) else meta[key].lower())
    for word in EXEMPT_EXIT_EFFECTS:
        if any(word in b for b in blobs):
            return word
    return None


# ------------------------------------------------------------------ helpers


def _as_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, :2]


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = _as_stereo(x)
    if len(x) >= n:
        return x[:n]
    return np.vstack([x, np.zeros((n - len(x), 2))])


def _db(v: float) -> Optional[float]:
    return 20.0 * math.log10(v) if v > 0 else None


def _short_term_lufs(x: np.ndarray, sr: int) -> Optional[float]:
    import pyloudnorm

    x = _as_stereo(x)
    if len(x) < int(0.4 * sr):
        return None
    meter = pyloudnorm.Meter(sr, block_size=len(x) / sr)  # one block = ungated short-term loudness
    v = float(meter.integrated_loudness(x))
    return v if math.isfinite(v) else None


def _flux(x: np.ndarray, sr: int, hop: int, n_fft: int = 1024) -> np.ndarray:
    from scipy.signal import stft

    mono = _as_stereo(x).mean(axis=1)
    if len(mono) < 2 * n_fft:
        return np.zeros(max(1, len(mono) // hop))
    _, _, spec = stft(mono, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    mag = np.log1p(100.0 * np.abs(spec))
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    return np.r_[np.zeros(n_fft // (2 * hop) + 1), flux]  # roughly centre frames on their time


def _frame_rms(x: np.ndarray, hop: int, win: int) -> np.ndarray:
    mono2 = np.asarray(x, dtype=float) ** 2
    if mono2.ndim == 2:
        mono2 = mono2.mean(axis=1)
    c = np.r_[0.0, np.cumsum(mono2)]
    starts = np.arange(0, max(1, len(mono2) - win + 1), hop)
    return np.sqrt(np.maximum(0.0, (c[np.minimum(starts + win, len(mono2))] - c[starts]) / win))


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def _check(status: str, **kw) -> dict:
    return {"status": status, **kw}


# ----------------------------------------------------------------- the checks


def check_duration(n: int, geo: dict, meta: dict, cfg: QAConfig) -> dict:
    sr = geo["sr"]
    tol = cfg.clip_tolerance_s
    dur = n / sr
    problems = []
    if geo["is_cut"]:
        rule = "cut: 10 s + 10 s = 20 s around the switch"
        expected = 2 * cfg.clip_side_s
        if abs(geo["switch"] / sr - cfg.clip_side_s) > tol:
            problems.append(f"switch at {geo['switch']/sr:.3f} s, expected {cfg.clip_side_s:.0f} s")
    else:
        rule = "blend/effect: 10 s before transition start + whole transition + 10 s after"
        expected = geo["end"] / sr + cfg.clip_side_s
        if abs(geo["start"] / sr - cfg.clip_side_s) > tol:
            problems.append(f"transition starts at {geo['start']/sr:.3f} s, expected {cfg.clip_side_s:.0f} s")
    if abs(dur - expected) > tol:
        problems.append(f"clip lasts {dur:.3f} s, expected {expected:.3f} s")
    declared = (meta.get("clip") or {}).get("length_samples")
    if declared is not None and int(declared) != n:
        problems.append(f"metadata clip length {declared} samples != PCM {n}")
    return _check("fail" if problems else "pass", rule=rule, duration_s=dur, expected_s=expected, problems=problems, origin=THRESHOLD_ORIGIN["clip durations"])


def check_true_peak(audio: np.ndarray, sr: int, cfg: QAConfig) -> dict:
    from musicmix.dj_render import true_peak_dbtp

    x = _as_stereo(audio)
    tp = true_peak_dbtp(x, sr)
    full = int(np.sum(np.abs(x) >= FULL_SCALE))
    status = "fail" if (tp is not None and tp > cfg.true_peak_max_dbtp) or full > 0 else "pass"
    return _check(status, true_peak_dbtp=tp if math.isfinite(tp) else None, sample_peak_dbfs=_db(float(np.max(np.abs(x))) if len(x) else 0.0),
                  full_scale_samples=full, threshold_dbtp=cfg.true_peak_max_dbtp, method="4x oversampled true peak (dj_render.true_peak_dbtp, BS.1770-4 style)")


def _beat_s(meta: dict) -> Optional[float]:
    st = meta.get("stretch") or {}
    bar = st.get("a_bar_s")
    return float(bar) / 4.0 if bar else None


def intended_gap(meta: dict) -> Optional[str]:
    """Why a silence in the seam is intended (creative technique or ``gap_beats`` > 0), else None."""
    cre = meta.get("creative") or {}
    tech = cre.get("technique") or meta.get("kind")
    params = cre.get("params") or {}
    if tech in INTENDED_GAP_TECHNIQUES:
        return f"technique {tech}"
    for k in ("gap_beats", "gap_bars"):
        if float(params.get(k) or 0) > 0:
            return f"{k} = {params[k]}"
    return None


def declared_gap_s(meta: dict, beat_s: float) -> Optional[float]:
    """Length of the silence the render declares (``gap_beats``/``gap_bars``), 0 if none; None = intended but unbounded
    (stop, tape stop, backspin: the silence is the technique)."""
    cre = meta.get("creative") or {}
    tech = cre.get("technique") or meta.get("kind")
    params = cre.get("params") or {}
    if tech in INTENDED_GAP_TECHNIQUES and not any(float(params.get(k) or 0) > 0 for k in ("gap_beats", "gap_bars")):
        return None
    return float(params.get("gap_beats") or 0) * beat_s + float(params.get("gap_bars") or 0) * 4.0 * beat_s


def check_dropout(audio: np.ndarray, geo: dict, meta: dict, cfg: QAConfig) -> dict:
    """[E citata] Huron 2006 p. 52: un silenzio dove era atteso un battito è un evento. Cerca nella transizione (± 1 s)
    il tratto più lungo con livello (RMS 50 ms, passo 10 ms) sotto ``dropout_rel_db`` rispetto ai 3 s prima. Dal .7
    (palestra dei difetti: un buco di 1 battito dava solo ``warn``) è ``fail`` (``dropout_severity``) se supera la pausa
    dichiarata dal render (``gap_beats``/``gap_bars``, 0 se nessuna) di almeno ``dropout_min_beats`` battiti; soglia
    calibrata sulle giunte naturali (``DROPOUT_MIN_BEATS``). Le tecniche la cui pausa è la tecnica stessa (stop, tape
    stop, backspin) restano ``intended_gap``. Finestre (.7): cut = ±1 s dallo switch; B sotto A = sovrapposizione ±1 s;
    uscita a effetto / tecnica con B pulito = da 0,1 s prima dell'ingresso di B a 1 s dopo la fine (soglia
    ``dropout_min_beats``) e, separatamente, la regione dell'effetto prima dell'ingresso (soglia ``dropout_effect_beats``,
    dove le eco e i filtri naturali scendono da soli sotto -30 dB; lì si sconta la pausa dichiarata)."""
    sr = geo["sr"]
    mono = audio.mean(axis=1)
    ref = mono[max(0, geo["start"] - 3 * sr) : geo["start"]]
    ref_rms = float(np.sqrt(np.mean(ref ** 2))) if len(ref) else 0.0
    beat = _beat_s(meta) or 0.5
    hop, win = int(0.01 * sr), int(0.05 * sr)
    thr = ref_rms * 10.0 ** (cfg.dropout_rel_db / 20.0)

    def longest_quiet(a0: int, a1: int) -> tuple[float, Optional[float]]:
        seg = mono[max(0, a0):max(0, min(len(mono), a1))]
        nf = max(0, (len(seg) - win) // hop + 1)
        if nf == 0:
            return 0.0, None
        c = np.r_[0.0, np.cumsum(seg ** 2)]
        st = np.arange(nf) * hop
        quiet = np.sqrt(np.maximum(0.0, (c[st + win] - c[st]) / win)) < thr
        run = best = best_end = 0
        for i, q in enumerate(quiet):
            run = run + 1 if q else 0
            if run > best:
                best, best_end = run, i
        return best * hop / sr, ((max(0, a0) + (best_end - best + 1) * hop) / sr if best else None)

    if ref_rms <= 0:
        return _check("skipped", reason="silent reference", longest_s=0.0, coverage="unavailable")
    effect_region = None
    if geo["is_cut"]:
        s0, s1, where = geo["switch"] - sr, geo["switch"] + sr, "switch +- 1 s"
    elif b_overlaps_a(meta, geo):
        s0, s1, where = geo["start"] - sr, geo["end"] + sr, "overlap +- 1 s"
    else:  # exit effect / technique, then B clean: inside the effect a dip is part of it (echo repeats, filter, build)
        pre = int(DROPOUT_PRE_S * sr)
        s0, s1, where = geo["switch"] - pre, geo["end"] + sr, f"from {DROPOUT_PRE_S:g} s before B's entry"
        effect_region = (geo["start"] - sr, geo["switch"] - pre)
    longest, at_s = longest_quiet(s0, s1)
    eff_longest, eff_at = longest_quiet(*effect_region) if effect_region else (0.0, None)
    why = intended_gap(meta)
    dec = declared_gap_s(meta, beat)
    excess = longest - (0.0 if effect_region else (dec or 0.0))
    eff_excess = eff_longest - (dec or 0.0)
    bad = excess >= cfg.dropout_min_beats * beat
    bad_eff = effect_region is not None and eff_excess >= cfg.dropout_effect_beats * beat
    info = dict(longest_s=longest, longest_beats=longest / beat, at_s=at_s, beat_s=beat, intended=why, declared_gap_s=dec,
                excess_beats=excess / beat, window=where, threshold_beats=cfg.dropout_min_beats,
                effect_region_longest_s=eff_longest if effect_region else None,
                effect_region_longest_beats=(eff_longest / beat) if effect_region else None,
                effect_region_excess_beats=(eff_excess / beat) if effect_region else None, effect_threshold_beats=cfg.dropout_effect_beats)
    if (bad or bad_eff) and dec is None:
        return _check("pass", coverage="intended_gap", reason=None, **info)
    if not (bad or bad_eff):
        return _check("pass", coverage="measured", reason=None, **info)
    if bad:
        msg = (f"silence of {longest:.2f} s ({longest / beat:.2f} beat) in the seam at {at_s:.2f} s"
               + (f", {excess / beat:.2f} beat longer than the declared pause ({why})" if (dec and not effect_region) else ", not declared by the render"))
    else:
        msg = (f"silence of {eff_longest:.2f} s ({eff_longest / beat:.2f} beat) inside the exit effect at {eff_at:.2f} s"
               + (f", {eff_excess / beat:.2f} beat longer than the declared pause ({why})" if dec else ", not declared by the render"))
    return _check(cfg.dropout_severity, coverage="measured", reason=msg + " (Huron p. 52: a missing beat is an event)", **info)


def _momentary_lufs(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Loudness momentary (400 ms, passo 100 ms) dell'intera clip: (centri in campioni, LUFS).

    K-weighting di pyloudnorm applicato una volta sola alla clip intera (niente
    transitori del filtro a ogni finestra), poi media dei quadrati per blocco.
    """
    import pyloudnorm

    x = _as_stereo(audio).copy()
    meter = pyloudnorm.Meter(sr)
    for stage in meter._filters.values():
        for ch in range(x.shape[1]):
            x[:, ch] = stage.apply_filter(x[:, ch])
    win, hop = int(round(LOUDNESS_MOMENTARY_S * sr)), int(round(LOUDNESS_MOMENTARY_HOP_S * sr))
    if len(x) < win:
        return np.zeros(0, int), np.zeros(0)
    c = np.r_[0.0, np.cumsum((x ** 2).sum(axis=1))]
    starts = np.arange(0, len(x) - win + 1, hop)
    ms = (c[starts + win] - c[starts]) / win
    with np.errstate(divide="ignore"):
        lufs = -0.691 + 10.0 * np.log10(ms)
    return starts + win // 2, lufs


def _momentary_max(centres: np.ndarray, lufs: np.ndarray, s0: int, s1: int, half: int) -> Optional[float]:
    """Massimo sui blocchi interamente contenuti in ``[s0, s1)`` (``half`` = mezzo blocco in campioni)."""
    sel = lufs[(centres - half >= s0) & (centres + half <= s1) & np.isfinite(lufs)]
    return float(sel.max()) if len(sel) else None


def check_loudness_jump(audio: np.ndarray, geo: dict, cfg: QAConfig) -> dict:
    """Salto di loudness (vedi le costanti ``LOUDNESS_*`` per la scelta delle finestre).

    cut: 3 s prima vs 3 s dopo lo switch. Effetti e blend: A prima dell'inizio
    della transizione vs B dopo la fine (effetti: dopo lo switch) e controllo
    che nella regione dell'effetto/sovrapposizione non ci siano picchi sopra A
    prima; la finestra attorno allo switch resta solo informativa.
    """
    sr, n = geo["sr"], len(audio)
    w = int(round(cfg.loudness_window_s * sr))
    sw, s, e = geo["switch"], geo["start"], geo["end"]
    has_region = not geo["is_cut"] and e - s > int(0.1 * sr)
    is_effect = bool(geo.get("exit_effect")) and has_region
    pairs: dict = {"switch": (sw - w, sw, sw, sw + w)}
    used = {"switch"}
    region = None
    if has_region:
        b_from = sw if is_effect and sw > s else e
        pairs["a_before_vs_b_after"] = (s - w, s, b_from, b_from + w)
        used = {"a_before_vs_b_after"}
        region = (s, sw if is_effect and sw > s else e)
    rows, worst = {}, None
    for name, (b0, b1, a0, a1) in pairs.items():
        if b0 < 0 or a1 > n:
            rows[name] = {"status": "skipped", "reason": "window outside the clip", "used_for_verdict": name in used}
            continue
        lb, la = _short_term_lufs(audio[b0:b1], sr), _short_term_lufs(audio[a0:a1], sr)
        jump = (la - lb) if lb is not None and la is not None else None
        rows[name] = {"lufs_before": lb, "lufs_after": la, "jump_db": jump, "before_s": [b0 / sr, b1 / sr], "after_s": [a0 / sr, a1 / sr],
                      "used_for_verdict": name in used}
        if name not in used:
            rows[name]["note"] = "informative only: around the switch the windows fall inside the effect / fade by construction"
        if name in used and jump is not None and (worst is None or abs(jump) > abs(worst[1])):
            worst = (name, jump)
    peak = None
    if region is not None and s - w >= 0:
        centres, mom = _momentary_lufs(audio, sr)
        half = int(round(LOUDNESS_MOMENTARY_S * sr)) // 2
        ref_a = _momentary_max(centres, mom, s - w, s, half)
        b_from = pairs["a_before_vs_b_after"][2]
        ref_b = None if is_effect else _momentary_max(centres, mom, b_from, min(n, b_from + w), half)
        refs = [v for v in (ref_a, ref_b) if v is not None]
        inside = _momentary_max(centres, mom, region[0], region[1], half)
        if refs and inside is not None:
            ref = max(refs)
            excess = inside - ref
            st = "fail" if excess > cfg.loudness_fail_db else "warn" if excess > cfg.loudness_warn_db else "pass"
            if not is_effect and st == "fail":
                st = "warn"  # blends: at most warn (summed decks; double bass has its own check)
            peak = {"status": st, "region_s": [region[0] / sr, region[1] / sr], "region_max_momentary_lufs": inside,
                    "a_before_max_momentary_lufs": ref_a, "b_after_max_momentary_lufs": ref_b, "reference_lufs": ref,
                    "excess_db": excess, "reference": "A before" if is_effect else "max(A before, B after)",
                    "method": "max momentary loudness (400 ms, hop 100 ms, BS.1770 K-weighting) inside the region vs the reference; dips are allowed"}
        rows["region_peak"] = peak or {"status": "skipped", "reason": "region or reference not measurable"}
    if worst is None and peak is None:
        return _check("skipped", reason="no measurable window", windows=rows)
    order = {"pass": 0, "warn": 1, "fail": 2}
    jump_status, reasons = "pass", []
    mag = abs(worst[1]) if worst else None
    if worst is not None:
        jump_status = "fail" if mag > cfg.loudness_fail_db else "warn" if mag > cfg.loudness_warn_db else "pass"
        if jump_status != "pass":
            reasons.append(f"short-term loudness jump {worst[1]:+.1f} dB ({worst[0]})")
    status = jump_status
    if peak is not None and peak.get("status") in ("warn", "fail"):
        reasons.append(f"level bump inside the {'effect' if is_effect else 'overlap'}: momentary max {peak['excess_db']:+.1f} dB above {peak['reference']}")
        status = max(status, peak["status"], key=lambda k: order[k])
    return _check(status, max_abs_jump_db=mag, worst_window=worst[0] if worst else None, signed_jump_db=worst[1] if worst else None,
                  region_peak=peak, windows=rows, reason="; ".join(reasons) if reasons else None,
                  scope="cut: around the switch" if not has_region else ("effect: A before the effect vs B after the switch + no bump inside the effect"
                                                                         if is_effect else "blend: A before vs B after the transition + no bump inside the overlap"),
                  thresholds_db={"fail": cfg.loudness_fail_db, "warn": cfg.loudness_warn_db},
                  method="short-term loudness (3 s, BS.1770 K-weighting, ungated, pyloudnorm) of the rendered clip; momentary 400 ms for the region peak")


def check_double_bass(decks: Optional[dict], geo: dict, cfg: QAConfig) -> dict:
    if not decks or decks.get("a") is None or decks.get("b") is None:
        return _check("skipped", reason="per-deck audio (post EQ/fader) not provided")
    from scipy.signal import butter, sosfiltfilt

    sr, n = geo["sr"], geo["n"]
    sos = butter(4, cfg.bass_hz, btype="low", fs=sr, output="sos")
    hop, win = int(0.010 * sr), int(BASS_RMS_WINDOW_S * sr)
    env = []
    for key in ("a", "b"):
        low = sosfiltfilt(sos, _fit(decks[key], n).mean(axis=1))
        env.append(_frame_rms(low, hop, win))
    m = min(len(env[0]), len(env[1]))
    ea, eb = env[0][:m], env[1][:m]
    pa, pb = float(ea.max()), float(eb.max())
    if pa <= 0 or pb <= 0:
        return _check("pass", reason="one deck has no energy below the cutoff", longest_s=0.0)
    g = 10.0 ** (cfg.bass_rel_db / 20.0)
    both = (ea >= g * pa) & (eb >= g * pb)
    longest = _longest_run(both) * hop / sr
    first = int(np.argmax(both)) * hop / sr if both.any() else None
    status = "fail" if longest > cfg.bass_max_s else "pass"
    return _check(status, longest_s=longest, total_s=float(both.sum() * hop / sr), first_at_s=first,
                  threshold={"rel_db": cfg.bass_rel_db, "max_s": cfg.bass_max_s, "below_hz": cfg.bass_hz},
                  method="Butterworth-4 low-pass (zero phase) per deck, RMS 150 ms / hop 10 ms, each deck vs its own peak in the clip")


def grid_phase_support(tempo: Optional[dict]) -> dict:
    """allin1/madmom concordano con la griglia usata per un deck? (blocco ``tempo`` di dj_select).

    Serve almeno un provider fra allin1 e madmom, diverso da quello della
    griglia, nella stessa ottava e con coincidenza dei beat >= ``PHASE_AGREE_MIN``.
    """
    if not tempo:
        return {"supported": False, "reason": "no grid provenance"}
    used = tempo.get("provider")
    agree = tempo.get("agreement") or {}
    agreeing = []
    for name in PHASE_AMBIGUITY_PROVIDERS:
        a = agree.get(name) or {}
        if name != used and a.get("same_tempo_octave") and float(a.get("beat_coincidence", 0.0)) >= PHASE_AGREE_MIN:
            agreeing.append(name)
    bt = agree.get("beat_this") or {}
    return {"supported": bool(agreeing), "grid_provider": used, "agreeing_providers": agreeing,
            "beat_this_beat_coincidence": 1.0 if used == "beat_this" else bt.get("beat_coincidence"),
            # .7: at another tempo octave a perfectly consistent Beat This! coincides with half the beats (0.5): off grid only below 0.25
            "beat_this_off_grid": bool(used != "beat_this" and bt and float(bt.get("beat_coincidence", 1.0))
                                       < (0.5 if bt.get("same_tempo_octave", True) else 0.25))}


def _phase_ambiguity(err_ms: Optional[float], period_s: Optional[float], support: Optional[dict], cfg: "QAConfig",
                     tol_ms: Optional[float] = None) -> Optional[dict]:
    """Info se ``|err|`` è a mezzo battito (+-soglia), le griglie di entrambi i deck sono confermate da allin1/madmom
    e (dal .7) Beat This! è già fuori dalla griglia nell'analisi di almeno uno dei due brani: solo allora il tracker,
    non il render, è sospetto (MAFIA, galleria v1). Se Beat This! concorda con le griglie nelle analisi, un errore a
    mezzo battito sulla clip è un errore vero (es. griglia sbagliata localmente: palestra dei difetti v1, §4.1)."""
    if err_ms is None or not period_s or not support:
        return None
    half = 500.0 * float(period_s)
    resid = abs(abs(float(err_ms)) - half)
    if resid > (cfg.downbeat_median_max_ms if tol_ms is None else tol_ms):
        return None
    a, b = support.get("a") or {}, support.get("b") or {}
    if not (a.get("supported") and b.get("supported")):
        return None
    suspects = [k for k, v in (("A", a), ("B", b)) if v.get("beat_this_off_grid")]
    if not suspects:
        return None
    return {"error_ms": float(err_ms), "half_beat_ms": half, "residual_from_half_beat_ms": resid,
            "agreeing": {"a": a.get("agreeing_providers"), "b": b.get("agreeing_providers")},
            "beat_this_off_grid_in_analysis": suspects,
            "message": (f"possibile ambiguità di fase del tracker: errore {float(err_ms):+.1f} ms ≈ mezzo battito ({half:.0f} ms); "
                        f"griglie del render confermate da A: {'/'.join(a.get('agreeing_providers') or [])}, B: {'/'.join(b.get('agreeing_providers') or [])}"
                        + (f"; Beat This! già fuori fase nell'analisi di {', '.join(suspects)}" if suspects else ""))}


def check_rhythm(audio: np.ndarray, geo: dict, decks: Optional[dict], tracker, cfg: QAConfig, phase_support: Optional[dict] = None,
                 meta: Optional[dict] = None) -> dict:
    sr = geo["sr"]
    out: dict = {"methods": {}}
    decided = []
    meta = meta or {}
    has_overlap = b_overlaps_a(meta, geo) and geo["end"] - geo["start"] > int(0.5 * sr)
    out["rhythmic_overlap"] = bool(has_overlap)
    if not has_overlap and tracker is not None:  # .7: clean entry, B's phase against A's pulse at the switch
        en = _rhythm_entry(audio, geo, decks, tracker, cfg, phase_support, meta)
        out["methods"]["beat_this_entry"] = en
        if en["status"] != "unreliable":
            decided.append(en)
        if not decided:
            return _check("warn", reason=en.get("reason") or "rhythmic alignment not measurable with confidence", coverage="unavailable", **out)
    if decks and has_overlap:
        a_raw = decks.get("a_raw")
        if a_raw is not None and decks.get("a_raw_opt_in") and not cfg.rhythm_on_raw_a:
            a_raw = None  # default: dj_render's raw A is ignored (legacy measurement on the EQ'd deck)
        a = a_raw if a_raw is not None else decks.get("a")
        if cfg.rhythm_on_raw_a:
            out["a_deck_measured"] = "a_raw" if a_raw is not None else "a"
        b = decks.get("b_stretched_raw", decks.get("b_raw", decks.get("b")))
        if a is not None and b is not None:
            a, b = _fit(a, geo["n"]), _fit(b, geo["n"])
            period = None
            if tracker is not None:
                bt = _rhythm_beat_this_decks(a, b, geo, tracker, cfg, phase_support)
                out["methods"]["beat_this_decks"] = bt
                period = bt.get("beat_period_s")
                if bt["status"] != "unreliable":
                    decided.append(bt)
            if bt_cap := out["methods"].get("beat_this_decks"):
                # .7: on natural blends (same track, two sections) the deck-vs-deck pairing of Beat This! errs by up to
                # 250 ms (median |downbeat error| p95 156 ms, calibration.json): diagnostic, at most a warn.
                if bt_cap.get("status") == "fail" and bt_cap in decided:
                    decided.remove(bt_cap)
                    decided.append(dict(bt_cap, status="warn", reason=(bt_cap.get("reason") or "") + " (Beat This! deck pairing: warn at most, .7)"))
            xc = _rhythm_xcorr(a, b, geo, period, cfg)
            out["methods"]["onset_xcorr_decks"] = xc
            if xc["status"] != "unreliable":
                decided.append(xc)
            if tracker is not None:  # .7: B's phase against A's pulse at the switch, raw decks, same reliability rules
                ol = _rhythm_entry(audio, geo, {"a": a, "b": b}, tracker, cfg, phase_support, meta, overlap=True)
                out["methods"]["beat_this_overlap_lines"] = ol
                if ol["status"] != "unreliable":
                    decided.append(ol)
    if not decided and tracker is not None:
        mo = _rhythm_mix_only(audio, geo, tracker, cfg, phase_support)
        out["methods"]["beat_this_mix"] = mo
        if mo["status"] != "unreliable":
            decided.append(mo)
    if not out["methods"]:
        return _check("skipped", reason="no tracker and no decks", methods={})
    if not decided:
        return _check("warn", reason="rhythmic alignment not measurable with confidence", coverage="unavailable", **out)
    order = {"pass": 0, "warn": 1, "fail": 2}
    worst = max(decided, key=lambda r: order[r["status"]])
    reasons = [r["reason"] for r in decided if r["status"] != "pass" and r.get("reason")]
    return _check(worst["status"], reason="; ".join(reasons) if reasons else None, **out)


def _pair_errors(ref: np.ndarray, other: np.ndarray, max_abs: float) -> tuple[np.ndarray, np.ndarray]:
    if len(ref) == 0 or len(other) == 0:
        return np.zeros(0), np.zeros(0)
    j = np.searchsorted(other, ref)
    cand = np.stack([other[np.clip(j - 1, 0, len(other) - 1)], other[np.clip(j, 0, len(other) - 1)]], 1)
    nearest = cand[np.arange(len(ref)), np.argmin(np.abs(cand - ref[:, None]), axis=1)]
    err = nearest - ref
    keep = np.abs(err) <= max_abs
    return ref[keep], err[keep]


def _drift(t: np.ndarray, err_ms: np.ndarray, span_s: float) -> Optional[dict]:
    """Deriva = pendenza Theil-Sen (robusta a eventi isolati sbagliati) x durata, con IC 95%."""
    if len(t) < 4:
        return None
    from scipy.stats import theilslopes

    slope, _, lo, hi = theilslopes(np.asarray(err_ms, float), np.asarray(t, float), alpha=0.95)
    return {"ms": float(slope * span_s), "ci95_ms": [float(lo * span_s), float(hi * span_s)], "significant": bool(lo > 0 or hi < 0)}


def _rate_status(median_ms: Optional[float], drift: Optional[dict], cfg: QAConfig, label: str,
                 floor_ms: Optional[float] = None) -> tuple[str, Optional[str]]:
    """fail: mediana > soglia, o deriva > soglia con IC 95% che esclude 0; warn: deriva > soglia non significativa.

    ``floor_ms`` (.7): rumore di misura del metodo sulle giunte naturali; la soglia della mediana è il maggiore tra la
    soglia dell'utente e questo rumore (sotto il rumore la misura non distingue un errore da un render corretto)."""
    fails, warns = [], []
    thr = max(cfg.downbeat_median_max_ms, floor_ms or 0.0)
    if median_ms is not None and median_ms > thr:
        fails.append(f"{label}: median |error| {median_ms:.1f} ms > {thr:.0f} ms")
    if drift is not None and abs(drift["ms"]) > cfg.drift_max_ms:
        msg = f"{label}: drift {drift['ms']:+.1f} ms across the overlap (95% CI {drift['ci95_ms'][0]:+.0f}..{drift['ci95_ms'][1]:+.0f}) > {cfg.drift_max_ms:.0f} ms"
        (fails if drift["significant"] else warns).append(msg if drift["significant"] else msg + ", not significant")
    if fails:
        return "fail", "; ".join(fails + warns)
    return ("warn", "; ".join(warns)) if warns else ("pass", None)


def _rhythm_beat_this_decks(a, b, geo, tracker, cfg, phase_support=None) -> dict:
    sr = geo["sr"]
    t0, t1 = geo["start"] / sr, geo["end"] / sr
    ra, rb = tracker(a, sr), tracker(b, sr)
    ba, bb = np.sort(np.asarray(ra["beats"], float)), np.sort(np.asarray(rb["beats"], float))
    da, dbb = np.sort(np.asarray(ra["downbeats"], float)), np.sort(np.asarray(rb["downbeats"], float))
    sel = ba[(ba >= t0 - 1.0) & (ba <= t1 + 1.0)]
    period = float(np.median(np.diff(sel))) if len(sel) >= 3 else (float(np.median(np.diff(ba))) if len(ba) >= 3 else 0.5)
    in_a = da[(da >= t0 - 1e-3) & (da <= t1 + 1e-3)]
    ref_d, err_d = _pair_errors(in_a, dbb, 0.5 * period)
    in_ab = ba[(ba >= t0 - 1e-3) & (ba <= t1 + 1e-3)]
    ref_b, err_b = _pair_errors(in_ab, bb, 0.25 * period)  # beats: a quarter beat, so off-beat B events are not paired
    err_d_ms, err_b_ms = 1000 * err_d, 1000 * err_b
    res = {
        "a_downbeats_in_overlap": int(len(in_a)),
        "paired_downbeats": int(len(err_d)),
        "unpaired_downbeats": int(len(in_a) - len(err_d)),
        "downbeat_errors_ms": err_d_ms.round(2).tolist(),
        "downbeat_median_abs_ms": float(np.median(np.abs(err_d_ms))) if len(err_d) else None,
        "downbeat_drift": _drift(ref_d, err_d_ms, t1 - t0),
        "paired_beats": int(len(err_b)),
        "beat_errors_ms": err_b_ms.round(2).tolist(),
        "beat_times_s": ref_b.round(3).tolist(),
        "beat_median_abs_ms": float(np.median(np.abs(err_b_ms))) if len(err_b) else None,
        "beat_drift": _drift(ref_b, err_b_ms, t1 - t0),
        "beat_period_s": period,
        "method": "Beat This! on deck A and on (unfaded) deck B separately; B event nearest to each A event in the overlap (within half a beat); error = B - A",
    }
    if len(err_d) < cfg.min_pairs and len(err_b) < cfg.min_pairs:
        res.update(status="unreliable", reason="Beat This!: too few paired downbeats/beats in the overlap")
        return res
    statuses = []
    if len(err_d) >= cfg.min_pairs:
        st_d = _rate_status(res["downbeat_median_abs_ms"], res["downbeat_drift"], cfg, "Beat This! downbeats", cfg.overlap_bt_noise_ms)
        amb = _phase_ambiguity(res["downbeat_median_abs_ms"], period, phase_support, cfg) if st_d[0] == "fail" else None
        if amb is not None:
            res["phase_ambiguity"] = amb
            st_d = ("warn", "Beat This! downbeats: " + amb["message"])
        statuses.append(st_d)
    else:
        statuses.append(("warn", f"Beat This!: only {len(err_d)} paired downbeats (B downbeats far from A's: possible bar-phase error or tracker miss)"))
    if len(err_b) >= cfg.min_pairs:
        statuses.append(_rate_status(res["beat_median_abs_ms"], res["beat_drift"], cfg, "Beat This! beats", cfg.overlap_bt_noise_ms))
    order = {"pass": 0, "warn": 1, "fail": 2}
    st = max(statuses, key=lambda s: order[s[0]])[0]
    msgs = [m for s, m in statuses if m]
    res.update(status=st, reason="; ".join(msgs) if msgs else None)
    return res


def _rhythm_xcorr(a, b, geo, period, cfg) -> dict:
    sr = geo["sr"]
    hop = max(1, sr // 1000)  # 1 ms
    ea, eb = _flux(a, sr, hop), _flux(b, sr, hop)
    m = min(len(ea), len(eb))
    ea, eb = ea[:m], eb[:m]
    max_lag = int(round(min(80.0, 250.0 * (period or 0.5))))  # +-quarter beat, at most 80 ms
    win, step = 2000, 1000
    t0, t1 = geo["start"] // hop, geo["end"] // hop
    rows = []
    for s in range(t0 + max_lag, t1 - win - max_lag + 1, step):
        ra = ea[s : s + win] - ea[s : s + win].mean()
        na = np.linalg.norm(ra)
        corr = np.zeros(2 * max_lag + 1)
        for i, lag in enumerate(range(-max_lag, max_lag + 1)):
            rb = eb[s + lag : s + lag + win]
            rb = rb - rb.mean()
            d = na * np.linalg.norm(rb)
            corr[i] = float(ra @ rb / d) if d > 0 else 0.0
        k = int(np.argmax(corr))
        frac = 0.0
        if 0 < k < len(corr) - 1:
            den = corr[k - 1] - 2 * corr[k] + corr[k + 1]
            frac = 0.5 * (corr[k - 1] - corr[k + 1]) / den if den < 0 else 0.0
        rows.append({"t_s": (s + win / 2) * hop / sr, "lag_ms": (k - max_lag + frac) * hop * 1000.0 / sr, "xcorr": float(corr[k])})
    conf = [r for r in rows if r["xcorr"] >= cfg.xcorr_min]
    res = {"windows": len(rows), "confident_windows": len(conf), "per_window": rows,
           "method": "spectral-flux envelopes (1 ms) of deck A vs deck B, 2 s windows, lag of B relative to A (+ = B late), max lag +-min(80 ms, beat/4)"}
    if len(conf) < max(cfg.min_pairs, (len(rows) + 1) // 2):
        res.update(status="unreliable", reason=f"onset xcorr: {len(conf)}/{len(rows)} windows with xcorr >= {cfg.xcorr_min} (different rhythmic content)")
        return res
    lags = np.array([r["lag_ms"] for r in conf])
    res["median_abs_lag_ms"] = float(np.median(np.abs(lags)))
    res["median_lag_ms"] = float(np.median(lags))
    res["drift"] = _drift(np.array([r["t_s"] for r in conf]), lags, (geo["end"] - geo["start"]) / sr)
    st, msg = _rate_status(res["median_abs_lag_ms"], res["drift"], cfg, "onset xcorr")
    res.update(status=st, reason=msg)
    return res


def _fit_beat_line(beats: np.ndarray) -> Optional[dict]:
    """Retta t = c + k*i con indici interi di battito (tollera battiti mancanti/spuri)."""
    beats = np.asarray(beats, float)
    if len(beats) < 4:
        return None
    k = float(np.median(np.diff(beats)))
    keep = np.ones(len(beats), bool)
    for _ in range(4):
        idx = np.round((beats - beats[0]) / k)
        if keep.sum() < 4 or len(np.unique(idx[keep])) < 4:
            return None
        k, c = np.polyfit(idx[keep], beats[keep], 1)
        resid = beats - (k * idx + c)
        keep = np.abs(resid) <= 0.12 * k
    if keep.sum() < 4:
        return None
    return {"period_s": float(k), "intercept_s": float(c), "n_used": int(keep.sum()), "n_rejected": int((~keep).sum()),
            "rms_resid_ms": float(1000 * np.sqrt(np.mean(resid[keep] ** 2)))}


def _rhythm_mix_only(audio, geo, tracker, cfg, phase_support=None) -> dict:
    """Beat This! sulla clip: rette di beat in A-sola e B-sola estrapolate allo switch."""
    sr = geo["sr"]
    r = tracker(_as_stereo(audio), sr)
    beats = np.sort(np.asarray(r["beats"], float))
    downs = np.sort(np.asarray(r["downbeats"], float))
    a_end = (geo["switch"] if geo["is_cut"] else geo["start"]) / sr
    b_start = (geo["switch"] if geo["is_cut"] else geo["end"]) / sr
    sw = geo["switch"] / sr
    ba = beats[(beats > 0.5) & (beats < a_end - 0.05)][-16:]
    bb = beats[(beats > b_start + 0.05) & (beats < geo["n"] / sr - 0.5)][:16]
    res = {"method": "Beat This! on the clip; straight beat lines fitted on the A-only region (before) and B-only region (after), both evaluated at the switch; low precision: extrapolation",
           "a_beats_used": int(len(ba)), "b_beats_used": int(len(bb))}
    if len(ba) < 4 or len(bb) < 4:
        res.update(status="unreliable", reason="Beat This! on the mix: too few beats in the A-only / B-only regions")
        return res
    fa, fb = _fit_beat_line(ba), _fit_beat_line(bb)
    if fa is None or fb is None or max(fa["rms_resid_ms"], fb["rms_resid_ms"]) > MIX_LINE_MAX_RMS_MS:
        res.update(status="unreliable", a_line=fa, b_line=fb,
                   reason=f"Beat This! on the mix: beat grid not straight enough in the A-only/B-only regions (rms > {MIX_LINE_MAX_RMS_MS:.0f} ms)")
        return res
    ka, ca, kb, cb = fa["period_s"], fa["intercept_s"], fb["period_s"], fb["intercept_s"]
    ratio = max(ka, kb) / min(ka, kb)
    folded = abs(ratio / 2.0 - 1.0) <= 0.08  # tracker at double/half tempo in one region
    p = min(ka, kb) if folded else ka
    phase_a = ((sw - ca) / (p if folded else ka)) % 1.0
    phase_b = ((sw - cb) / (p if folded else kb)) % 1.0
    off = ((phase_b - phase_a + 0.5) % 1.0 - 0.5) * p * 1000.0  # + = B beats late vs A grid
    res["octave_folded"] = bool(folded)
    db_near = downs[np.argmin(np.abs(downs - sw))] if len(downs) else None
    res.update({
        "a_line": fa, "b_line": fb, "tempo_change_pct": 100.0 * ((ka * (2.0 if folded and ka < kb else 1.0)) / (kb * (2.0 if folded and kb < ka else 1.0)) - 1.0),
        "beat_phase_offset_at_switch_ms": off,
        "a_grid_phase_at_switch_beats": float(phase_a if phase_a < 0.5 else phase_a - 1.0),
        "b_grid_phase_at_switch_beats": float(phase_b if phase_b < 0.5 else phase_b - 1.0),
        "nearest_mix_downbeat_to_switch_ms": (1000.0 * (db_near - sw)) if db_near is not None else None,
    })
    msgs = []
    if abs(off) > cfg.downbeat_median_max_ms:
        msgs.append(f"mix-only: B beat grid {off:+.1f} ms from A's grid at the switch")
    if geo["is_cut"]:
        status = "fail" if msgs else "pass"
    else:  # extrapolation over the whole overlap: indicative only
        status = "warn" if msgs else "pass"
        res["note"] = "blend without decks: phase extrapolated across the overlap, reported as warn at most"
    if msgs:
        amb = _phase_ambiguity(off, p, phase_support, cfg)
        if amb is not None:
            res["phase_ambiguity"] = amb
            status = "warn"
            msgs = [f"mix-only: {amb['message']}"]
    res.update(status=status, reason="; ".join(msgs) if msgs else None)
    return res


# ------------------------------------------------- dj_qa.7: phase of B at a clean entry (no rhythmic overlap)


def b_overlaps_a(meta: dict, geo: dict) -> bool:
    """B beatmatched *under* A (blend, creative overlaps with B stretched): the deck-vs-deck methods apply.

    Otherwise (cut, echo/filter out, build, halving, break, spin...) B enters clean on its downbeat after A's exit
    and its phase is checked against A's pulse extrapolated to the switch (``_rhythm_entry``)."""
    if geo["is_cut"]:
        return False
    return geo["kind"] == "blend" or bool((meta.get("stretch") or {}).get("applied"))


def _wrap_frac(x: float) -> float:
    """Fraction of a period wrapped to [-0.5, 0.5)."""
    return float((x + 0.5) % 1.0 - 0.5)


def _flux_phase(sig: np.ndarray, sr: int, t0: float, t1: float, at_s: float, period_s: float) -> Optional[dict]:
    """Second detector (independent of Beat This!): phase of the onset envelope at the beat frequency.

    Spectral flux (5 ms hop) of ``sig`` in ``[t0, t1]``; the first circular moment at ``1/period_s`` gives where the
    onsets concentrate relative to a grid through ``at_s`` (fraction of a period, + = late) and how concentrated they
    are (0..1). Sign: onsets *late* by d give a phase of -d/period (same convention as a beat line read at ``at_s``)."""
    s0, s1 = max(0, int(t0 * sr)), min(len(sig), int(t1 * sr))
    if s1 - s0 < int(2.0 * sr) or not period_s:
        return None
    hop = max(1, int(0.005 * sr))
    env = _flux(sig[s0:s1], sr, hop)
    if not np.any(env > 0):
        return None
    times = s0 / sr + np.arange(len(env)) * hop / sr
    z = np.sum(env * np.exp(-2j * np.pi * (times - at_s) / period_s))
    tot = float(np.sum(env))
    return {"phase_frac": _wrap_frac(float(np.angle(z)) / (2 * np.pi)), "concentration": float(abs(z) / tot) if tot > 0 else 0.0}


def _half_offsets_ms(beats: np.ndarray, line: dict, mid: float) -> Optional[float]:
    """Spread (ms) between the mean residuals of the beats before/after ``mid`` against the full line: a tracker
    that jumps to the off-beats or a tempo change inside the window shows up here (multi-window consensus)."""
    k, c = line["period_s"], line["intercept_s"]
    resid = beats - (c + k * np.round((beats - c) / k))
    lo, hi = resid[beats < mid], resid[beats >= mid]
    if len(lo) < 2 or len(hi) < 2:
        return None
    return float(1000.0 * abs(np.median(lo) - np.median(hi)))


def _declared_beats(meta: dict) -> tuple[Optional[float], Optional[float]]:
    """Beat of A and of B in the clip as declared by the render (``stretch``: A's local bar, B's bar after its cue)."""
    st = meta.get("stretch") or {}
    pa = float(st["a_beat_s"]) if st.get("a_beat_s") else (float(st["a_bar_s"]) / 4.0 if st.get("a_bar_s") else None)
    pb = float(st["b_bar_s"]) / 4.0 if st.get("b_bar_s") else None
    if pb and st.get("applied") and st.get("tempo_ratio_b"):
        pb = pb / float(st["tempo_ratio_b"])
    return pa, pb


def _rhythm_entry(audio, geo, decks, tracker, cfg, phase_support=None, meta=None, overlap: bool = False) -> dict:
    """Phase of B's beat at a clean entry against A's pulse continued to the switch (dj_qa.7).

    Beat This! on deck A (A's normal playback before the transition) and on deck B (after its downbeat) — or on the
    clip in the A-only / B-only regions without decks; straight beat lines on each side, evaluated at the switch.
    The measure counts only if it is *reliable*: enough beats on both sides, straight lines (rms <= 20 ms, few
    rejected beats), the two halves of each window on the same grid (consensus), the onset-flux detector agreeing
    with Beat This! within ``entry_agree_beats``, and Beat This! not known to be off the analysis grid of a track whose
    grid is confirmed by allin1/madmom (then the tracker, not the render, is suspect). Reliable and above
    ``entry_phase_fail_ms`` = ``fail`` whatever the render declares (the declared phase is only as good as the analysis
    grid it used); not reliable = ``unreliable`` (the check becomes ``unavailable``, never a pass)."""
    meta = meta or {}
    sr, n = geo["sr"], geo["n"]
    sw = geo["switch"] / sr
    a_end = sw if (geo["is_cut"] or overlap) else geo["start"] / sr
    dry = None if overlap else a_dry_end(meta)
    if dry is not None and dry["sample"] / sr < a_end - 0.05:
        a_end = dry["sample"] / sr
    win = cfg.entry_window_s
    a_sig = b_sig = None
    src = {}
    if decks:
        a_raw = decks.get("a_raw")
        if a_raw is not None and decks.get("a_raw_opt_in") and not cfg.rhythm_on_raw_a:
            a_raw = None
        a_sig = a_raw if a_raw is not None else decks.get("a")
        b_sig = decks.get("b_stretched_raw", decks.get("b_raw", decks.get("b")))
    b_from = sw
    if overlap and (a_sig is None or b_sig is None):
        return {"status": "unreliable", "reason": "overlap lines: raw decks A and B needed", "method": "overlap lines"}
    if a_sig is None:
        a_sig, src["a"] = audio, "clip (A-only region)"
    else:
        a_sig, src["a"] = _fit(a_sig, n), "deck A"
    if b_sig is None:
        b_sig, src["b"] = audio, "clip (B-only region)"
        b_from = max(sw, geo["end"] / sr)
    else:
        b_sig, src["b"] = _fit(b_sig, n), "deck B"
    cache: dict = {}

    def beats_of(sig):
        if id(sig) not in cache:
            cache[id(sig)] = np.sort(np.asarray(tracker(_as_stereo(sig), sr)["beats"], float))
        return cache[id(sig)]

    a_t0, a_t1 = max(0.3, a_end - win), a_end - 0.05
    b_t0, b_t1 = b_from + 0.05, min(n / sr - 0.3, b_from + win)
    ba = beats_of(a_sig)
    bb = beats_of(b_sig)
    ba, bb = ba[(ba >= a_t0) & (ba <= a_t1)], bb[(bb >= b_t0) & (bb <= b_t1)]
    res: dict = {"method": "Beat This! on A before its exit and on B after its downbeat (decks when available), straight beat lines "
                           "evaluated at the switch; onset-flux phase as a second detector; + = B late against A's pulse",
                 "sources": src, "a_window_s": [a_t0, a_t1], "b_window_s": [b_t0, b_t1],
                 "a_beats_used": int(len(ba)), "b_beats_used": int(len(bb)), "threshold_ms": cfg.entry_phase_fail_ms}
    problems = []
    if len(ba) < cfg.entry_min_beats or len(bb) < cfg.entry_min_beats:
        res.update(status="unreliable", reason=f"entry phase: too few beats (A {len(ba)}, B {len(bb)} < {cfg.entry_min_beats})")
        return res
    fa, fb = _fit_beat_line(ba), _fit_beat_line(bb)
    res.update(a_line=fa, b_line=fb)
    if fa is None or fb is None:
        res.update(status="unreliable", reason="entry phase: no straight beat line on A or B")
        return res
    for side, f, nb in (("A", fa, len(ba)), ("B", fb, len(bb))):
        if f["rms_resid_ms"] > MIX_LINE_MAX_RMS_MS:
            problems.append(f"{side} beat line rms {f['rms_resid_ms']:.0f} ms > {MIX_LINE_MAX_RMS_MS:.0f} ms")
        if f["n_rejected"] > max(1, int(0.25 * nb)):
            problems.append(f"{side}: {f['n_rejected']} of {nb} beats off the line")
    ka, ca, kb, cb = fa["period_s"], fa["intercept_s"], fb["period_s"], fb["intercept_s"]
    # Each side is read in its own *declared* beat (A's beat in the clip; B's own beat, unstretched at a clean entry):
    # where does the switch fall in A's beat cycle, and in B's? Both are 0 for a correct render (A's cue and B's cue are
    # downbeats placed on the switch); A's value is the tracker's bias on this clip, B's minus A's is the error.
    # Beat This! may track a side at the declared tempo, at half of it (every other beat: the phase is still valid) or
    # at double (eighths: phase known only modulo half a beat, so a half-beat displacement is invisible on that side).
    pa_decl, pb_decl = _declared_beats(meta)
    sides = {}
    for side, k, c, P, sig, t0, t1 in (("a", ka, ca, pa_decl or ka, a_sig, a_t0, a_t1), ("b", kb, cb, pb_decl or kb, b_sig, b_t0, b_t1)):
        r = k / P
        best = min((1.0, 2.0, 4.0, 0.5), key=lambda m: abs(r / m - 1.0))
        if abs(r / best - 1.0) > cfg.entry_tempo_tol:
            problems.append(f"{side.upper()} tracked beat {1000 * k:.0f} ms is not its declared beat ({1000 * P:.0f} ms) or an octave of it")
        mod = 0.5 if best == 0.5 else 1.0
        ph = (sw - c) / P
        off_s = (((ph / mod) + 0.5) % 1.0 - 0.5) * mod * P
        fl = _flux_phase(sig, sr, t0, t1, sw, P)
        sides[side] = {"tracked_beat_s": k, "declared_beat_s": P, "ratio": r, "modulus_beats": mod, "switch_phase_ms": 1000.0 * off_s,
                       "flux": fl, "flux_phase_ms": None if fl is None else 1000.0 * ((((fl["phase_frac"] * 1.0) / mod) + 0.5) % 1.0 - 0.5) * mod * P}
    # sign convention: + = B's beats late. A beat line through c puts the switch at phase (sw - c)/P: a beat *after* the
    # switch (late) gives a negative phase, hence the minus signs.
    pb_, mb = sides["b"]["declared_beat_s"], sides["b"]["modulus_beats"]
    span = mb * pb_ * 1000.0

    def wrap_ms(x):
        return ((x / span) + 0.5) % 1.0 * span - 0.5 * span

    off = wrap_ms(-(sides["b"]["switch_phase_ms"] - sides["a"]["switch_phase_ms"]))
    folded = mb < 1.0 or sides["a"]["modulus_beats"] < 1.0
    p = pb_
    phase_a, phase_b = sides["a"]["switch_phase_ms"], sides["b"]["switch_phase_ms"]
    spread_a = _half_offsets_ms(ba, fa, 0.5 * (a_t0 + a_t1))
    spread_b = _half_offsets_ms(bb, fb, 0.5 * (b_t0 + b_t1))
    for side, sp in (("A", spread_a), ("B", spread_b)):
        if sp is None:
            problems.append(f"{side}: too few beats in one half of the window (consensus not measurable)")
        elif sp > cfg.entry_consensus_ms:
            problems.append(f"{side}: the two halves of the window disagree by {sp:.0f} ms (> {cfg.entry_consensus_ms:.0f} ms)")
    fla, flb = sides["a"]["flux"], sides["b"]["flux"]
    flux = None
    if fla and flb:
        flux = {"a": fla, "b": flb, "offset_ms": wrap_ms(-(sides["b"]["flux_phase_ms"] - sides["a"]["flux_phase_ms"]))}
        flux["concentrated"] = bool(min(fla["concentration"], flb["concentration"]) >= cfg.entry_flux_min_concentration)
        flux["disagreement_ms"] = abs(wrap_ms(flux["offset_ms"] - off))
    res.update({"octave_folded": bool(folded), "period_s": p, "phase_modulus_beats": mb, "sides": sides,
                "declared_beat_s": {"a": pa_decl, "b": pb_decl}, "tracked_beat_s": {"a": ka, "b": kb},
                "beat_phase_offset_at_switch_ms": off, "a_switch_phase_ms": phase_a, "b_switch_phase_ms": phase_b,
                "consensus_spread_ms": {"a": spread_a, "b": spread_b}, "flux": flux})
    if flux is None or not flux["concentrated"]:
        problems.append("onset-flux detector not measurable (onsets not concentrated at the beat frequency)")
    elif flux["disagreement_ms"] > cfg.entry_agree_beats * p * 1000.0:
        problems.append(f"detectors disagree: Beat This! {off:+.0f} ms, onset flux {flux['offset_ms']:+.0f} ms "
                        f"(> {cfg.entry_agree_beats:g} beat apart)")
    support = phase_support or {}
    off_grid = [k.upper() for k in ("a", "b") if (support.get(k) or {}).get("beat_this_off_grid")]
    if off_grid and all((support.get(k) or {}).get("supported") for k in ("a", "b")):
        amb = _phase_ambiguity(off, p, support, cfg, tol_ms=cfg.entry_phase_fail_ms)
        if amb is not None:
            res["phase_ambiguity"] = amb
            problems.append(amb["message"])
    res["reliability_problems"] = problems
    if problems:
        res.update(status="unreliable", reliable=False,
                   reason="entry phase not measurable with confidence: " + "; ".join(problems) + f" (Beat This! offset {off:+.0f} ms)")
        return res
    bad = abs(off) > cfg.entry_phase_fail_ms
    res.update(status="fail" if bad else "pass", reliable=True,
               reason=(f"B's beat {off:+.0f} ms from A's pulse at the switch (reliable: two detectors agree, "
                       f"> {cfg.entry_phase_fail_ms:.0f} ms), whatever phase the render declares") if bad else None)
    return res


def vocal_activity(stem: np.ndarray, sr: int, gain: Optional[np.ndarray] = None, threshold: Optional[float] = None) -> tuple[np.ndarray, float]:
    """Frame (20 ms) in cui lo stem, dopo ``gain``, supera la soglia dello stem grezzo."""
    from scipy.ndimage import binary_closing

    hop = int(round(VOCAL_HOP_S * sr))
    x = _as_stereo(stem).mean(axis=1)
    raw = _frame_rms(x, hop, hop)
    if threshold is None:
        voiced = raw[raw > VOCAL_ABS_FLOOR]
        threshold = max(VOCAL_ABS_FLOOR, float(np.percentile(voiced, 95)) * 10 ** (VOCAL_REL_DB / 20.0)) if len(voiced) else float("inf")
    heard = _frame_rms(x * gain, hop, hop) if gain is not None else raw
    act = heard > threshold
    close = max(1, int(round(VOCAL_CLOSE_S / VOCAL_HOP_S)))
    act = binary_closing(act, structure=np.ones(close, dtype=bool), border_value=0) | act
    return act, float(threshold)


def deck_gains(meta: dict, n: int, decks: Optional[dict] = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Guadagno udibile della voce di A e B nella clip.

    Ordine: ``decks['gains']`` di dj_render (fader x ducking); altrimenti forme
    dichiarate dal renderer (cut: A fino allo switch, B dopo; blend: quarto di
    seno/coseno); altrimenti entrambi pieni nella transizione (prudente).
    """
    geo = transition_geometry(meta, n)
    if decks and isinstance(decks.get("gains"), dict) and "fader_a" in decks["gains"]:
        g = decks["gains"]
        ga = np.asarray(g["fader_a"], float)[:n] * np.asarray(g.get("duck_a_vocal", np.ones(n)), float)[:n]
        gb = np.asarray(g["fader_b"], float)[:n] * np.asarray(g.get("section_gain_b", np.ones(n)), float)[:n]
        return ga, gb, "renderer gains (decks['gains']: fader x vocal ducking; B x section gain)"
    t = np.arange(n)
    s, w, e = geo["start"], geo["switch"], geo["end"]
    if geo["is_cut"]:
        return (t < w).astype(float), (t >= s).astype(float), "cut: A until the switch, B from the crossfade start"
    if geo["kind"] == "blend":
        v = np.clip((t - s) / max(w - s, 1), 0, 1)
        u = np.clip((t - w) / max(e - w, 1), 0, 1)
        ga = np.where(t >= e, 0.0, np.cos(0.5 * np.pi * u))
        gb = np.where(t < s, 0.0, np.sin(0.5 * np.pi * v))
        note = "blend: fader shapes declared by dj_render (ducking not visible without decks)"
        return ga, gb, note
    return (t < e).astype(float), (t >= s).astype(float), "unknown kind: both decks assumed full during the transition (conservative)"


def check_vocal_clash(vocals_a, vocals_b, meta, decks, geo, cfg, thresholds=(None, None)) -> dict:
    if vocals_a is None or vocals_b is None:
        return _check("skipped", reason="vocal stems of both decks (in clip coordinates) not provided")
    n, sr = geo["n"], geo["sr"]
    ga, gb, note = deck_gains(meta, n, decks)
    act_a, thr_a = vocal_activity(_fit(vocals_a, n), sr, ga, thresholds[0])
    act_b, thr_b = vocal_activity(_fit(vocals_b, n), sr, gb, thresholds[1])
    m = min(len(act_a), len(act_b))
    both = act_a[:m] & act_b[:m]
    clash = float(both.sum() * VOCAL_HOP_S)
    status = "fail" if clash > cfg.vocal_clash_max_s else "pass"
    return _check(status, clash_s=clash, longest_s=_longest_run(both) * VOCAL_HOP_S, a_heard_s=float(act_a.sum() * VOCAL_HOP_S),
                  b_heard_s=float(act_b.sum() * VOCAL_HOP_S), threshold_s=cfg.vocal_clash_max_s, gains=note,
                  reason=f"both vocals heard for {clash:.2f} s > {cfg.vocal_clash_max_s:.1f} s" if status == "fail" else None,
                  method="stem RMS 20 ms after fader/ducking gain above max(0.002, p95 - 24 dB) of the raw stem; gaps < 150 ms closed")


def _a_exit(geo: dict, decks: Optional[dict]) -> tuple[int, Optional[float], str]:
    """Campione in cui A esce del tutto e durata della discesa da -6 dB al silenzio."""
    sr = geo["sr"]
    if decks and decks.get("a") is not None:
        env = _frame_rms(_fit(decks["a"], geo["n"]), int(0.002 * sr), int(0.004 * sr))
        if env.max() > 0:
            alive = np.where(env > env.max() * 1e-3)[0]  # -60 dB from the deck peak
            last = int(alive[-1]) if len(alive) else 0
            # Reference level = A before its exit starts. For a cut that is the last
            # second before the end; for a fader exit (blend) the last second is
            # already inside the fade, whose median made every gradual fade look
            # "abrupt" (correction 20/10/2026): measure the second before the switch.
            ref_end = last - 25
            if not geo["is_cut"]:
                ref_end = min(ref_end, int(geo["switch"] / (0.002 * sr)))
            ref = float(np.median(env[max(0, ref_end - 475) : max(1, ref_end)])) if ref_end > 0 else float(env.max())
            above = np.where(env[: last + 1] > 0.5 * ref)[0]
            fade = (last - int(above[-1])) * 0.002 if len(above) else None
            return (last + 1) * int(0.002 * sr), fade, "measured on deck A (-60 dB from its peak)"
    if geo["is_cut"]:
        return geo["switch"], 0.0, "metadata switch (cut)"
    return geo["end"], (geo["end"] - geo["switch"]) / sr, "metadata transition end"


def a_dry_end(meta: Optional[dict]) -> Optional[dict]:
    """Campione della clip in cui finisce la riproduzione secca e continua di A (con la sua voce), se dichiarato.

    * ``effect.dry_a_cut_output_sample`` (``dj_effects.echo_out``): il dry è tagliato, la coda d'eco continua;
    * ``a_map`` (``dj_creative``): fine della catena di segmenti contigui a velocità 1 con la voce di A che parte
      dall'inizio della clip (dopo comincia un loop, uno stutter o il mute della voce).
    Senza questi metadati restituisce None (il chiamante usa l'uscita misurata del deck A).
    """
    if not meta:
        return None
    eff = meta.get("effect") or {}
    if eff.get("dry_a_cut_output_sample") is not None:
        return {"sample": int(eff["dry_a_cut_output_sample"]), "source": "effect.dry_a_cut_output_sample (echo: the delay tail continues after it)"}
    amap = [m for m in (meta.get("a_map") or []) if m.get("out_start") is not None and m.get("out_end") is not None]
    end = off = None
    for m in sorted(amap, key=lambda m: (int(m["out_start"]), int(m["out_end"]))):
        if m.get("src_start") is None or (m.get("rate", 1.0) or 0.0) != 1.0 or not m.get("vocals", True):
            break
        o0, o1, s0 = int(m["out_start"]), int(m["out_end"]), int(m["src_start"])
        if end is None:
            if o0 > 0:
                break
            end, off = o1, s0 - o0
            continue
        if o0 != end or s0 - o0 != off:
            break
        end = o1
    if end is None:
        return None
    return {"sample": int(end), "source": "a_map: end of A's continuous playback with its vocal (a loop, stutter or vocal mute starts there)"}


def _voice_band_db(x: np.ndarray, sr: int, s0: int, s1: int) -> Optional[float]:
    from scipy.signal import butter, sosfiltfilt

    s0, s1 = max(0, int(s0)), min(len(x), int(s1))
    if s1 - s0 < int(0.02 * sr):
        return None
    seg = _as_stereo(x[max(0, s0 - int(0.05 * sr)) : min(len(x), s1 + int(0.05 * sr))]).mean(axis=1)
    y = sosfiltfilt(butter(4, [1000.0, 4000.0], btype="band", fs=sr, output="sos"), seg)
    lo = s0 - max(0, s0 - int(0.05 * sr))
    core = y[lo : lo + (s1 - s0)]
    r = float(np.sqrt(np.mean(core ** 2))) if len(core) else 0.0
    return 20.0 * math.log10(r) if r > 0 else None


def _effect_exit_cut(geo, decks, words_a, vocals_a, cfg, threshold, meta, exit_s, base) -> dict:
    """Parola tagliata quando A esce con un effetto (regola [H] in ``EFFECT_WORD_EPS_S``)."""
    sr = geo["sr"]
    dry = a_dry_end(meta)
    point, how = (min(dry["sample"], exit_s), dry["source"]) if dry else (exit_s, base["exit_source"])
    if dry and exit_s < dry["sample"]:
        how += "; deck A silent earlier (measured exit used)"
    t = point / sr
    base = dict(base, rule="effect exit: word cut if A's dry, continuous playback stops inside it (a tail does not complete it) [H]",
                a_dry_end_s=t, a_dry_end_source=how, eps_s=cfg.effect_word_eps_s)
    sev = "warn" if cfg.effect_word_cut == "warn" else "fail"
    faded = None
    if decks and decks.get("a") is not None:
        ref = _voice_band_db(decks["a"], sr, geo["start"] - 3 * sr, geo["start"])
        here = _voice_band_db(decks["a"], sr, point - int(EFFECT_FADED_WIN_S * sr), point)
        if ref is not None and here is not None:
            base["a_voice_band_at_dry_end_db"] = here - ref
            if here - ref < EFFECT_FADED_DB:
                # only used to discount a word that does cross the stop: a quieter band may also mean "no voice here"
                faded = f"A's voice band already {here - ref:.0f} dB down where its dry playback stops"
    eps = cfg.effect_word_eps_s
    if words_a:
        hits, seen = [], set()
        for w in words_a:
            end = float(w.get("end_unclipped_s", w["end_s"]))
            key = (str(w.get("text", "")), round(float(w.get("start_unclipped_s", w["start_s"])), 3), round(end, 3))
            if w["start_s"] < t - eps and end > t + eps and key not in seen:  # same source word in two map segments: once
                seen.add(key)
                hits.append(w)
        if hits and faded:
            return _check("not_applicable", coverage="not_applicable", words_crossing=hits, reason=faded + ": the crossing word is not heard", **base)
        return _check(sev if hits else "pass", coverage="measured", evidence="ASR words of A", words_cut=hits,
                      reason=(f"A word(s) {' '.join(str(w.get('text', '')) for w in hits)!r} cut where A's dry playback stops "
                              f"({geo['exit_effect'] or 'loop/stutter'} exit at {t:.2f} s)") if hits else None, **base)
    if vocals_a is not None:
        act, _thr = vocal_activity(_fit(vocals_a, geo["n"]), sr, None, threshold)
        k, w = int(round(t / VOCAL_HOP_S)), max(1, int(round(EFFECT_STEM_WIN_S / VOCAL_HOP_S)))
        before_any = bool(act[max(0, k - w) : k].any()) if k > 0 else False
        before = bool(act[max(0, k - w) : k].all()) if k > 0 else False
        if not (meta or {}).get("a_map"):
            # dj_render/dj_effects: vocals_a is A's source stem in clip coordinates, so it continues past the stop point.
            after = bool(act[k : k + w].all())
            cut = before and after
            if cut and faded:
                return _check("not_applicable", coverage="not_applicable", reason=faded, **base)
            return _check(sev if cut else "pass", coverage="measured", evidence="vocal stem of A (no ASR words): active on both sides of the stop",
                          stem_active_before=before, stem_active_after=after,
                          reason=f"A's vocal stem active across the point where its dry playback stops ({t:.2f} s)" if cut else None, **base)
        if not before_any:  # dj_creative: the heard stem after the stop is the loop/tail, but silence before it is enough
            return _check("pass", coverage="measured", evidence="heard vocal of A silent in the 60 ms before its dry playback stops",
                          stem_active_before=False, **base)
    return _check("skipped", coverage="unavailable",
                  reason="no ASR words of A and its vocal is active at the stop (after a loop/stutter the heard stem does not show "
                         "the source continuation)" if vocals_a is not None else "no ASR words or vocal stem for A", **base)


def check_verse_cut(geo, decks, words_a, vocals_a, cfg, threshold=None, meta: Optional[dict] = None) -> dict:
    sr = geo["sr"]
    exit_s, fade_s, how = _a_exit(geo, decks)
    exit_t = exit_s / sr
    base = {"a_exit_s": exit_t, "a_exit_fade_s": fade_s, "exit_source": how, "exit_effect": geo["exit_effect"], "guard_s": cfg.word_guard_s}
    dry = a_dry_end(meta)
    loop_exit = dry is not None and dry["sample"] < exit_s - int(0.02 * sr)  # A keeps sounding (loop/stutter) after its dry stops
    if geo["exit_effect"] or loop_exit:
        if cfg.effect_word_cut == "off":  # behaviour of dj_qa.4 (kept for reproductions): not measured
            return _check("not_applicable", coverage="not_applicable", reason=f"A exits through {geo['exit_effect'] or 'a loop'}: check off (effect_word_cut='off')", **base)
        return _effect_exit_cut(geo, decks, words_a, vocals_a, cfg, threshold, meta, exit_s, base)
    abrupt = geo["is_cut"] or (fade_s is not None and fade_s < cfg.abrupt_exit_s)
    if not abrupt:
        return _check("not_applicable", coverage="not_applicable", reason="A fades out gradually: check not applicable", **base)
    lo, hi = exit_t - cfg.word_guard_s, exit_t + cfg.word_guard_s
    if words_a:
        hits = [w for w in words_a if w["end_s"] > lo and w["start_s"] < hi]
        status = "fail" if hits else "pass"
        return _check(status, evidence="ASR words of A", words_near_exit=hits,
                      reason=(f"A word(s) {' '.join(str(w.get('text','')) for w in hits)!r} active within +-{1000*cfg.word_guard_s:.0f} ms of A's exit") if hits else None, **base)
    if vocals_a is not None:
        act, thr = vocal_activity(_fit(vocals_a, geo["n"]), sr, None, threshold)
        f0, f1 = int(max(0, lo) / VOCAL_HOP_S), int(math.ceil(hi / VOCAL_HOP_S))
        active = bool(act[f0:f1].any())
        return _check("fail" if active else "pass", evidence="vocal stem activity of A (no ASR words)", stem_active_near_exit=active,
                      reason=f"A's vocal stem active within +-{1000*cfg.word_guard_s:.0f} ms of A's exit" if active else None, **base)
    return _check("skipped", reason="no ASR words or vocal stem for A", **base)


def b_audible_start(meta: dict, geo: dict, decks: Optional[dict], cfg: QAConfig) -> dict:
    """Campione della clip in cui B diventa udibile e se l'ingresso è brusco (``abrupt``) o a fader.

    Ordine: ``meta['b_entry']`` dichiarato (lead-in di ``dj_render.add_b_lead_in``); se B sale col fader (blend o due
    brani sovrapposti) ingresso graduale; deck B del render (-60 dB dal picco, salita a -6 dB entro ``abrupt_exit_s``);
    metadati (``b_map``, ``downbeat_alignment.b_entry_output_sample``, switch del cut).
    """
    sr = geo["sr"]
    be = meta.get("b_entry") or {}
    if be.get("audible_output_sample") is not None:
        return {"sample": int(be["audible_output_sample"]), "abrupt": True, "source": "meta.b_entry (declared lead-in)"}
    overlap = geo["kind"] == "blend" or (bool((meta.get("stretch") or {}).get("applied")) and not geo["is_cut"])
    if overlap:
        return {"sample": geo["start"], "abrupt": False, "source": "B enters with a fader over A (blend / overlap)"}
    if decks and decks.get("b") is not None:
        hop = max(1, int(0.002 * sr))
        env = _frame_rms(_fit(decks["b"], geo["n"]), hop, 2 * hop)
        if env.size and env.max() > 0:
            alive = np.flatnonzero(env > env.max() * 1e-3)
            if alive.size:
                k0 = int(alive[0])
                ref = float(np.median(env[k0 + int(0.05 * sr / hop): k0 + int(1.0 * sr / hop)])) if len(env) > k0 + 30 else float(env.max())
                up = np.flatnonzero(env[k0:] > 0.5 * ref) if ref > 0 else np.array([0])
                rise = (int(up[0]) * hop / sr) if up.size else None
                abrupt = rise is not None and rise <= cfg.abrupt_exit_s
                return {"sample": k0 * hop, "abrupt": bool(abrupt), "rise_s": rise, "source": "measured on deck B (-60 dB from its peak)"}
    bm = [m for m in (meta.get("b_map") or []) if m.get("out_start") is not None]
    if bm:
        return {"sample": int(min(int(m["out_start"]) for m in bm)), "abrupt": True, "source": "b_map (first B segment)"}
    da = meta.get("downbeat_alignment") or {}
    if da.get("b_entry_output_sample") is not None:
        return {"sample": int(da["b_entry_output_sample"]), "abrupt": True, "source": "downbeat_alignment.b_entry_output_sample"}
    if geo["is_cut"]:
        return {"sample": geo["switch"], "abrupt": True, "source": "metadata switch (cut)"}
    return {"sample": None, "abrupt": None, "source": "unknown"}


def _b_beat_s(meta: dict) -> tuple[float, str]:
    st = meta.get("stretch") or {}
    if st.get("b_bar_s"):
        return float(st["b_bar_s"]) / 4.0, "stretch.b_bar_s / 4"
    return B_DEFAULT_BEAT_S, "default 0.5 s (B's bar not in the metadata)"


def check_b_entry(geo: dict, decks: Optional[dict], words_b: Optional[Sequence[dict]], vocals_b, cfg: QAConfig,
                  meta: Optional[dict] = None) -> dict:
    """Parola o frase di B troncata nel punto in cui B diventa udibile (regole [H] in ``B_ENTRY_EPS_S``)."""
    meta = meta or {}
    if cfg.b_entry_check == "off":
        return _check("not_applicable", coverage="not_applicable", reason="b_entry_check='off' (behaviour of dj_qa.5)")
    sr = geo["sr"]
    st = b_audible_start(meta, geo, decks, cfg)
    base = {"b_audible_s": None if st["sample"] is None else st["sample"] / sr, "b_audible_source": st["source"],
            "eps_s": cfg.b_entry_eps_s, "rest_beats": cfg.b_rest_beats, "lead_in": meta.get("b_entry"),
            "rule": "word of B cut if it starts before and ends after B's audible start (> eps); phrase begun earlier if B's "
                    "previous word ends < rest_beats before the first heard word [H]"}
    if st["sample"] is None:
        return _check("skipped", coverage="unavailable", reason="B's audible start not found (no decks or entry metadata)", **base)
    if not st["abrupt"]:
        return _check("not_applicable", coverage="not_applicable", reason=f"B enters gradually ({st['source']}): no truncation", **base)
    sev = "warn" if cfg.b_entry_check == "warn" else "fail"
    t = st["sample"] / sr
    eps = cfg.b_entry_eps_s
    beat, beat_src = _b_beat_s(meta)
    base.update(b_beat_s=beat, b_beat_source=beat_src)
    if words_b:
        ws = sorted(words_b, key=lambda w: float(w["start_s"]))
        cut = [w for w in ws if float(w["start_s"]) < t - eps and float(w["end_s"]) > t + eps]
        heard = [w for w in ws if float(w["start_s"]) >= t - eps and float(w["end_s"]) > t + eps]
        before = [w for w in ws if float(w["end_s"]) <= t + eps]
        first = heard[0] if heard else None
        prev = before[-1] if before else None
        gap = (float(first["start_s"]) - float(prev["end_s"])) if (first is not None and prev is not None) else None
        mid = bool(not cut and gap is not None and gap < cfg.b_rest_beats * beat)
        info = dict(base, evidence="words of B", words_cut=cut, first_heard_word=first, previous_word=prev,
                    gap_before_first_s=gap)
        if cut:
            return _check(sev, coverage="measured", problem="b_word_cut",
                          reason=f"B word(s) {' '.join(str(w.get('text', '')) for w in cut)!r} cut where B becomes audible ({t:.2f} s)", **info)
        if mid:
            return _check(sev, coverage="measured", problem="b_phrase_started_before",
                          reason=(f"B's first heard word {first.get('text', '')!r} follows {prev.get('text', '')!r} by {gap:.2f} s "
                                  f"(< {cfg.b_rest_beats:g} beat, no breath): B enters halfway through a phrase ({t:.2f} s)"), **info)
        return _check("pass", coverage="measured", **info)
    if vocals_b is not None and not meta.get("b_map"):
        # dj_render/dj_effects: vocals_b is B's source stem in clip coordinates (continues before B's audible start)
        act, _thr = vocal_activity(_fit(vocals_b, geo["n"]), sr, None, None)
        k, w = int(round(t / VOCAL_HOP_S)), max(1, int(round(EFFECT_STEM_WIN_S / VOCAL_HOP_S)))
        before = bool(act[max(0, k - w):k].all()) if k > 0 else False
        after = bool(act[k:k + w].all())
        return _check(sev if (before and after) else "pass", coverage="measured", evidence="vocal stem of B (no words)",
                      stem_active_before=before, stem_active_after=after, problem="b_word_cut" if before and after else None,
                      reason=f"B's vocal stem active across the point where B becomes audible ({t:.2f} s)" if before and after else None, **base)
    return _check("skipped", coverage="unavailable", reason="no words of B and no source-continuous vocal stem of B", **base)


# ------------------------------------------------------- dj_qa.7: clicks and off-tempo echoes


def junction_points(meta: dict, geo: dict) -> list[dict]:
    """Samples of the clip where the render joins material: switch, transition start/end, end of A's dry playback,
    B's audible start, and the borders of the declared ``a_map``/``b_map`` segments (loops, stutters, chops)."""
    n = geo["n"]
    pts = [("switch", geo["switch"]), ("transition_start", geo["start"]), ("transition_end", geo["end"])]
    dry = a_dry_end(meta)
    if dry:
        pts.append(("a_dry_end", dry["sample"]))
    be = meta.get("b_entry") or {}
    if be.get("audible_output_sample") is not None:
        pts.append(("b_audible", int(be["audible_output_sample"])))
    for key in ("a_map", "b_map"):
        for m in meta.get(key) or []:
            for edge in ("out_start", "out_end"):
                if m.get(edge) is not None:
                    pts.append((f"{key}.{edge}", int(m[edge])))
    out, seen = [], []
    for name, smp in pts:
        smp = int(smp)
        if 0 < smp < n and all(abs(smp - q) > int(0.002 * geo["sr"]) for q in seen):
            seen.append(smp)
            out.append({"name": name, "sample": smp})
    return out


def check_click(audio: np.ndarray, geo: dict, meta: dict, cfg: QAConfig) -> dict:
    """Clic alla giunta (.7). [H] Una discontinuità del segnale (campioni che saltano, impulso, gradino) ha energia fino
    a Nyquist; la musica delle sorgenti (quasi tutte da codifiche con passa-basso) ne ha pochissima sopra ~20 kHz.
    Statistica: massimo di |banda 20,5-23,5 kHz| nei ±5 ms attorno a ogni punto di giunta diviso il percentile 99,9 della
    stessa banda nel secondo prima e dopo (esclusi ±20 ms dal punto): un clic è un evento che la clip non ha altrove.
    Soglia ``click_ratio`` [E] calibrata sulle giunte naturali. Sorgenti a 44,1 kHz o meno: ``unavailable``."""
    if cfg.click_check == "off":
        return _check("not_applicable", coverage="not_applicable", reason="click_check='off'")
    from scipy.signal import butter, sosfiltfilt

    sr = geo["sr"]
    lo, hi = CLICK_BAND_HZ
    if hi >= 0.5 * sr:
        return _check("skipped", coverage="unavailable", reason=f"sample rate {sr} Hz: no band above {lo / 1000:.1f} kHz")
    mono = _as_stereo(audio).mean(axis=1)
    sos = butter(8, [lo, hi], btype="band", fs=sr, output="sos")
    rows, worst = [], None
    for pt in junction_points(meta, geo):
        p = pt["sample"]
        s0, s1 = max(0, p - sr), min(len(mono), p + sr)
        if s1 - s0 < int(0.2 * sr):
            continue
        h = np.abs(sosfiltfilt(sos, mono[s0:s1]))
        c = p - s0
        w = int(CLICK_WIN_S * sr)
        centre = float(h[max(0, c - w):c + w].max())
        mask = np.ones(len(h), bool)
        mask[max(0, c - int(0.02 * sr)):c + int(0.02 * sr)] = False
        ref = float(np.percentile(h[mask], 99.9)) if mask.any() else 0.0
        stat = centre / max(ref, 1e-7)
        row = {"point": pt["name"], "t_s": p / sr, "ratio": stat, "peak": centre, "reference_p999": ref}
        rows.append(row)
        if worst is None or stat > worst["ratio"]:
            worst = row
    if worst is None:
        return _check("skipped", coverage="unavailable", reason="no junction point measurable")
    bad = worst["ratio"] > cfg.click_ratio
    warn = (not bad) and worst["ratio"] > cfg.click_warn_ratio
    return _check(cfg.click_severity if bad else ("warn" if warn else "pass"), coverage="measured", statistic=worst["ratio"], worst_point=worst["point"],
                  threshold=cfg.click_ratio, points=rows,
                  reason=(f"click at the junction ({worst['point']}, {worst['t_s']:.3f} s): energy above {lo / 1000:.1f} kHz "
                          f"{worst['ratio']:.1f}x the clip's own maximum around it") if bad else (
                      f"possible soft click at {worst['point']} ({worst['t_s']:.3f} s): ratio {worst['ratio']:.1f}" if warn else None),
                  method="|band 20.5-23.5 kHz| max within +-5 ms of each junction / its 99.9th percentile in +-1 s (excluding +-20 ms)")


def _echo_templates(a: np.ndarray, sr: int, end: int, beat: int, n_templates: int = 3) -> list[tuple[int, np.ndarray]]:
    """The ``n_templates`` strongest, non-overlapping 60 ms transients (high-passed at 300 Hz) of A's last audible beat
    before ``end`` (an echo may repeat the whole beat with a window that hides its first transient)."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, 300.0, btype="high", fs=sr, output="sos")
    for back in range(0, 8):
        e = end - back * beat
        seg = a[max(0, e - beat):e]
        if len(seg) == beat and float(np.sqrt(np.mean(seg ** 2))) > 1e-3:
            hp = sosfiltfilt(sos, seg)
            L = int(ECHO_TEMPLATE_S * sr)
            env = np.convolve(hp ** 2, np.ones(L) / L, mode="valid")
            out = []
            for _ in range(n_templates):
                k = int(np.argmax(env))
                if env[k] <= 0:
                    break
                out.append((max(0, e - beat) + k, hp[k:k + L] * np.hanning(L)))
                env[max(0, k - L):k + L] = 0.0
            return out
    return []


def _echo_progressions(mono, sr, sos, t_at, tmpl, dry, geo, end, beat, beat_s, exit_s, grid):
    """Copies of one template after A's exit (see ``check_echo_tempo``): (progressions, n_peaks, region_s) or None."""
    from scipy.signal import fftconvolve, find_peaks, sosfiltfilt

    r0, r1 = max(t_at + len(tmpl), (dry["sample"] if dry else geo["start"]) - int(0.05 * sr)), min(len(mono), end + int(ECHO_WINDOW_S * sr))
    if r1 - r0 < 4 * len(tmpl):
        return None
    reg = sosfiltfilt(sos, mono[r0:r1])
    num = fftconvolve(reg, tmpl[::-1], mode="valid")
    den = np.sqrt(np.maximum(fftconvolve(reg ** 2, np.ones(len(tmpl)), mode="valid"), 1e-20)) * float(np.linalg.norm(tmpl))
    ncc = num / np.maximum(den, 1e-12)
    med = float(np.median(ncc))
    mad = 1.4826 * float(np.median(np.abs(ncc - med))) + 1e-9
    z = (ncc - med) / mad
    idx, _ = find_peaks(z, height=ECHO_MIN_Z, distance=max(1, int(0.15 * beat)))
    idx = idx[ncc[idx] >= ECHO_MIN_NCC]
    times = (r0 + idx) / sr
    vals = ncc[idx]
    progs = []
    # arithmetic progressions t0 + k*d (consecutive copies) among the peaks: d in [0.2, 1.1] beats, tolerance 8 ms, first
    # copy within one beat of A's exit (an echo starts where A's dry playback stops; B's own periodic material later does
    # not count). For each start only the progression with the strongest copies (sum of correlations) is kept.
    for i in range(len(times)):
        if not (exit_s - 0.05 <= times[i] <= end / sr + beat_s):
            continue
        best_i = None
        for j in range(i + 1, len(times)):
            d = times[j] - times[i]
            if not (0.2 * beat_s <= d <= 1.1 * beat_s):
                continue
            hits, score = [times[i], times[j]], float(vals[i] + vals[j])
            for k in range(2, 8):
                m = np.abs(times - (times[i] + k * d))
                if not (m.size and m.min() <= ECHO_TOL_S):
                    break
                hits.append(float(times[int(np.argmin(m))]))
                score += float(vals[int(np.argmin(m))])
            if len(hits) >= ECHO_MIN_REPEATS and (best_i is None or score > best_i["score"]):
                dd = float(np.polyfit(np.arange(len(hits)), np.asarray(hits), 1)[0])
                sb = dd / beat_s
                best_i = {"repeats": len(hits), "spacing_s": dd, "spacing_beats": float(sb), "first_s": float(times[i]), "score": score,
                          "mean_ncc": score / len(hits), "off_grid_ms": float(1000.0 * beat_s * min(abs(sb - g) for g in grid))}
        if best_i:
            progs.append(best_i)
    return progs, int(len(idx)), (r0 / sr, r1 / sr)


def check_echo_tempo(audio: np.ndarray, geo: dict, decks: Optional[dict], meta: dict, cfg: QAConfig) -> dict:
    """Eco fuori tempo (.7). [P] Webber 2008: l'eco di uscita va a tempo (1 battito o frazioni musicali; dj_effects
    ammette 1/2 e 1). [H] Misura: le copie dei tre transienti più forti dell'ultimo battito udibile di A (filtro adattato,
    passa-alto 300 Hz, correlazione normalizzata) nei 3 s dopo l'uscita di A; se almeno ``ECHO_MIN_REPEATS`` picchi
    consecutivi (z robusto >= ``ECHO_MIN_Z``, ncc >= ``ECHO_MIN_NCC``) formano una progressione aritmetica che parte entro
    un battito dall'uscita, il passo deve essere un multiplo di 1/4 o 1/3 di
    battito di A entro ``echo_grid_tol_ms``. Nessuna progressione = nessuna eco ripetuta: ``pass`` misurato."""
    if cfg.echo_check == "off":
        return _check("not_applicable", coverage="not_applicable", reason="echo_check='off'")
    from scipy.signal import butter

    sr = geo["sr"]
    beat_s = _beat_s(meta)
    if not beat_s:
        return _check("skipped", coverage="unavailable", reason="A's beat not declared")
    beat = int(round(beat_s * sr))
    a = (decks or {}).get("a") if decks else None
    end = geo["switch"]
    dry = a_dry_end(meta)
    a_sig = _as_stereo(a if a is not None else audio).mean(axis=1)
    tms = _echo_templates(a_sig, sr, min(end, dry["sample"]) if (dry and a is None) else end, beat)
    if not tms:
        return _check("skipped", coverage="unavailable", reason="A silent in the 8 beats before its exit")
    mono = _as_stereo(audio).mean(axis=1)
    sos = butter(2, 300.0, btype="high", fs=sr, output="sos")
    exit_s = min(end, dry["sample"]) / sr if dry else end / sr
    grid = [m / 4.0 for m in range(1, 5)] + [m / 3.0 for m in range(1, 4)]
    progs, n_peaks, regions = [], 0, []
    for t_at, tmpl in tms:
        r = _echo_progressions(mono, sr, sos, t_at, tmpl, dry, geo, end, beat, beat_s, exit_s, grid)
        if r is None:
            continue
        progs += r[0]
        n_peaks += r[1]
        regions.append(r[2])
    if not regions:
        return _check("skipped", coverage="unavailable", reason="no room after A's exit")
    base = {"templates_at_s": [t / sr for t, _ in tms], "region_s": list(regions[0]), "peaks": n_peaks, "beat_s": beat_s, "a_exit_s": exit_s,
            "method": "matched filter of A's 3 strongest last-beat transients (HP 300 Hz) after its exit; consecutive arithmetic "
                      "progressions of peaks (>= 3 copies, z >= 6, ncc >= 0.2) starting within a beat of the exit"}
    if not progs:
        return _check("pass", coverage="measured", statistic=0.0, spacing_beats=None, threshold=cfg.echo_grid_tol_ms,
                      reason=None, note="no repeated copies of A after its exit", **base)
    bad = [pg for pg in progs if pg["off_grid_ms"] > cfg.echo_grid_tol_ms]
    worst = max(bad, key=lambda pg: (pg["repeats"], pg["off_grid_ms"])) if bad else max(progs, key=lambda pg: pg["repeats"])
    return _check(cfg.echo_severity if bad else "pass", coverage="measured", statistic=worst["off_grid_ms"], spacing_beats=worst["spacing_beats"],
                  repeats=worst["repeats"], first_repeat_s=worst["first_s"], threshold=cfg.echo_grid_tol_ms, progressions=progs[:12],
                  reason=(f"echo of A repeating every {worst['spacing_beats']:.3f} beat ({worst['repeats']} copies from {worst['first_s']:.2f} s): "
                          f"{worst['off_grid_ms']:.0f} ms from a multiple of 1/4 or 1/3 beat (> {cfg.echo_grid_tol_ms:.0f} ms)") if bad else None, **base)


# ------------------------------------------------------- dj_qa.7: phrase position of B at its entry

_ANALYSIS_CACHE: dict = {}


def _load_json_cached(path: Path) -> Optional[dict]:
    key = str(path)
    if key not in _ANALYSIS_CACHE:
        try:
            _ANALYSIS_CACHE[key] = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _ANALYSIS_CACHE[key] = None
    return _ANALYSIS_CACHE[key]


def _resolve(recorded: Optional[str]) -> Optional[Path]:
    if not recorded:
        return None
    from musicmix.paths import localize

    p = localize(recorded)
    if not p.is_absolute():
        p = ROOT / p
    return p if p.is_file() else None


def track_phrase_table(prov: Optional[dict]) -> tuple[Optional[list], str]:
    """Phrase table (``dj_select`` rows: ``bar_index``, ``time_s``, ``phrase_level``, ``section_start``) of the analysis the
    render used (``provenance.<deck>.analysis``); otherwise the track card's entries (``track_cards/<slug>.json``, same
    audio SHA-256) as a partial table of phrase starts. ``(None, why)`` when neither is available."""
    prov = prov or {}
    p = _resolve(prov.get("analysis"))
    if p is not None:
        an = _load_json_cached(p)
        if an and an.get("phrases"):
            return an["phrases"], f"analysis {prov.get('analysis')}"
    slug = prov.get("slug")
    if slug:
        card = _load_json_cached(ROOT / "track_cards" / f"{slug}.json")
        sha = ((card or {}).get("audio") or {}).get("sha256")
        if card and card.get("entries") and (not prov.get("audio_sha256") or sha == prov.get("audio_sha256")):
            return [dict(e, bar_index=int(e["bar"])) for e in card["entries"] if e.get("bar") is not None], f"track card track_cards/{slug}.json (entries only)"
    return None, "no analysis or track card for this deck"


def _cue_naturalness_cached(slug: Optional[str], t: float) -> dict:
    """``self_supervision.cue_naturalness`` read from its per-track cache only (never computed here): a feature, not a verdict."""
    if not slug:
        return {"status": "unavailable", "why": "no slug"}
    try:
        from musicmix import self_supervision as ss

        if ss._cue_doc(slug, Path(ss.CUE_DIR)) is None:
            return {"status": "unavailable", "why": f"no cache {ss.CUE_DIR}/{slug}.json"}
        r = ss.cue_naturalness(slug, float(t), variant="full")
        return r if r.get("status") == "cached" else {"status": "unavailable", "why": "no cached candidate within 60 ms"}
    except Exception as exc:  # optional feature: never breaks the QA
        return {"status": "unavailable", "why": f"{type(exc).__name__}: {exc}"[:160]}


def check_phrase(geo: dict, meta: dict, cfg: QAConfig) -> dict:
    """B's entry on a phrase start of B (.7, palestra dei difetti: B ±1 battuta passava "per caso").

    [P] B&B 2003 cap. 10 «How to match phrases»: «Cue up your incoming record so its first beat is the start of a
    phrase»; Butler 2006 p. 185. The cue of B (``cue_in_b_s``, B's source seconds) must fall on a phrase start of B's
    phrase table (the ``dj_select`` analysis the render used: ``phrase_level`` >= 4 or a section start; the same
    definition as the planner's cue candidates). Off the phrase grid = ``fail``. The cue naturalness of
    ``self_supervision`` (trained to tell a point where the song articulates from the same point moved by +-1/2 bar) is
    read from its cache as a second opinion: if it scores the actual cue at least as natural as the nearest phrase start
    (and >= 0.5), the analysis phrase grid may be off locally and the result is ``warn``. A relocation declared by the
    render (``meta['b_entry']['relocated_bars']``, the 1-2 "upbeat" bars of INGRESSO_B R-B2) is ``warn`` too: a chosen
    trade-off, reported. Without a phrase table: ``unavailable``. A's exit position is reported, not judged."""
    if cfg.phrase_check == "off":
        return _check("not_applicable", coverage="not_applicable", reason="phrase_check='off'")
    sr = geo["sr"]
    prov = meta.get("provenance") or {}
    pb = prov.get("b") or {}
    t_b = meta.get("cue_in_b_s")
    sw = meta.get("switch") or {}
    if t_b is None and sw.get("b_source_sample") is not None:
        t_b = int(sw["b_source_sample"]) / sr
    rows, src = track_phrase_table(pb)
    base = {"b_cue_s": t_b, "phrase_table": src, "rule": "B's cue on a phrase start of B (phrase_level >= 4 or section start) [P] B&B cap. 10",
            "required": bool(pb)}
    if t_b is None or not rows:
        return _check("skipped", coverage="unavailable", reason="B's cue or B's phrase table not available: phrase position not measured", **base)
    rows = sorted(rows, key=lambda r: int(r["bar_index"]))
    times = np.array([float(r["time_s"]) for r in rows])
    full_table = "entries only" not in src
    k = int(np.argmin(np.abs(times - float(t_b))))
    bar_s = float(np.median(np.diff(times))) if len(times) > 2 else None
    dist = float(t_b) - float(times[k])
    j = int(rows[k]["bar_index"])
    starts = sorted(int(r["bar_index"]) for r in rows if int(r.get("phrase_level") or 0) >= 4 or r.get("section_start"))
    base.update(b_cue_bar=j, b_cue_from_downbeat_s=dist, bar_s=bar_s)
    if full_table and bar_s and abs(dist) > 0.25 * bar_s:
        return _check("skipped", coverage="unavailable", reason=f"B's cue {dist:+.2f} s from the nearest downbeat of the phrase table (another grid?)", **base)
    if not full_table and abs(dist) > 0.25 * (bar_s or 2.0):
        return _check("skipped", coverage="unavailable", reason="B's cue not among the track card's entries (partial table): not measured", **base)
    prev = [b for b in starts if b <= j]
    nxt = [b for b in starts if b > j]
    base.update(bars_after_phrase_start=(j - prev[-1]) if prev else None, bars_to_next_phrase_start=(nxt[0] - j) if nxt else None)
    pa = prov.get("a") or {}
    if meta.get("cue_out_a_s") is not None:
        ra, srca = track_phrase_table(pa)
        if ra:
            ta = np.array([float(r["time_s"]) for r in ra])
            ka = int(np.argmin(np.abs(ta - float(meta["cue_out_a_s"]))))
            base["a_exit"] = {"bar": int(ra[ka]["bar_index"]), "phrase_level": ra[ka].get("phrase_level"), "section_start": ra[ka].get("section_start"),
                              "note": "reported, not judged"}
    if j in starts:
        return _check("pass", coverage="measured", **base)
    near = min(([prev[-1]] if prev else []) + ([nxt[0]] if nxt else []), key=lambda b: abs(b - j), default=None)
    off_bars = None if near is None else j - near
    base["off_bars"] = off_bars
    cn = _cue_naturalness_cached(pb.get("slug"), float(t_b))
    cn_start = _cue_naturalness_cached(pb.get("slug"), float(times[[int(r["bar_index"]) for r in rows].index(near)])) if near is not None else {"status": "unavailable"}
    base["cue_naturalness"] = {"entry": cn, "nearest_phrase_start": cn_start,
                               "use": "second opinion only (self_supervision, held-out artist AUC 0.87); cache only, never computed here"}
    sev = "warn" if cfg.phrase_check == "warn" else "fail"
    why = []
    if cn.get("status") == "cached" and cn_start.get("status") == "cached" and cn["p"] >= 0.5 and cn["p"] >= cn_start["p"]:
        sev = "warn"
        why.append(f"cue naturalness scores the cue ({cn['p']:.2f}) at least as natural as the phrase start ({cn_start['p']:.2f}): "
                   "the analysis phrase grid may be off here")
    reloc = (meta.get("b_entry") or {}).get("relocated_bars")
    if reloc:
        sev = "warn"
        why.append(f"relocation declared by the render ({reloc} bar(s) before B's phrase start, INGRESSO_B R-B2)")
    where = (f"{-off_bars} bar(s) before" if off_bars is not None and off_bars < 0 else f"{off_bars} bar(s) after") if off_bars is not None else "away from"
    return _check(sev, coverage="measured", problem="b_off_phrase",
                  reason=f"B enters on its bar {j}, {where} its phrase start (bar {near}): B's phrases no longer line up with A's"
                         + (f" ({'; '.join(why)})" if why else ""), **base)


# --------------------------------------------------------------- entry point


def evaluate_render(
    audio: np.ndarray,
    metadata: dict,
    *,
    sample_rate: Optional[int] = None,
    decks: Optional[dict] = None,
    vocals_a: Optional[np.ndarray] = None,
    vocals_b: Optional[np.ndarray] = None,
    vocal_thresholds: tuple = (None, None),
    words_a: Optional[Sequence[dict]] = None,
    tracker: Any = "auto",
    config: Optional[QAConfig] = None,
    grid_support: Optional[dict] = None,
    words_b: Optional[Sequence[dict]] = None,
) -> dict:
    """Esegue tutti i controlli e restituisce un dict JSON-serializzabile.

    * ``audio``: la clip (frames, ch) come scritta dal renderer;
    * ``decks``: facoltativo, ``{'a', 'b'}`` (post EQ/fader, coordinate della
      clip) ed eventualmente ``'b_stretched_raw'``/``'a_raw'`` e ``'gains'``
      (``return_decks=True`` di dj_render);
    * ``vocals_a``/``vocals_b``: stem vocali *prima* di fader/ducking, già in
      coordinate della clip (``stems_to_output``);
    * ``words_a``: parole di A ``{'start_s','end_s','text'}`` nella clip
      (``words_to_output``);
    * ``words_b`` (.6): parole di B nella clip (``words_to_output(..., deck="b")``), per ``b_entry``; meglio le parole
      raffinate di ``dj_phrase`` (Whisper incolla le parole: senza pause la regola della frase iniziata prima scatta
      sempre);
    * ``tracker``: ``"auto"`` (Beat This! se disponibile), ``None`` o callable;
    * ``grid_support``: facoltativo, ``{'a': tempo, 'b': tempo}`` con i blocchi
      ``tempo`` di dj_select (provider della griglia usata e ``agreement``):
      abilita il warn "possibile ambiguità di fase del tracker".
    """
    cfg = config or QAConfig()
    sr = int(sample_rate or metadata.get("sample_rate", 48000))
    x = _as_stereo(audio)
    geo = transition_geometry({**metadata, "sample_rate": sr}, len(x))
    trk = _get_tracker(tracker)
    phase_support = None
    if grid_support:
        phase_support = {k: grid_phase_support(grid_support.get(k)) for k in ("a", "b")}
    checks = {
        "rhythm": check_rhythm(x, geo, decks, trk, cfg, phase_support, meta=metadata),
        "double_bass": check_double_bass(decks, geo, cfg),
        "loudness_jump": check_loudness_jump(x, geo, cfg),
        "true_peak": check_true_peak(x, sr, cfg),
        "vocal_clash": check_vocal_clash(vocals_a, vocals_b, metadata, decks, geo, cfg, vocal_thresholds),
        "verse_cut": check_verse_cut(geo, decks, list(words_a) if words_a else None, vocals_a, cfg, vocal_thresholds[0], meta=metadata),
        "b_entry": check_b_entry(geo, decks, list(words_b) if words_b else None, vocals_b, cfg, meta=metadata),
        "duration": check_duration(len(x), geo, metadata, cfg),
        "phrase": check_phrase(geo, metadata, cfg),
        "click": check_click(x, geo, metadata, cfg),
        "echo_tempo": check_echo_tempo(x, geo, decks, metadata, cfg),
    }
    if cfg.dropout_check:
        checks["dropout"] = check_dropout(x, geo, metadata, cfg)
    reasons = []
    for name, c in checks.items():
        c.setdefault("coverage", _coverage_state(c))
        if c["status"] in ("fail", "warn"):
            msg = c.get("reason") or "; ".join(c.get("problems", [])) or _default_reason(name, c)
            reasons.append({"check": name, "severity": c["status"], "message": msg})
    coverage = coverage_summary(checks, metadata, geo)
    if cfg.require_coverage and coverage["missing_required"] and not any(r["severity"] == "fail" for r in reasons) and not reasons:
        reasons.append({"check": "coverage", "severity": "warn",
                        "message": "required check(s) not measured: " + ", ".join(coverage["missing_required"]) + " (a missing check is not a pass)"})
    verdict = "fail" if any(r["severity"] == "fail" for r in reasons) else "warn" if reasons else "pass"
    tracker_info = trk.info() if hasattr(trk, "info") else (None if trk is None else {"model": repr(trk)})
    return _jsonable({
        "schema_version": DJ_QA_VERSION,
        "verdict": verdict,
        "passed": verdict != "fail",
        "reasons": reasons,
        "coverage": coverage,
        "checks": checks,
        "geometry": {k: (v / sr if k in ("start", "switch", "end", "n") else v) for k, v in geo.items()},
        "geometry_units": "seconds in the clip",
        "inputs": {"decks": bool(decks), "vocals": vocals_a is not None and vocals_b is not None, "words_a": bool(words_a),
                   "words_b": bool(words_b), "tracker": tracker_info,
                   "grid_support": phase_support, "b_projection": b_projection_info(metadata)},
        "thresholds": asdict(cfg),
        "threshold_origin": THRESHOLD_ORIGIN,
        "note": "technical integrity only; a pass is not a judgement of musical quality",
    })


def _coverage_state(c: dict) -> str:
    st = c.get("status")
    if st == "skipped":
        return "unavailable"
    if st == "not_applicable":
        return "not_applicable"
    return "measured"


def coverage_summary(checks: dict, meta: dict, geo: dict) -> dict:
    """Stato di copertura di ogni controllo e controlli richiesti non misurati (P0.6, [H] in ``REQUIRED_*``)."""
    overlap = bool((meta.get("stretch") or {}).get("applied")) and not geo["is_cut"]
    required = list(REQUIRED_ALWAYS) + (list(REQUIRED_WITH_OVERLAP) if overlap else [])
    if (checks.get("phrase") or {}).get("required"):  # .7: real renders (B's provenance declared) must measure B's phrase position
        required.append("phrase")
    states = {k: v.get("coverage", _coverage_state(v)) for k, v in checks.items()}
    missing = [k for k in required if states.get(k, "unavailable") == "unavailable"]
    return {"checks": states, "required": required, "missing_required": missing, "complete": not missing,
            "two_tracks_overlap": overlap, "states": list(COVERAGE_STATES),
            "rule": "[H] a required check that is unavailable turns a pass into warn; not_applicable and intended_gap are covered"}


def _default_reason(name: str, c: dict) -> str:
    if name == "true_peak":
        return f"true peak {c.get('true_peak_dbtp')} dBTP > {c.get('threshold_dbtp')} dBTP or {c.get('full_scale_samples')} full-scale samples"
    if name == "double_bass":
        return f"both decks' low band (< {c['threshold']['below_hz']:.0f} Hz) within {c['threshold']['rel_db']:.0f} dB of their peak for {1000*c['longest_s']:.0f} ms (> {1000*c['threshold']['max_s']:.0f} ms)"
    if name == "loudness_jump":
        return f"short-term loudness jump {c['signed_jump_db']:+.1f} dB ({c['worst_window']})"
    return name


# ---------------------------------------------------- coordinate conversion


def a_offset_samples(meta: dict) -> int:
    """Campione sorgente di A = campione clip + offset (A non viene stretchato)."""
    ov = meta["overlap"]
    return int(ov["a_source_start_sample"]) - int(ov["output_start_sample"])


def b_source_position(meta: dict, n: int) -> np.ndarray:
    """Posizione (campioni, float) nella sorgente B per ogni campione della clip.

    cut / nessuno stretch: offset costante, ancorato allo switch (campione della
    clip in cui suona ``switch.b_source_sample``). Negli effetti di
    ``dj_effects`` (echo_out con ``entry_bars`` > 0, filter_out) la regione
    ``overlap`` inizia prima dell'ingresso di B mentre
    ``overlap.b_source_start_sample`` è il campione d'ingresso di B: ancorarsi
    a ``overlap.output_start_sample`` spostava lo stem di B di tutto l'effetto
    (correzione 20/10/2026). Per ``cut`` di dj_render le due ancore coincidono.
    Blend: se il render dichiara ``stretch.time_map`` (dj_render dal 21/10/2026,
    dj_creative con B stirato) si usa la mappa a tratti per battuta; altrimenti
    (metadati precedenti) la retta per lo switch con il rapporto medio
    ``tempo_ratio_b``, approssimazione dichiarata in ``b_projection_info``.
    """
    ov, sw = meta["overlap"], meta.get("switch", {})
    st = meta.get("stretch", {})
    t = np.arange(n, dtype=float)
    tm = _time_map(meta)
    if tm is not None:  # P0.6: the map the renderer actually used (per-bar nodes), not a straight line
        sr = float(meta.get("sample_rate", 48000))
        return _map_inv(tm, t / sr) * sr
    if not st.get("applied"):
        if sw.get("b_source_sample") is not None and sw.get("output_sample") is not None:
            return t + (int(sw["b_source_sample"]) - int(sw["output_sample"]))
        return t + (int(ov["b_source_start_sample"]) - int(ov["output_start_sample"]))
    rate = float(st.get("tempo_ratio_b", 1.0))
    return float(sw["b_source_sample"]) + (t - float(sw["output_sample"])) * rate


def _time_map(meta: dict) -> Optional[dict]:
    st = meta.get("stretch") or {}
    tm = st.get("time_map")
    if not st.get("applied") or not tm:
        return None
    src, dst = np.asarray(tm["b_source_s"], float), np.asarray(tm["output_s"], float)
    if len(src) < 2 or len(src) != len(dst) or np.any(np.diff(src) <= 0) or np.any(np.diff(dst) <= 0):
        return None
    return {"src": src, "dst": dst, "sb": float(tm["slope_before"]), "sa": float(tm["slope_after"]), "kind": tm.get("kind")}


def _map_fwd(tm: dict, s) -> np.ndarray:
    """B source seconds -> clip seconds (linear extrapolation outside the nodes)."""
    s = np.asarray(s, dtype=float)
    out = np.interp(s, tm["src"], tm["dst"])
    out = np.where(s < tm["src"][0], tm["dst"][0] + (s - tm["src"][0]) * tm["sb"], out)
    return np.where(s > tm["src"][-1], tm["dst"][-1] + (s - tm["src"][-1]) * tm["sa"], out)


def _map_inv(tm: dict, d) -> np.ndarray:
    """Clip seconds -> B source seconds."""
    d = np.asarray(d, dtype=float)
    out = np.interp(d, tm["dst"], tm["src"])
    out = np.where(d < tm["dst"][0], tm["src"][0] + (d - tm["dst"][0]) / tm["sb"], out)
    return np.where(d > tm["dst"][-1], tm["src"][-1] + (d - tm["dst"][-1]) / tm["sa"], out)


def b_projection_info(meta: dict) -> dict:
    """Come le coordinate sorgente di B sono proiettate nella clip (dichiarato nel report del QA)."""
    st = meta.get("stretch") or {}
    tm = _time_map(meta)
    if tm is not None:
        return {"method": "time_map", "kind": tm["kind"], "nodes": int(len(tm["src"])),
                "note": "B source -> clip through the renderer's time map (piecewise per bar when keyframes were used)"}
    if st.get("applied"):
        return {"method": "linear_approximation", "tempo_ratio_b": st.get("tempo_ratio_b"),
                "note": "no stretch.time_map in the metadata (render before 21/10/2026): straight line through the switch at the mean ratio"}
    return {"method": "offset", "note": "no stretch: constant offset anchored on the switch"}


def b_output_position(meta: dict, src_samples) -> np.ndarray:
    """Campioni sorgente di B -> campioni della clip (inverso di ``b_source_position``)."""
    src = np.asarray(src_samples, dtype=float)
    sr = float(meta.get("sample_rate", 48000))
    tm = _time_map(meta)
    if tm is not None:
        return _map_fwd(tm, src / sr) * sr
    ov, sw = meta["overlap"], meta.get("switch", {})
    st = meta.get("stretch", {})
    if not st.get("applied"):
        if sw.get("b_source_sample") is not None and sw.get("output_sample") is not None:
            return src - (int(sw["b_source_sample"]) - int(sw["output_sample"]))
        return src - (int(ov["b_source_start_sample"]) - int(ov["output_start_sample"]))
    rate = float(st.get("tempo_ratio_b", 1.0))
    return float(sw["output_sample"]) + (src - float(sw["b_source_sample"])) / rate


def stems_to_output(meta: dict, n: int, stem_a: Optional[np.ndarray] = None, stem_b: Optional[np.ndarray] = None) -> tuple:
    """Stem vocali sorgente -> coordinate della clip (prima di fader/ducking)."""
    out = []
    if stem_a is not None:
        a = _as_stereo(stem_a)
        off = a_offset_samples(meta)
        seg = np.zeros((n, 2))
        lo, hi = max(0, off), min(len(a), off + n)
        if hi > lo:
            seg[lo - off : hi - off] = a[lo:hi]
        out.append(seg)
    else:
        out.append(None)
    if stem_b is not None:
        b = _as_stereo(stem_b)
        pos = b_source_position(meta, n)
        idx = np.arange(len(b), dtype=float)
        out.append(np.stack([np.interp(pos, idx, b[:, c], left=0.0, right=0.0) for c in range(2)], 1))
    else:
        out.append(None)
    return tuple(out)


def stem_threshold(stem: np.ndarray, sr: int) -> float:
    """Soglia di attività calcolata sull'intero stem sorgente (non solo la clip)."""
    return vocal_activity(stem, sr)[1]


def words_to_output(meta: dict, words: Sequence[dict], n: int, deck: str = "a") -> list[dict]:
    """Parole ASR (``start_sample``/``end_sample`` sorgente) -> secondi della clip.

    ``deck="a"``: offset costante (A non è stirato). ``deck="b"``: attraverso il time-map del render
    (``b_output_position``), così le parole di B stirato cadono dove suonano nel PCM.
    """
    sr = int(meta.get("sample_rate", 48000))
    out = []
    if deck == "b":
        if not words:
            return out
        pos = b_output_position(meta, np.array([[int(w["start_sample"]), int(w["end_sample"])] for w in words], dtype=float))
        for w, (s, e) in zip(words, pos):
            if e > 0 and s < n:
                out.append({"start_s": float(s) / sr, "end_s": float(e) / sr, "text": w.get("text", w.get("word", "")),
                            "confidence": w.get("confidence"), "projection": b_projection_info(meta)["method"]})
        return out
    off = a_offset_samples(meta)
    for w in words:
        s, e = int(w["start_sample"]) - off, int(w["end_sample"]) - off
        if e > 0 and s < n:
            out.append({"start_s": s / sr, "end_s": e / sr, "text": w.get("text", w.get("word", "")), "confidence": w.get("confidence")})
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    import soundfile as sf

    p = argparse.ArgumentParser(description="DJ render QA (clip + metadata JSON; mix-only mode)")
    p.add_argument("wav")
    p.add_argument("json")
    p.add_argument("--no-beat-this", action="store_true")
    args = p.parse_args(argv)
    audio, sr = sf.read(args.wav, dtype="float64", always_2d=True)
    meta = json.loads(Path(args.json).read_text(encoding="utf-8"))
    rep = evaluate_render(audio, meta, sample_rate=sr, tracker=None if args.no_beat_this else "auto")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
