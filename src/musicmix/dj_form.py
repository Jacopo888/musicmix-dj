"""Categorie formali di Butler (2006) come feature calcolabili sull'audio, per battuta.

Butler, *Unlocking the Groove* (Indiana UP 2006) descrive la forma dei brani da
DJ in termini di **strati** (layer) che entrano ed escono, della **cassa** come
strato decisivo e di cinque funzioni formali (cap. 6, pp. 220–224): *intro*
(tessitura sottile, spesso solo cassa), *buildup* (strati aggiunti uno alla
volta), *core* ("the song", densità tipica e stabile), *breakdown* (calo
improvviso, cassa tolta, per almeno 4 battute: cap. 2 pp. 91–92), *outro*
(specchio dell'intro, più corto). Il ritorno della cassa dopo un'assenza è il
"drop" (*dropping the beat*, pp. 246–248); le rimozioni brevi (1–2 battiti o
1–2 battute) cadono subito prima di un downbeat ipermetrico (cap. 5 pp. 188–190).
I DJ evitano di sovrapporre due superfici ritmiche troppo attive ("busy",
p. 245) e due voci predominanti (p. 244). [Pagine di Butler verificate il 21/10/2026 sul
testo per pagina: diverse citazioni erano una pagina prima; corrette qui e in dj_set.]

Questo modulo traduce quelle categorie in misure per battuta, senza modelli
addestrati:

* :func:`bar_layers` — livelli per battuta di cinque strati: ``kick`` (flusso
  positivo dell'ampiezza 40–120 Hz sui battiti), ``bass`` (mediana dell'energia
  40–250 Hz nella battuta), ``mid`` (250–4000 Hz), ``high`` (4–16 kHz),
  ``vocals`` (energia dello stem vocale), più ``onsets_per_beat`` (attacchi
  dell'accompagnamento per battito). Con gli stem Demucs a 4 tracce (``drums``,
  ``bass``, ``other``) usa quelli; senza, l'accompagnamento = mix − voce.
  :func:`bar_layers_from_allin1_spec` fa lo stesso dagli spettrogrammi per stem
  che All-In-One salva comunque (``.cache/dj_allin1*/spec/<slug>.npy``), senza
  rifare la separazione. Nell'hip-hop la "cassa" è spesso un 808 che Demucs mette
  nel basso: la cassa qui è l'attacco basso di drums **+** bass [H].
* :func:`layer_activity` — strato attivo per battuta (soglie relative al
  brano, valori [H]). Cassa (dj_form.3): media dei picchi sui battiti >= ``kick_on``
  **oppure** attacco basso più forte della battuta >= ``kick_peak_on`` (808 lunghi e
  sincopati, uno o due colpi per battuta fuori dal battito; ``kick_peak_only`` dice in
  quali battute decide solo questa seconda regola).
* :func:`form_segments` — etichette per battuta ``intro``/``build``/``core``/
  ``groove``/``breakdown``/``outro`` (``unknown`` se il brano non ha un core
  misurabile: ``core_status = "no_core"``), eventi ``drops`` e ``kick_dropouts``.
* :func:`overlap_profile` — riassunto di due finestre di battute (A in uscita,
  B in ingresso) per l'albero/il costo di :mod:`musicmix.dj_set` e per il
  giudice: strati sommati, battute "busy" in entrambi, voci in entrambi.

Le corrispondenze sono **[H] ipotesi di progetto**: Butler descrive la techno e
la trance (pochi brani, ascolto dell'autore, interviste); le soglie qui non sono
validate su ascolti. Nell'hip-hop/pop con voce la forma è strofa/ritornello
(allin1): queste feature la affiancano, non la sostituiscono.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

FORM_VERSION = "musicmix.dj_form.3"  # .2 (21/10/2026): senza core le battute sono "unknown", non "outro"; .3 (21/10/2026): cassa anche
# dal picco d'attacco basso nella battuta (808 lunghi e sincopati dell'hip-hop, ``kick_peak_on``; None = regola di .2)

#: Bande (Hz) degli strati stimati dal mix/accompagnamento. [H] valori convenzionali
#: (cassa 40–120 Hz; basso fino a 250 Hz: BT in Webber 2008 p. 37 'shelving 150 Hz and lower out of
#: everything other than your kick drum and bass line').
BANDS: dict[str, tuple[float, float]] = {
    "kick": (40.0, 120.0),
    "bass": (40.0, 250.0),
    "mid": (250.0, 4000.0),
    "high": (4000.0, 16000.0),
}

#: Parametri con origine.
PARAMS: dict[str, tuple[Any, str]] = {
    "n_fft": (4096, "[H] ~85 ms a 48 kHz: risoluzione sufficiente per 40–120 Hz"),
    "hop": (1024, "[H] ~21 ms"),
    "kick_window_s": ((-0.04, 0.06), "[H] finestra attorno a ogni battito per il picco di flusso basso"),
    "kick_on": (0.30, "[H] cassa presente in una battuta se il picco medio sui battiti >= 0,30 del 95° percentile"),
    "kick_peak_on": (0.50, "[H] (dj_form.3) cassa presente anche se il massimo dell'attacco basso (flusso d'ampiezza 30/40-120 Hz di "
                           "drums + bass, in tutta la battuta tranne l'ultimo battito, non solo sui battiti) è >= 0,50 dello stesso "
                           "riferimento (95° percentile dei picchi sui battiti). Finestra: dal downbeat - 40 ms all'ultimo battito - "
                           "40 ms: un colpo isolato sull'ultimo battito è un'anacrusi verso la battuta dopo ([A] Butler 2006 p. 189: "
                           "le variazioni 'almost always occur just before a metrical or hypermetrical downbeat'). Motivo: nell'hip-hop la 'cassa' è spesso un 808 lungo e sincopato "
                           "(uno o due colpi per battuta, spesso fuori dal battito, con coda tonale e glissando) che Demucs mette "
                           "nello stem bass: la media sui quattro battiti resta sotto 0,30 anche con l'808 udibile (gods_plan: "
                           "media 0,2-0,3, massimo 1,0-1,2 nelle battute con l'808). Il glissando non crea attacchi perché la banda "
                           "somma l'ampiezza di tutti i bin. Limite: un attacco di basso tenuto (nota nuova) conta come cassa; "
                           "Butler 2006 descrive la cassa della techno, non l'808 (estensione nostra). None = regola di dj_form.2"),
    "layer_on_db": (-15.0, "[H] strato attivo se il livello della battuta è entro 15 dB dal suo 90° percentile nel brano"),
    "layer_floor_db": (-60.0, "[H] sotto -60 dBFS lo strato è spento comunque"),
    "breakdown_min_bars": (4, "[A] Butler 2006 pp. 91-92 [corretto: era p. 91]: breakdown 'typically lasting for at least four measures'"),
    "breakdown_drop_layers": (2, "[H] calo di almeno 2 strati rispetto al livello del core (Butler p. 224 [corretto: era p. 223]: 'sudden, dramatic drop "
                                 "to a thin texture')"),
    "core_percentile": (75.0, "[H] livello di core = 75° percentile del numero di strati attivi"),
    "onset_min_gap_s": (0.07, "[H] attacchi distinti se distano almeno 70 ms (una semicroma a ~210 BPM)"),
    "smooth_bars": (3, "[H] mediana mobile di 3 battute sul numero di strati e maggioranza sulla cassa prima di etichettare: Butler "
                       "descrive sezioni di almeno 4 battute (p. 189 [corretto: era p. 188]: cambi che durano 'at a minimum, four measures')"),
    "busy_onsets_per_beat": (6.0, "[H] battuta 'busy' se l'accompagnamento ha >= 6 attacchi per battito (oltre i sedicesimi: "
                                  "rulli, hi-hat raddoppiati); Butler p. 245 [corretto: era p. 244] ('if it's too busy, I don't think it sounds good') non dà "
                                  "numeri. Verifica 21/10/2026 su 4 brani: con 3 quasi tutte le battute erano 'busy' (mediana 4–6,5 "
                                  "attacchi/battito) e il conteggio dipende dalla rappresentazione (spettrogramma allin1 vs stem: "
                                  "correlazione 0,0–0,5): misura debole, usarla poco"),
}

LAYER_NAMES = ("kick", "bass", "mid", "high", "vocals")


def _p(name: str, overrides: Optional[Mapping[str, Any]] = None) -> Any:
    if overrides and name in overrides:
        return overrides[name]
    return PARAMS[name][0]


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _stft_power(x: np.ndarray, sr: int, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Potenza |STFT|^2 (frame x bin), tempi centrali dei frame e frequenze dei bin."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    win = np.hanning(n_fft)
    spec = np.fft.rfft(x[idx] * win, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2) / (win.sum() ** 2)
    times = (np.arange(n) * hop + n_fft / 2) / sr
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return power, times, freqs


def _band(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (freqs >= lo) & (freqs < hi)
    return power[:, sel].sum(axis=1) if sel.any() else np.zeros(power.shape[0])


def _flux(energy: np.ndarray) -> np.ndarray:
    """Flusso positivo del logaritmo dell'energia di banda (attacchi); pavimento a -50 dB dal 99° percentile
    della banda, così il silenzio digitale non produce attacchi enormi."""
    floor = 1e-5 * float(np.percentile(energy, 99)) if len(energy) else 0.0
    le = np.log10(energy + floor + 1e-12)
    d = np.diff(le, prepend=le[:1])
    return np.maximum(d, 0.0)


def _flux_amp(energy: np.ndarray) -> np.ndarray:
    """Flusso positivo dell'ampiezza di banda (sqrt dell'energia): un suono tenuto (basso) non conta,
    un colpo che si aggiunge sì, anche sopra un basso continuo."""
    a = np.sqrt(np.maximum(energy, 0.0))
    return np.maximum(np.diff(a, prepend=a[:1]), 0.0)


def _onset_strength(powers: Sequence[np.ndarray], freqs: np.ndarray) -> list[np.ndarray]:
    """Flussi (log) delle bande basse/medie/alte di ogni sorgente: gli attacchi si contano per banda e poi si
    uniscono, così un hi-hat in sedicesimi conta quanto una cassa [H]."""
    return [_flux(_band(P, freqs, lo, hi)) for P in powers
            for lo, hi in ((40.0, 250.0), (250.0, 4000.0), (4000.0, 16000.0))]


def _bar_bounds(downbeats_s: Sequence[float], duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    db = np.asarray(downbeats_s, dtype=float)
    ends = np.append(db[1:], min(duration_s, db[-1] + (np.median(np.diff(db)) if len(db) > 1 else 2.0)))
    return db, ends


def _db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-12))


def bar_layers(audio: np.ndarray, sr: int, downbeats_s: Sequence[float], beats_s: Sequence[float], *,
               vocals: Optional[np.ndarray] = None, stems: Optional[Mapping[str, np.ndarray]] = None,
               params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Livelli per battuta degli strati di Butler (vedi docstring del modulo).

    ``audio``: mix (campioni x canali o mono); ``vocals``: stem vocale allineato
    (stessa frequenza di campionamento); ``stems``: facoltativo, ``{"drums",
    "bass", "other"}`` Demucs a 4 tracce. Restituisce per ogni strato il livello
    in dB per battuta (``*_db``), il punteggio di cassa (``kick``, 0–~1) e
    ``onsets_per_beat``, più ``source`` ("stems" o "bands").
    """
    n_fft, hop = int(_p("n_fft", params)), int(_p("hop", params))
    mix = _mono(audio)
    duration = len(mix) / sr
    db, ends = _bar_bounds(downbeats_s, duration)
    beats = np.asarray(beats_s, dtype=float)
    voc = _mono(vocals) if vocals is not None else None
    if voc is not None:
        m = min(len(voc), len(mix))
        voc = voc[:m]
    use_stems = stems is not None and all(k in stems for k in ("drums", "bass", "other"))
    if use_stems:
        drums, bass_s, other = (_mono(stems[k]) for k in ("drums", "bass", "other"))
        P_d, t, f = _stft_power(drums, sr, n_fft, hop)
        P_b, _, _ = _stft_power(bass_s, sr, n_fft, hop)
        P_o, _, _ = _stft_power(other, sr, n_fft, hop)
        kick_e = _band(P_d, f, *BANDS["kick"]) + _band(P_b, f, *BANDS["kick"])  # 808 incluso [H]
        onset_src = _onset_strength([P_d, P_o], f)
        bass_e = _band(P_b, f, 20.0, 400.0)
        mid_e = _band(P_o, f, *BANDS["mid"])
        high_e = _band(P_d, f, *BANDS["high"]) + _band(P_o, f, *BANDS["high"])
    else:
        acc = mix.copy()
        if voc is not None:
            acc[: len(voc)] -= voc
        P, t, f = _stft_power(acc, sr, n_fft, hop)
        kick_e = _band(P, f, *BANDS["kick"])
        onset_src = _onset_strength([P], f)
        bass_e = _band(P, f, *BANDS["bass"])
        mid_e = _band(P, f, *BANDS["mid"])
        high_e = _band(P, f, *BANDS["high"])
    if voc is not None:
        # energia della voce per frame (stessa griglia temporale degli altri strati)
        nfr = len(t)
        voc_e = np.zeros(nfr)
        for i in range(nfr):
            c = int(t[i] * sr)
            seg = voc[max(0, c - hop // 2): c + hop // 2]
            voc_e[i] = float(np.mean(seg ** 2)) if len(seg) else 0.0
    else:
        voc_e = None
    return _aggregate_bars(t, kick_e, bass_e, mid_e, high_e, voc_e, onset_src, downbeats_s, beats_s,
                           hop_s=hop / sr, source="stems" if use_stems else "bands", params=params,
                           duration_s=duration, extra_params={"n_fft": n_fft, "hop": hop})


def _aggregate_bars(t: np.ndarray, kick_e: np.ndarray, bass_e: np.ndarray, mid_e: np.ndarray, high_e: np.ndarray,
                    voc_e: Optional[np.ndarray], onset_src: Sequence[np.ndarray], downbeats_s: Sequence[float],
                    beats_s: Sequence[float], *, hop_s: float, source: str, params: Optional[Mapping[str, Any]] = None,
                    duration_s: Optional[float] = None, extra_params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Da energie per frame (tempi ``t``) a livelli per battuta (comune a mix, stem e spettrogrammi allin1)."""
    beats = np.asarray(beats_s, dtype=float)
    db, ends = _bar_bounds(downbeats_s, float(duration_s if duration_s is not None else (t[-1] + hop_s if len(t) else 0.0)))
    kick_flux = _flux_amp(kick_e)
    w0, w1 = _p("kick_window_s", params)
    beat_peak = np.zeros(len(beats))
    for i, b in enumerate(beats):
        sel = (t >= b + w0) & (t <= b + w1)
        beat_peak[i] = float(kick_flux[sel].max()) if sel.any() else 0.0
    ref = float(np.percentile(beat_peak, 95)) if len(beat_peak) else 1.0
    beat_kick = beat_peak / ref if ref > 0 else beat_peak
    # onset density: one onset strength = sum of the band fluxes, each divided by its standard deviation;
    # peaks above a 1 s moving median + 1, at least onset_min_gap_s apart [H] (cf. librosa/madmom peak picking)
    if onset_src:
        comb = np.zeros(len(t))
        for fl in onset_src:
            sd = float(fl.std())
            comb += fl / sd if sd > 0 else fl
    else:
        comb = np.zeros(len(t))
    gap = max(1, int(round(float(_p("onset_min_gap_s", params)) / hop_s)))
    half = max(1, int(round(0.5 / hop_s)))
    is_peak = np.zeros(len(t), dtype=bool)
    if len(comb) > 2:
        pad = np.pad(comb, half, mode="edge")
        # moving median via strided windows (chunked to bound memory)
        med = np.empty(len(comb))
        for a0 in range(0, len(comb), 4096):
            a1 = min(len(comb), a0 + 4096)
            win = np.lib.stride_tricks.sliding_window_view(pad[a0:a1 + 2 * half], 2 * half + 1)
            med[a0:a1] = np.median(win, axis=1)
        cand = np.where((comb[1:-1] > comb[:-2]) & (comb[1:-1] >= comb[2:]) & (comb[1:-1] > med[1:-1] + 1.0))[0] + 1
        last = -gap
        for i in cand:
            if i - last >= gap:
                is_peak[i] = True
                last = i
    n = len(db)
    out: dict[str, list[float]] = {k: [] for k in ("kick", "kick_peak", "kick_sustain", "bass_db", "mid_db", "high_db", "vocals_db",
                                                   "onsets_per_beat")}
    for j in range(n):
        a, e = db[j], ends[j]
        fs = (t >= a) & (t < e)
        bs = (beats >= a - 1e-3) & (beats < e - 1e-3)
        out["kick"].append(float(beat_kick[bs].mean()) if bs.any() else 0.0)
        # dj_form.3: strongest low attack in the bar (808 off the beat), same reference as the beat peaks [H]. Window: from
        # the downbeat (- 40 ms, as kick_window_s) to the bar's last beat (- 40 ms): a lone hit on the last beat is an
        # anacrusis into the next bar ([A] Butler 2006 p. 189: variations 'almost always occur just before a metrical or
        # hypermetrical downbeat'), and an attack a few ms before the next downbeat belongs to that downbeat.
        bb = beats[bs]
        p0 = a + w0
        p1 = (float(bb[-1]) + w0) if len(bb) >= 2 else e + w0
        fp = (t >= p0) & (t < p1)
        out["kick_peak"].append(float(kick_flux[fp].max() / ref) if (fp.any() and ref > 0) else 0.0)
        # diagnostic only: median / max low-band amplitude in the bar (long 808 tail ~ high, short kick ~ low)
        if fs.any():
            amp = np.sqrt(np.maximum(kick_e[fs], 0.0))
            mx = float(amp.max())
            out["kick_sustain"].append(float(np.median(amp) / mx) if mx > 0 else 0.0)
        else:
            out["kick_sustain"].append(0.0)
        # basso: mediana dei frame (il colpo di cassa è breve, il basso è tenuto) [H]
        out["bass_db"].append(float(_db(np.median(bass_e[fs]))) if fs.any() else -120.0)
        for name, en in (("mid_db", mid_e), ("high_db", high_e)):
            out[name].append(float(_db(en[fs].mean())) if fs.any() else -120.0)
        nb = max(1, int(bs.sum()))
        out["onsets_per_beat"].append(float(is_peak[fs].sum()) / nb if fs.any() else 0.0)
        out["vocals_db"].append(float(_db(voc_e[fs].mean())) if (voc_e is not None and fs.any()) else -120.0)
    return {"version": FORM_VERSION, "source": source,
            "n_bars": n, "downbeats_s": [float(x) for x in db],
            **{k: [round(v, 4) for v in vals] for k, vals in out.items()},
            "params": {**dict(extra_params or {}), "kick_window_s": _p("kick_window_s", params),
                       "onset_min_gap_s": _p("onset_min_gap_s", params)}}


#: Frequenze centrali (Hz) degli 81 bin degli spettrogrammi per stem di All-In-One (allin1.spectrogram:
#: madmom FilteredSpectrogram, frame 2048 a 44,1 kHz, 12 bande/ottava, 30–17000 Hz, log(1 + |X|), 100 fps;
#: stem nell'ordine bass, drums, other, vocals). Calcolate con madmom.audio.filters.LogarithmicFilterbank.
ALLIN1_SPEC_CENTERS_HZ = (
    43.1, 64.6, 86.1, 107.7, 129.2, 150.7, 172.3, 193.8, 215.3, 236.9, 258.4, 279.9, 301.5, 323.0, 344.5, 366.1, 387.6,
    409.1, 430.7, 452.2, 495.3, 516.8, 538.3, 581.4, 624.5, 646.0, 689.1, 732.1, 775.2, 839.8, 882.9, 925.9, 990.5,
    1055.1, 1098.2, 1184.3, 1248.9, 1313.5, 1399.7, 1485.8, 1571.9, 1658.1, 1765.7, 1873.4, 1981.1, 2088.7, 2217.9,
    2347.1, 2497.9, 2627.1, 2799.3, 2950.0, 3143.8, 3316.1, 3509.9, 3725.2, 3940.6, 4177.4, 4435.8, 4694.2, 4974.2,
    5275.6, 5577.1, 5921.6, 6266.2, 6653.8, 7041.4, 7450.5, 7902.7, 8376.4, 8871.7, 9388.5, 9948.3, 10551.3, 11175.7,
    11843.3, 12553.9, 13286.0, 14082.7, 14922.5, 15805.4)
ALLIN1_STEMS = ("bass", "drums", "other", "vocals")
ALLIN1_FPS = 100.0


def bar_layers_from_allin1_spec(spec: np.ndarray, downbeats_s: Sequence[float], beats_s: Sequence[float], *,
                                fps: float = ALLIN1_FPS, centers_hz: Sequence[float] = ALLIN1_SPEC_CENTERS_HZ,
                                params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Come :func:`bar_layers`, ma dagli spettrogrammi per stem (Demucs htdemucs) già calcolati da All-In-One.

    ``spec``: array (4, frame, 81) in ``log(1 + |X|)``; stem ``bass, drums, other, vocals``. Evita di
    rifare la separazione: All-In-One la esegue comunque per le sezioni. Cassa = bin <= 120 Hz dello stem
    drums; basso = stem bass; medi = other 250–4000 Hz; alti = drums + other >= 4 kHz; voce = stem vocals.
    """
    spec = np.asarray(spec, dtype=np.float64)
    if spec.ndim != 3 or spec.shape[0] != 4 or spec.shape[2] != len(centers_hz):
        raise ValueError(f"spettrogramma allin1 atteso (4, frame, {len(centers_hz)}), trovato {spec.shape}")
    mag = np.expm1(np.maximum(spec, 0.0))
    power = mag ** 2
    c = np.asarray(centers_hz, dtype=float)
    P = dict(zip(ALLIN1_STEMS, power))

    def band(name: str, lo: float, hi: float) -> np.ndarray:
        sel = (c >= lo) & (c < hi)
        return P[name][:, sel].sum(axis=1) if sel.any() else np.zeros(power.shape[1])

    t = np.arange(power.shape[1]) / fps
    kick_e = band("drums", 30.0, 120.0) + band("bass", 30.0, 120.0)  # 808 incluso [H]
    bass_e = band("bass", 30.0, 400.0)
    mid_e = band("other", *BANDS["mid"])
    high_e = band("drums", BANDS["high"][0], 20000.0) + band("other", BANDS["high"][0], 20000.0)
    voc_e = P["vocals"].sum(axis=1)
    onset_src = [_flux(band(stem, lo, hi)) for stem in ("drums", "other")
                 for lo, hi in ((30.0, 250.0), (250.0, 4000.0), (4000.0, 20000.0))]
    return _aggregate_bars(t, kick_e, bass_e, mid_e, high_e, voc_e, onset_src, downbeats_s, beats_s,
                           hop_s=1.0 / fps, source="allin1_spec", params=params)


def layer_activity(layers: Mapping[str, Any], params: Optional[Mapping[str, Any]] = None) -> dict[str, list[bool]]:
    """Strato attivo per battuta: cassa sopra ``kick_on``; gli altri entro ``layer_on_db`` dal 90° percentile."""
    on_db, floor = float(_p("layer_on_db", params)), float(_p("layer_floor_db", params))
    kick_on, peak_on = float(_p("kick_on", params)), _p("kick_peak_on", params)
    peaks = layers.get("kick_peak")
    by_mean = [bool(k >= kick_on) for k in layers["kick"]]
    if peak_on is not None and peaks is not None and len(peaks) == len(by_mean):
        by_peak = [bool(x >= float(peak_on)) for x in peaks]
    else:  # dj_form.2 rule (or layers cached before .3, without kick_peak)
        by_peak = [False] * len(by_mean)
    act: dict[str, list[bool]] = {"kick": [m or q for m, q in zip(by_mean, by_peak)]}
    for name in ("bass", "mid", "high", "vocals"):
        v = np.asarray(layers[f"{name}_db"], dtype=float)
        if not len(v) or np.all(v <= floor):
            act[name] = [False] * len(v)
            continue
        ref = float(np.percentile(v[v > floor], 90))
        act[name] = [bool(x > floor and x >= ref + on_db) for x in v]
    # diagnostic (not a layer): bars whose kick comes only from the low-attack peak (808 rule of dj_form.3)
    act["kick_peak_only"] = [q and not m for m, q in zip(by_mean, by_peak)]
    return act


def _runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """Intervalli [a, b) dove ``mask`` è vero."""
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


def form_segments(layers: Mapping[str, Any], params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Etichette formali per battuta e eventi, dalle categorie di Butler (cap. 2, 5, 6).

    Regole [H] (traduzione delle descrizioni di Butler pp. 91–92, 188–189, 222–225):

    1. ``n_layers`` = strati attivi (cassa, basso, medi, alti, voce); ``core_level``
       = 75° percentile (almeno 2).
    2. Il core comincia alla prima battuta con cassa e ``n_layers >= core_level`` e
       finisce all'ultima: prima c'è intro (+ build, battute dopo l'ultimo minimo di
       strati), dopo c'è outro.
    3. Fra i due, *breakdown* = run di almeno ``breakdown_min_bars`` battute senza
       cassa o con ``n_layers <= core_level - breakdown_drop_layers``; dopo un
       breakdown, le battute che risalgono verso il core sono *build*.
    4. *drop* = prima battuta con cassa dopo almeno una battuta senza cassa (a partire
       dal primo ingresso della cassa); ``kick_dropouts`` = assenze di cassa più corte
       di un breakdown seguite da ritorno (gesto anacrusico, Butler pp. 188–190).
    5. Le battute rimanenti: *core* se ``n_layers >= core_level - 1`` con cassa,
       altrimenti *groove*.
    6. **Nessun core** (nessuna battuta con cassa e ``n_layers >= core_level``):
       intro/build/outro sono definiti *rispetto al core*, quindi non si possono
       assegnare. Tutte le battute sono ``unknown``, ``core_status = "no_core"`` e
       ``no_core_reason`` dice se manca la cassa (``no_kick``: la soglia ``kick_on``
       non è mai superata, es. cassa debole; gli 808 lunghi dal .3 passano per ``kick_peak_on``) o la tessitura piena
       (``kick_without_full_texture``). Drop, dropout e battute "busy" restano
       misurati. Prima della versione .2 questo caso dava otto battute ``outro``
       (controesempio dell'audit 21/10/2026, ``tools/book_pipeline_probes.py``):
       non è una conclusione di Butler ma un limite del nostro algoritmo [H].
    """
    act = layer_activity(layers, params)
    n = len(layers["kick"])
    if n == 0:
        return {"version": FORM_VERSION, "labels": [], "core_status": "no_core", "no_core_reason": "no_bars",
                "n_layers": [], "core_level": 0, "drops": [],
                "kick_dropouts": [], "breakdowns": [], "busy": [], "active": act}
    nl_raw = np.array([sum(int(act[k][i]) for k in LAYER_NAMES) for i in range(n)])
    kick_raw = np.array(act["kick"], dtype=bool)
    w = int(_p("smooth_bars", params))
    if w > 1 and n >= w:
        h = w // 2
        pad_n = np.pad(nl_raw, h, mode="edge")
        pad_k = np.pad(kick_raw.astype(int), h, mode="edge")
        nl = np.array([int(np.median(pad_n[i:i + w])) for i in range(n)])
        kick = np.array([pad_k[i:i + w].sum() * 2 > w for i in range(n)])
    else:
        nl, kick = nl_raw, kick_raw
    core_level = max(2, int(round(float(np.percentile(nl, float(_p("core_percentile", params)))))))
    full = kick & (nl >= core_level)
    labels = ["groove"] * n
    if not full.any():
        first_core, last_core = n, -1
        labels = ["unknown"] * n  # no core: intro/build/outro are undefined (rule 6), never a silent "outro"
    else:
        first_core = int(np.argmax(full))
        last_core = int(n - 1 - np.argmax(full[::-1]))
    # intro / build before the core
    if 0 < first_core < n:
        pre = nl[:first_core]
        s = int(np.where(pre == pre.min())[0][-1])
        for i in range(first_core):
            labels[i] = "intro" if i <= s else "build"
    # outro after the core (only if there is a core)
    if last_core >= 0:
        for i in range(last_core + 1, n):
            labels[i] = "outro"
    bmin, bdrop = int(_p("breakdown_min_bars", params)), int(_p("breakdown_drop_layers", params))
    thin = (~kick) | (nl <= core_level - bdrop)
    breakdowns = []
    if last_core >= first_core:
        for a, b in _runs([bool(thin[i]) if first_core <= i <= last_core else False for i in range(n)]):
            if b - a >= bmin:
                breakdowns.append((a, b))
                for i in range(a, b):
                    labels[i] = "breakdown"
                # build back towards the core after the breakdown (layers rising)
                j = b
                while j <= last_core and j < n and nl[j] < core_level and (j == b or nl[j] >= nl[j - 1]):
                    labels[j] = "build"
                    j += 1
        for i in range(first_core, last_core + 1):
            if labels[i] == "groove":
                labels[i] = "core" if (kick[i] and nl[i] >= core_level - 1) else "groove"
    # drops and short kick dropouts (after the first kick)
    drops, dropouts = [], []
    if kick_raw.any():
        k0 = int(np.argmax(kick_raw))
        for a, b in _runs([(not kick_raw[i]) and i > k0 for i in range(n)]):
            if b < n:
                drops.append(b)
                if b - a < bmin:
                    dropouts.append({"start_bar": a, "end_bar": b, "bars": b - a})
    busy_thr = float(_p("busy_onsets_per_beat", params))
    busy = [bool(x >= busy_thr) for x in layers["onsets_per_beat"]]
    if full.any():
        core_status, no_core_reason = "found", None
    else:
        core_status = "no_core"
        no_core_reason = "no_kick" if not kick.any() else "kick_without_full_texture"
    return {"version": FORM_VERSION, "labels": labels, "core_status": core_status, "no_core_reason": no_core_reason, "n_layers": [int(x) for x in nl_raw], "n_layers_smooth": [int(x) for x in nl],
            "core_level": core_level,
            "first_core_bar": first_core if first_core < n else None, "last_core_bar": last_core if last_core >= 0 else None,
            "drops": drops, "kick_dropouts": dropouts, "breakdowns": [list(x) for x in breakdowns], "busy": busy,
            "active": act, "source": layers.get("source"),
            "params": {k: _p(k, params) for k in ("kick_on", "kick_peak_on", "layer_on_db", "breakdown_min_bars", "breakdown_drop_layers",
                                                   "core_percentile", "busy_onsets_per_beat", "smooth_bars")}}


def analyse(audio: np.ndarray, sr: int, downbeats_s: Sequence[float], beats_s: Sequence[float], *,
            vocals: Optional[np.ndarray] = None, stems: Optional[Mapping[str, np.ndarray]] = None,
            params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """:func:`bar_layers` + :func:`form_segments` in un unico documento serializzabile."""
    lay = bar_layers(audio, sr, downbeats_s, beats_s, vocals=vocals, stems=stems, params=params)
    seg = form_segments(lay, params)
    return {"version": FORM_VERSION, "layers": lay, **{k: v for k, v in seg.items() if k != "version"}}


def analyse_from_allin1_spec(spec: np.ndarray, downbeats_s: Sequence[float], beats_s: Sequence[float], *,
                             params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """:func:`bar_layers_from_allin1_spec` + :func:`form_segments`."""
    lay = bar_layers_from_allin1_spec(spec, downbeats_s, beats_s, params=params)
    seg = form_segments(lay, params)
    return {"version": FORM_VERSION, "layers": lay, **{k: v for k, v in seg.items() if k != "version"}}


def _window(seq: Sequence[Any], start: int, n: int) -> list[Any]:
    return list(seq[max(0, start): max(0, start) + n])


def overlap_profile(form_a: Mapping[str, Any], form_b: Mapping[str, Any], a_bar: int, b_bar: int, n: int) -> dict[str, Any]:
    """Profilo di sovrapposizione di ``n`` battute: A da ``a_bar``, B da ``b_bar``.

    Restituisce strati medi di A e B, la loro somma, la quota di battute "busy" in
    entrambi (Butler p. 245), di battute con voce attiva in entrambi (p. 244), le
    etichette formali alle due estremità e se B entra su un *drop* (pp. 246–248).
    """
    la, lb = _window(form_a["n_layers"], a_bar, n), _window(form_b["n_layers"], b_bar, n)
    ba, bb = _window(form_a["busy"], a_bar, n), _window(form_b["busy"], b_bar, n)
    va, vb = _window(form_a["active"]["vocals"], a_bar, n), _window(form_b["active"]["vocals"], b_bar, n)
    m = min(len(la), len(lb))
    both_busy = sum(1 for i in range(min(len(ba), len(bb))) if ba[i] and bb[i])
    both_voc = sum(1 for i in range(min(len(va), len(vb))) if va[i] and vb[i])
    lab_a, lab_b = form_a["labels"], form_b["labels"]
    return {
        "bars": m,
        "a_layers": float(np.mean(la)) if la else None, "b_layers": float(np.mean(lb)) if lb else None,
        "sum_layers": float(np.mean(np.array(la[:m]) + np.array(lb[:m]))) if m else None,
        "both_busy_share": both_busy / m if m else None,
        "both_vocals_share": both_voc / m if m else None,
        "a_form_before": lab_a[a_bar - 1] if 0 < a_bar <= len(lab_a) else None,
        "a_form_at": lab_a[a_bar] if 0 <= a_bar < len(lab_a) else None,
        "b_form_at": lab_b[b_bar] if 0 <= b_bar < len(lab_b) else None,
        "b_entry_drop": bool(b_bar in set(form_b.get("drops", []))),
    }


def form_summary(form: Mapping[str, Any]) -> dict[str, Any]:
    """Riassunto leggibile: sequenza di segmenti ``[(label, start_bar, bars)]`` e conteggi."""
    labels = form["labels"]
    segs = []
    for i, lab in enumerate(labels):
        if not segs or segs[-1][0] != lab:
            segs.append([lab, i, 1])
        else:
            segs[-1][2] += 1
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    return {"segments": [tuple(s) for s in segs], "bars_by_label": counts, "core_status": form.get("core_status"),
            "no_core_reason": form.get("no_core_reason"), "drops": list(form.get("drops", [])),
            "breakdowns": list(form.get("breakdowns", [])), "core_level": form.get("core_level"), "source": form.get("source")}
