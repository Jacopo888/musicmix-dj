"""Critico automatico di transizioni DJ (auto-feedback senza ascoltatore).

Quattro critici indipendenti, ognuno con punteggio e incertezza, più una
combinazione dichiarata *prima* della validazione sui giudizi umani:

* ``aesthetic`` (a): Audiobox Aesthetics (Meta 2025; assi CE/CU/PC/PQ) e
  SongEval (ASLP 2025; testa ufficiale su MuQ-large-msd-iter) su finestre di
  10 s: giunta contro i lati della stessa clip (A prima, B dopo) e contro le
  sorgenti originali alla stessa posizione ("restare su A" e "B al suo cue").
* ``surprisal`` (b): sorpresa (NLL per frame, somma sui 4 codebook EnCodec) di
  MusicGen-small non condizionato lungo la clip; picco alla giunta come
  z-score robusto rispetto alla sorpresa "naturale" nei tratti A-solo e
  B-solo della stessa clip, poi percentile rispetto a giunte naturali.
* ``seam`` (c): classificatore auto-supervisionato "giunta plausibile vs
  giunta con difetto noto" (sfasamento, salto di volume, tonalità, parola
  tagliata, doppio basso, fuori frase) su embedding MuQ prima/dopo/attraverso
  la giunta; logistica con validazione per brano (GroupKFold).
* ``technical`` (d): i controlli esistenti di ``dj_qa`` come feature.

Nessun critico è un oracolo del gusto: sono misure con domini di
addestramento diversi (canzoni intere generate, clip di 10 s, musica
licenziata, giunte sintetiche). La validazione sui pochi giudizi umani è
riportata con n esplicito e senza tarare pesi sugli stessi esempi.

Il modulo importa solo numpy/scipy/soundfile a livello di modulo: gli
adattatori dei modelli (torch, transformers, muq, audiobox) si caricano in
modo pigro, così lo stesso file gira nel venv principale (numpy 2) e nel venv
``venv-np1`` (numpy<2, transformers 4.42.3, MuQ).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .paths import models_dir, python_main, python_np1

DJ_CRITIC_VERSION = "musicmix.dj_critic.1"
ROOT = Path(__file__).resolve().parents[2]
MODELS = models_dir()

# ----------------------------------------------------------------- provenance
# Revisioni e hash dei pesi ufficiali usati (verificati al caricamento).
PROVENANCE = {
    "audiobox": {
        "repo": "facebook/audiobox-aesthetics",
        "revision": "9b1dd8e5df9af7216e836a98974fe3b82c56ded6",
        "file": ".cache/models/audiobox_aesthetics/model.safetensors",
        "sha256": "a5a3c2412649cc2384ec525ffd5180ce6c4778f43bed6108e0a1303de04d014e",
        "package": "audiobox_aesthetics 0.0.4",
        "license": "CC-BY-4.0 (pesi); WavLM MIT",
        "paper": "Tjandra et al. 2025, arXiv:2502.05139",
    },
    "muq": {
        "repo": "OpenMuQ/MuQ-large-msd-iter",
        "revision": "0562a57814f6f8bbd9fdea0a25921a2fce1a841a",
        "file": ".cache/song_eval_runtime/MuQ-large-msd-iter/model.safetensors",
        "sha256": "273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6",
        "license": "CC-BY-NC-4.0",
        "paper": "Zhu et al. 2025, arXiv:2501.01108",
    },
    "songeval": {
        "repo": "ASLP-lab/SongEval (ckpt/model.safetensors)",
        "revision": "848fb2ff3a2a9d64dcb20d46f07238c28abd7add",
        "file": ".cache/song_eval_runtime/weights/songeval-model.safetensors",
        "sha256": "0cb1eb8be02b96e9949ca3a2b66997c5ac7d9da3567240df4dbfc2048eb98efd",
        "license": "CC-BY-NC-SA-4.0",
        "paper": "Yao et al. 2025, arXiv:2505.10793",
    },
    "musicgen": {
        "repo": "facebook/musicgen-small",
        "revision": "4c8334b02c6ec4e8664a91979669a501ec497792",
        "file": "hf/hub/models--facebook--musicgen-small/snapshots/4c8334b02c6ec4e8664a91979669a501ec497792/model.safetensors",
        "sha256": None,  # registrato a runtime (manifest della run)
        "license": "CC-BY-NC-4.0 (pesi), MIT (codice)",
        "paper": "Copet et al. 2023, NeurIPS, arXiv:2306.05284",
        "use": "unconditional (null text condition, HF get_unconditional_inputs), teacher forcing on EnCodec 32 kHz tokens, delay pattern",
    },
}

AUDIOBOX_AXES = ("CE", "CU", "PC", "PQ")
# PC (complessità) non è una qualità: escluso dal punteggio estetico.
AUDIOBOX_QUALITY_AXES = ("CE", "CU", "PQ")
SONGEVAL_AXES = ("Coherence", "Musicality", "Memorability", "Clarity", "Naturalness")

SR_AUDIOBOX = 16000
SR_MUQ = 24000
SR_MUSICGEN = 32000
MUSICGEN_FPS = 50.0
MUQ_FPS = 25.0

AESTHETIC_WINDOW_S = 10.0
AESTHETIC_JITTER_S = (-1.0, 0.0, 1.0)
SEAM_WINDOW_S = 12.0          # finestra MuQ centrata sulla giunta (c)
SEAM_SIDE_S = 3.0             # medie prima/dopo
SEAM_GUARD_S = 0.25           # esclusione attorno alla giunta per prima/dopo
SURPRISAL_MAX_S = 30.0        # MusicGen addestrato su crop di 30 s
SURPRISAL_SMOOTH_S = 0.25
SURPRISAL_SEARCH_PRE_S = 0.5
SURPRISAL_SEARCH_POST_S = 2.0
SURPRISAL_BASE_GUARD_S = 1.0
SURPRISAL_BASE_SKIP_S = 2.0   # i primi frame (senza contesto) non entrano nella baseline

# --------------------------------------------------------------- combination
# Regola dichiarata prima della validazione (vedi PREREGISTRATION.json della
# run): pesi uguali sui critici disponibili; ogni critico è mappato in [0,1]
# come "bontà" (0.5 = come una giunta naturale tipica).
COMBINATION = {
    "weights": {"aesthetic": 1.0, "surprisal": 1.0, "seam": 1.0, "technical": 1.0},
    "technical_map": {"pass": 1.0, "warn": 0.6, "fail": 0.15},
    "rule": "media pesata dei critici disponibili; sd = sqrt(sum w_i^2 sd_i^2)/sum w_i; disaccordo = sd tra critici",
    "declared": "2026-10-21, prima di qualsiasi confronto con i giudizi umani",
}


# ===================================================================== audio
def load_audio(path: os.PathLike | str, *, start_s: float = 0.0, duration_s: Optional[float] = None,
               sr: Optional[int] = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Legge [start_s, start_s+duration_s) e ricampiona (resample_poly).

    Fuori dai bordi del file la finestra è *accorciata*, mai riempita di zeri:
    il chiamante riceve la lunghezza reale.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    info = sf.info(str(path))
    native = int(info.samplerate)
    s0 = max(0, int(round(start_s * native)))
    n = None if duration_s is None else max(0, int(round(duration_s * native)))
    stop = info.frames if n is None else min(info.frames, s0 + n)
    x, _ = sf.read(str(path), start=s0, stop=stop, dtype="float32", always_2d=True)
    if mono:
        x = x.mean(axis=1)
    if sr is not None and sr != native:
        g = math.gcd(native, sr)
        x = resample_poly(x, sr // g, native // g, axis=0).astype(np.float32)
        native = sr
    return np.ascontiguousarray(x, dtype=np.float32), native


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    g = math.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)


def sha256_file(path: os.PathLike | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ===================================================================== items
@dataclass
class TransitionItem:
    """Una transizione da valutare, con coordinate nella clip e nelle sorgenti.

    ``clip_offset_s`` permette di leggere la clip da un file più lungo (il
    set intero): la clip è ``wav[clip_offset_s : clip_offset_s + clip_dur_s]``.
    """

    item_id: str
    dataset: str
    wav: str
    clip_offset_s: float
    clip_dur_s: float
    start_s: float
    switch_s: float
    end_s: float
    kind: str
    group: str = ""
    a_audio: Optional[str] = None
    a_switch_s: Optional[float] = None
    a_track_id: Optional[str] = None
    b_audio: Optional[str] = None
    b_switch_s: Optional[float] = None
    b_track_id: Optional[str] = None
    qa: Optional[dict] = None
    label: Optional[dict] = None
    wav_sha256: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "TransitionItem":
        return cls(**d)

    def read_clip(self, sr: int, start_s: float = 0.0, duration_s: Optional[float] = None) -> np.ndarray:
        """Mono, coordinate della clip (0 = inizio clip)."""
        dur = self.clip_dur_s - start_s if duration_s is None else duration_s
        dur = max(0.0, min(dur, self.clip_dur_s - start_s))
        x, _ = load_audio(self.wav, start_s=self.clip_offset_s + start_s, duration_s=dur, sr=sr)
        return x


def item_from_render(render: dict, *, item_id: str, dataset: str, wav: str, qa: Optional[dict] = None,
                     clip_offset_s: float = 0.0, clip_dur_s: Optional[float] = None, group: str = "",
                     label: Optional[dict] = None, root: Path = ROOT) -> TransitionItem:
    """Costruisce un item dai metadati di ``dj_render``/``dj_effects``."""
    sr = int(render.get("sample_rate", 48000))
    ov = render.get("overlap") or {}
    sw = render.get("switch") or {}
    start = int(ov.get("output_start_sample", sw.get("output_sample", 0)))
    end = int(ov.get("output_end_sample", sw.get("output_sample", start)))
    switch = int(sw.get("output_sample", (start + end) // 2))
    clip = render.get("clip") or {}
    n = clip.get("length_samples")
    dur = clip_dur_s if clip_dur_s is not None else (n / sr if n else None)
    if dur is None:
        import soundfile as sf
        dur = sf.info(str(wav)).duration
    prov = render.get("provenance") or {}
    pa, pb = prov.get("a") or {}, prov.get("b") or {}

    def _abs(p):
        if not p:
            return None
        q = Path(p)
        return str(q if q.is_absolute() else root / q)

    a_src = sw.get("a_source_sample")
    b_src = sw.get("b_source_sample")
    return TransitionItem(
        item_id=item_id, dataset=dataset, wav=str(wav), clip_offset_s=float(clip_offset_s), clip_dur_s=float(dur),
        start_s=start / sr, switch_s=switch / sr, end_s=end / sr, kind=str(render.get("kind", "unknown")), group=group,
        a_audio=_abs(pa.get("audio")), a_switch_s=None if a_src is None else a_src / sr, a_track_id=pa.get("audio_sha256"),
        b_audio=_abs(pb.get("audio")), b_switch_s=None if b_src is None else b_src / sr, b_track_id=pb.get("audio_sha256"),
        qa=qa, label=label,
        meta={"a_slug": pa.get("slug"), "b_slug": pb.get("slug"), "bars": render.get("bars"),
              "stretch_pct": (render.get("stretch") or {}).get("tempo_jump_pct")},
    )


# ======================================================= (a) aesthetic critic
def aesthetic_windows(item: TransitionItem, window_s: float = AESTHETIC_WINDOW_S,
                      jitter: Sequence[float] = AESTHETIC_JITTER_S) -> list[dict]:
    """Finestre (clip e sorgenti) per il critico estetico.

    * ``seam@j``: 10 s centrati sullo switch (+ jitter j) nella clip;
    * ``pre``: i 10 s che finiscono all'inizio della transizione (A solo);
    * ``post``: i 10 s che iniziano alla fine della transizione (B solo);
    * ``srcA``: A originale centrata sul punto sorgente dello switch
      (l'alternativa "restare su A"); ``srcB``: B originale centrata sul suo
      punto sorgente allo switch.
    Le finestre troncate dai bordi restano più corte (mai zeri).
    """
    half = window_s / 2
    out = []
    for j in jitter:
        c = item.switch_s + j
        s = min(max(0.0, c - half), max(0.0, item.clip_dur_s - window_s))
        out.append({"name": f"seam@{j:+.0f}", "source": "clip", "start_s": s, "dur_s": window_s})
    out.append({"name": "pre", "source": "clip", "start_s": max(0.0, item.start_s - window_s),
                "dur_s": min(window_s, item.start_s)})
    out.append({"name": "post", "source": "clip", "start_s": item.end_s,
                "dur_s": min(window_s, item.clip_dur_s - item.end_s)})
    if item.a_audio and item.a_switch_s is not None:
        out.append({"name": "srcA", "source": "a", "start_s": max(0.0, item.a_switch_s - half), "dur_s": window_s})
    if item.b_audio and item.b_switch_s is not None:
        out.append({"name": "srcB", "source": "b", "start_s": max(0.0, item.b_switch_s - half), "dur_s": window_s})
    return [w for w in out if w["dur_s"] >= 3.0]


def read_window(item: TransitionItem, w: dict, sr: int) -> np.ndarray:
    if w["source"] == "clip":
        return item.read_clip(sr, w["start_s"], w["dur_s"])
    path = item.a_audio if w["source"] == "a" else item.b_audio
    x, _ = load_audio(path, start_s=w["start_s"], duration_s=w["dur_s"], sr=sr)
    return x


def _axis_mean(scores: dict, axes: Sequence[str]) -> float:
    return float(np.mean([scores[a] for a in axes]))


def aesthetic_relative(window_scores: dict[str, dict], axes: Sequence[str]) -> dict:
    """Confronta la giunta con lati e sorgenti, per un modello (assi dati).

    ``window_scores``: nome finestra -> {asse: valore}. Restituisce i delta
    (giunta - riferimento; negativo = la giunta peggiora) e l'incertezza della
    giunta come deviazione standard sul jitter.
    """
    seams = [v for k, v in window_scores.items() if k.startswith("seam@")]
    if not seams:
        return {"status": "missing"}
    seam_vals = np.array([_axis_mean(s, axes) for s in seams])
    seam = float(seam_vals.mean())
    seam_sd = float(seam_vals.std(ddof=1)) if len(seam_vals) > 1 else float("nan")
    res: dict[str, Any] = {"status": "ok", "axes": list(axes), "seam": seam, "seam_sd_jitter": seam_sd,
                           "per_axis_seam": {a: float(np.mean([s[a] for s in seams])) for a in axes}}
    sides = [window_scores[k] for k in ("pre", "post") if k in window_scores]
    if sides:
        ref = float(np.mean([_axis_mean(s, axes) for s in sides]))
        res["sides"] = ref
        res["delta_vs_sides"] = seam - ref
        res["per_axis_delta_vs_sides"] = {a: res["per_axis_seam"][a] - float(np.mean([s[a] for s in sides])) for a in axes}
    srcs = {k: window_scores[k] for k in ("srcA", "srcB") if k in window_scores}
    if srcs:
        for k, v in srcs.items():
            res[f"delta_vs_{k}"] = seam - _axis_mean(v, axes)
        res["delta_vs_sources"] = seam - float(np.mean([_axis_mean(v, axes) for v in srcs.values()]))
    return res


# ======================================================= (b) surprisal critic
def smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.asarray(x, dtype=np.float64)
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), (pad, n - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def robust_z(x: np.ndarray | float, ref: np.ndarray) -> np.ndarray | float:
    ref = np.asarray(ref, dtype=np.float64)
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med))) * 1.4826
    if not mad > 1e-9:
        mad = float(np.std(ref)) or 1.0
    return (np.asarray(x, dtype=np.float64) - med) / mad


def surprisal_seam_stats(nll: np.ndarray, *, fps: float, start_s: float, switch_s: float, end_s: float,
                         offset_s: float = 0.0, smooth_s: float = SURPRISAL_SMOOTH_S, n_boot: int = 200,
                         seed: int = 0) -> dict:
    """Picco di sorpresa alla giunta rispetto alla sorpresa naturale della clip.

    ``nll``: NLL per frame (nats, somma sui codebook) della porzione di clip
    che inizia a ``offset_s``. Baseline = frame A-solo (da 2 s dopo l'inizio
    fino a 1 s prima della transizione) e B-solo (da 1 s dopo la fine della
    transizione). Il picco è il massimo della NLL lisciata (0,25 s) tra 0,5 s
    prima dell'inizio e 2 s dopo la fine della transizione. z robusto
    (mediana/MAD). Incertezza: bootstrap dei frame di baseline.
    """
    nll = np.asarray(nll, dtype=np.float64)
    t = offset_s + np.arange(len(nll)) / fps
    sm = smooth(nll, max(1, int(round(smooth_s * fps))))
    base = ((t >= offset_s + SURPRISAL_BASE_SKIP_S) & (t < start_s - SURPRISAL_BASE_GUARD_S)) | (t >= end_s + SURPRISAL_BASE_GUARD_S)
    a_base = base & (t < start_s)
    b_base = base & (t >= end_s)
    search = (t >= start_s - SURPRISAL_SEARCH_PRE_S) & (t <= end_s + SURPRISAL_SEARCH_POST_S)
    if base.sum() < int(2 * fps) or search.sum() == 0:
        return {"status": "insufficient_context", "n_base": int(base.sum())}
    peak_idx = int(np.flatnonzero(search)[np.argmax(sm[search])])
    peak = float(sm[peak_idx])
    ref = sm[base]
    z = float(robust_z(peak, ref))
    post = (t >= switch_s) & (t < switch_s + 2.0)
    excess = float(np.mean(nll[post]) - np.median(nll[base])) if post.any() else float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    idx = np.flatnonzero(base)
    for _ in range(n_boot):
        bi = rng.choice(idx, size=len(idx), replace=True)
        boots.append(float(robust_z(peak, sm[bi])))
    return {
        "status": "ok",
        "peak_z": z,
        "peak_z_sd": float(np.std(boots)),
        "peak_t_s": float(t[peak_idx]),
        "peak_nll": peak,
        "base_median": float(np.median(ref)),
        "base_mad": float(np.median(np.abs(ref - np.median(ref))) * 1.4826),
        "a_base_median": float(np.median(sm[a_base])) if a_base.any() else None,
        "b_base_median": float(np.median(sm[b_base])) if b_base.any() else None,
        "excess_post2s": excess,
        "n_base_frames": int(base.sum()),
        "mean_nll": float(nll.mean()),
    }


def musicgen_delay_targets(codes: np.ndarray, pad_id: int, bos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Sequenza con pattern "delay" di MusicGen: (input, target) di forma (K, F+K-1).

    ``codes``: (K, F). Il codebook k del frame f sta in posizione f+k.
    """
    codes = np.asarray(codes)
    k_, f_ = codes.shape
    s_ = f_ + k_ - 1
    delayed = np.full((k_, s_), pad_id, dtype=np.int64)
    for k in range(k_):
        delayed[k, k:k + f_] = codes[k]
    inp = np.concatenate([np.full((k_, 1), bos_id, dtype=np.int64), delayed[:, :-1]], axis=1)
    return inp, delayed


def frame_nll_from_logprobs(logprobs: np.ndarray, delayed: np.ndarray, n_frames: int) -> np.ndarray:
    """NLL per frame (somma sui codebook) da log-probabilità (K, S, V)."""
    k_ = delayed.shape[0]
    out = np.zeros((k_, n_frames))
    for k in range(k_):
        pos = np.arange(n_frames) + k
        out[k] = -logprobs[k, pos, delayed[k, pos]]
    return out


class MusicGenSurprisal:
    """MusicGen-small non condizionato come modello di aspettativa audio.

    Gira nel venv ``venv-np1`` (transformers 4.42.3). Teacher forcing sui token
    EnCodec 32 kHz della clip; condizione testuale nulla come in
    ``MusicgenForConditionalGeneration.get_unconditional_inputs``.
    """

    def __init__(self, model_dir: Optional[Path] = None, threads: int = 2, dtype: str = "bfloat16"):
        self.model_dir = Path(model_dir or MODELS / PROVENANCE["musicgen"]["file"]).parent
        self.threads = threads
        # bf16 (AMX sulla macchina di analisi): memoria del decoder dimezzata; verificato contro fp32
        # su due clip: r = 0,9997 della NLL per frame, picco z invariato entro 0,02.
        self.dtype = dtype
        self.model = None

    def load(self):
        """Carica solo decoder ed EnCodec dal safetensors ufficiale, tensore per tensore.

        T5 non serve: con la condizione nulla di ``get_unconditional_inputs`` gli
        stati dell'encoder sono moltiplicati per una maschera di zeri, quindi
        equivalgono a un tensore nullo mascherato. Memoria ~1,5 GB invece di ~2,7.
        """
        if self.model is not None:
            return
        import torch
        from safetensors import safe_open
        from transformers import EncodecModel, MusicgenConfig, MusicgenForCausalLM

        torch.set_num_threads(self.threads)
        self.torch = torch
        cfg = MusicgenConfig.from_pretrained(str(self.model_dir), local_files_only=True)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(getattr(torch, self.dtype))
        try:
            dec = MusicgenForCausalLM(cfg.decoder).eval()
        finally:
            torch.set_default_dtype(prev)
        enc = EncodecModel(cfg.audio_encoder).eval()
        loaded = {"decoder": 0, "audio_encoder": 0}
        with safe_open(str(self.model_dir / "model.safetensors"), framework="pt") as f, torch.no_grad():
            for prefix, mod in (("decoder.", dec), ("audio_encoder.", enc)):
                own = dict(mod.state_dict(keep_vars=True))
                for k in f.keys():
                    if not k.startswith(prefix):
                        continue
                    name = k[len(prefix):]
                    if name not in own:
                        raise RuntimeError(f"chiave inattesa nel checkpoint: {k}")
                    own[name].copy_(f.get_tensor(k))
                    loaded[prefix[:-1]] += 1
        self.model = dec
        self.encodec = enc
        self.cfg = cfg
        self.loaded_tensors = loaded

    def frame_nll(self, mono32k: np.ndarray) -> np.ndarray:
        """(K, F) NLL in nats per codebook e frame (50 Hz)."""
        self.load()
        torch, dec = self.torch, self.model
        with torch.inference_mode():
            enc = self.encodec.encode(torch.from_numpy(np.ascontiguousarray(mono32k, dtype=np.float32))[None, None, :])
            codes = enc.audio_codes[0, 0].cpu().numpy()  # (K, F)
            pad = int(self.cfg.decoder.pad_token_id)
            bos = int(self.cfg.decoder.bos_token_id)
            inp, delayed = musicgen_delay_targets(codes, pad, bos)
            enc_h = torch.zeros(1, 1, self.cfg.decoder.hidden_size, dtype=next(dec.parameters()).dtype)
            mask = torch.zeros(1, 1, dtype=torch.long)
            out = dec(input_ids=torch.from_numpy(inp), encoder_hidden_states=enc_h, encoder_attention_mask=mask)
            logp = torch.log_softmax(out.logits.float(), dim=-1).cpu().numpy()  # (K, S, V)
        return frame_nll_from_logprobs(logp, delayed, codes.shape[1])


# ============================================== (c) seam discriminator critic
def seam_feature_block(frames: np.ndarray, *, fps: float, seam_s: float, side_s: float = SEAM_SIDE_S,
                       guard_s: float = SEAM_GUARD_S) -> dict:
    """Riassunti di una sequenza di embedding (T, D) attorno a ``seam_s``.

    Restituisce i vettori medi ``pre``/``post``/``across`` e statistiche
    compatte di discontinuità:
    * ``cos_pre_post`` e ``l2_pre_post`` (normalizzata per sqrt(D));
    * ``jump_z``: salto frame-a-frame massimo entro +-0,3 s dalla giunta,
      z robusto rispetto ai salti nel resto della finestra;
    * ``foote_z``: novità a scacchiera (kernel 1,5 s, Foote 2000) alla giunta,
      z rispetto alle altre posizioni valide della finestra.
    """
    x = np.asarray(frames, dtype=np.float64)
    t_ = x.shape[0]
    t = (np.arange(t_) + 0.5) / fps
    pre = (t >= seam_s - side_s) & (t < seam_s - guard_s)
    post = (t > seam_s + guard_s) & (t <= seam_s + side_s)
    acr = (t >= seam_s - 0.75) & (t <= seam_s + 0.75)
    if pre.sum() < 3 or post.sum() < 3:
        raise ValueError("finestra troppo corta attorno alla giunta")
    v_pre, v_post, v_acr = x[pre].mean(0), x[post].mean(0), x[acr].mean(0)
    cos = float(v_pre @ v_post / (np.linalg.norm(v_pre) * np.linalg.norm(v_post) + 1e-9))
    l2 = float(np.linalg.norm(v_pre - v_post) / math.sqrt(x.shape[1]))
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    d = np.linalg.norm(np.diff(xn, axis=0), axis=1)  # salto tra frame i e i+1
    td = t[1:]
    near = np.abs(td - seam_s) <= 0.3
    far = np.abs(td - seam_s) > 1.0
    jump_z = float(robust_z(d[near].max(), d[far])) if near.any() and far.sum() > 5 else 0.0
    # Foote checkerboard sulla matrice di similarità coseno
    sim = xn @ xn.T
    w = max(2, int(round(1.5 * fps)))
    nov = np.full(t_, np.nan)
    for i in range(w, t_ - w):
        a = sim[i - w:i, i - w:i].mean() + sim[i:i + w, i:i + w].mean()
        b = sim[i - w:i, i:i + w].mean() * 2
        nov[i] = a - b
    valid = ~np.isnan(nov)
    seam_i = int(np.argmin(np.abs(t - seam_s)))
    lo, hi = max(0, seam_i - int(0.3 * fps)), min(t_, seam_i + int(0.3 * fps) + 1)
    near_nov = nov[lo:hi][~np.isnan(nov[lo:hi])]
    other = valid.copy()
    other[max(0, seam_i - int(1.0 * fps)):seam_i + int(1.0 * fps) + 1] = False
    foote_z = float(robust_z(near_nov.max(), nov[other])) if near_nov.size and other.sum() > 5 else 0.0
    return {"pre": v_pre, "post": v_post, "across": v_acr,
            "compact": {"cos_pre_post": cos, "l2_pre_post": l2, "jump_z": jump_z, "foote_z": foote_z}}


COMPACT_KEYS = ("cos_pre_post", "l2_pre_post", "jump_z", "foote_z")


def seam_design_row(blocks: dict[str, dict], extra: Optional[dict] = None) -> tuple[np.ndarray, np.ndarray]:
    """Vettori di feature per il classificatore da blocchi per layer.

    ``blocks``: layer -> output di ``seam_feature_block``. Restituisce
    (compact, full): compact = statistiche per layer (+ extra);
    full = [pre, post, |pre-post|, across] concatenati per layer.
    """
    compact, full = [], []
    for layer in sorted(blocks):
        b = blocks[layer]
        compact.extend(b["compact"][k] for k in COMPACT_KEYS)
        full.append(np.concatenate([b["pre"], b["post"], np.abs(b["pre"] - b["post"]), b["across"]]))
    if extra:
        compact.extend(extra[k] for k in sorted(extra))
    return np.asarray(compact, dtype=np.float64), np.concatenate(full).astype(np.float32)


class LinearSeamModel:
    """Logistica standardizzata serializzabile in JSON (niente pickle)."""

    def __init__(self, mean, scale, coef, intercept, pca_mean=None, pca_components=None, meta=None):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.pca_mean = None if pca_mean is None else np.asarray(pca_mean, dtype=np.float64)
        self.pca_components = None if pca_components is None else np.asarray(pca_components, dtype=np.float64)
        self.meta = meta or {}

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if self.pca_components is not None:
            x = (x - self.pca_mean) @ self.pca_components.T
        return (x - self.mean) / self.scale

    def decision(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x) @ self.coef + self.intercept

    def proba(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(x)))

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(),
                "intercept": self.intercept,
                "pca_mean": None if self.pca_mean is None else self.pca_mean.tolist(),
                "pca_components": None if self.pca_components is None else self.pca_components.tolist(),
                "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "LinearSeamModel":
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], d.get("pca_mean"), d.get("pca_components"), d.get("meta"))


def fit_linear_seam_model(x: np.ndarray, y: np.ndarray, *, c: float = 1.0, pca: Optional[int] = None,
                          seed: int = 0) -> LinearSeamModel:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(x, dtype=np.float64)
    pm = pc = None
    if pca:
        p = PCA(n_components=min(pca, x.shape[0] - 1, x.shape[1]), random_state=seed).fit(x)
        pm, pc = p.mean_, p.components_
        x = (x - pm) @ pc.T
    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-9] = 1.0
    clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit((x - mu) / sd, y)
    return LinearSeamModel(mu, sd, clf.coef_[0], clf.intercept_[0], pm, pc, {"C": c, "pca": pca})


# ===================================================== (d) technical critic
# v2: sulla sola clip true_peak dipende dal master delle sorgenti e verse_cut
# presume che A esca; su una continuazione naturale entrambi falliscono senza
# difetto di giunta. Restano le verifiche misurate sul mix attorno alla giunta.
TECHNICAL_V2_CHECKS = ("rhythm", "loudness_jump")


def technical_from_qa(qa: Optional[dict], only: Optional[Sequence[str]] = None) -> dict:
    """Riassume un report ``dj_qa`` in feature e in un punteggio dichiarato."""
    if not qa:
        return {"status": "missing"}
    checks = qa.get("checks") or {}
    stat = {k: (v or {}).get("status") for k, v in checks.items()}
    rh = ((checks.get("rhythm") or {}).get("methods") or {})
    dec = rh.get("beat_this_decks") or {}
    mix = rh.get("beat_this_mix") or {}
    lj = checks.get("loudness_jump") or {}
    feats = {
        "verdict": qa.get("verdict"),
        "n_fail": sum(1 for s in stat.values() if s == "fail"),
        "n_warn": sum(1 for s in stat.values() if s == "warn"),
        "downbeat_median_abs_ms": dec.get("downbeat_median_abs_ms"),
        "mix_phase_ms": mix.get("phase_error_ms") if isinstance(mix, dict) else None,
        "loudness_jump_db": lj.get("signed_jump_db", lj.get("max_abs_jump_db", lj.get("jump_db"))),
        "double_bass_s": (checks.get("double_bass") or {}).get("longest_s"),
        "vocal_clash_s": (checks.get("vocal_clash") or {}).get("clash_s"),
        "true_peak_dbtp": (checks.get("true_peak") or {}).get("true_peak_dbtp"),
        "statuses": stat,
    }
    # Escluso dal punteggio: "duration" (regola delle clip d'ascolto, non musica)
    relevant = {k: v for k, v in stat.items() if k != "duration" and v in COMBINATION["technical_map"]
                and (only is None or k in only)}
    if not relevant:
        return {"status": "missing", "features": feats}
    vals = [COMBINATION["technical_map"][v] for v in relevant.values()]
    score = float(min(vals))
    # incertezza: un warn è per costruzione ambiguo
    sd = 0.2 if any(v == "warn" for v in relevant.values()) else 0.1
    return {"status": "ok", "score": score, "sd": sd, "features": feats, "rule": "min sulle verifiche (pass 1, warn 0.6, fail 0.15)"}


# ============================================================== combination
def empirical_percentile(value: float, null: Sequence[float]) -> float:
    """Percentile (0-1) con correzione (r+0.5)/(n+1) contro una distribuzione nulla."""
    null = np.asarray([v for v in null if v is not None and np.isfinite(v)], dtype=np.float64)
    if null.size == 0 or value is None or not np.isfinite(value):
        return float("nan")
    r = float(np.sum(null < value) + 0.5 * np.sum(null == value))
    return (r + 0.5) / (null.size + 1)


def percentile_sd(value: float, sd: float, null: Sequence[float], n: int = 200, seed: int = 0) -> float:
    """Incertezza del percentile propagando ``sd`` sul valore (Monte Carlo)."""
    if sd is None or not np.isfinite(sd) or sd <= 0:
        return 0.05
    rng = np.random.default_rng(seed)
    vals = [empirical_percentile(v, null) for v in rng.normal(value, sd, n)]
    return float(np.nanstd(vals))


def combine(critics: dict[str, dict], weights: Optional[dict] = None) -> dict:
    """Combina critici già espressi come bontà in [0,1] con sd.

    ``critics``: nome -> {'score': float, 'sd': float}. Critici mancanti o con
    score non finito sono esclusi (e riportati).
    """
    weights = weights or COMBINATION["weights"]
    used = {k: v for k, v in critics.items() if v and v.get("score") is not None and np.isfinite(v["score"]) and weights.get(k, 0) > 0}
    if not used:
        return {"status": "missing", "score": float("nan"), "sd": float("nan"), "used": []}
    w = np.array([weights[k] for k in used])
    s = np.array([used[k]["score"] for k in used])
    sd = np.array([used[k].get("sd") if used[k].get("sd") is not None and np.isfinite(used[k].get("sd")) else 0.25 for k in used])
    score = float(np.sum(w * s) / w.sum())
    unc = float(math.sqrt(np.sum((w * sd) ** 2)) / w.sum())
    dis = float(np.std(s)) if len(s) > 1 else 0.0
    return {"status": "ok", "score": score, "sd": unc, "disagreement_sd": dis, "used": list(used),
            "missing": [k for k in weights if k not in used]}


# =============================================================== validation
def pairwise_accuracy(pred: Sequence[float], truth: Sequence[float], groups: Optional[Sequence[str]] = None,
                      within_groups_only: bool = False) -> dict:
    """Accuratezza a coppie: tra coppie con verità diversa, quota in cui il
    critico ordina come l'umano (pareggi del critico = 0,5)."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    n_ok = 0.0
    pairs = []
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if truth[i] == truth[j] or not np.isfinite(pred[i]) or not np.isfinite(pred[j]):
                continue
            if within_groups_only and groups is not None and groups[i] != groups[j]:
                continue
            sgn_t = np.sign(truth[i] - truth[j])
            d = pred[i] - pred[j]
            ok = 0.5 if d == 0 else float(np.sign(d) == sgn_t)
            n_ok += ok
            pairs.append((i, j, ok))
    n = len(pairs)
    return {"n_pairs": n, "accuracy": (n_ok / n) if n else float("nan"), "pairs": pairs}


def wilson_interval(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def spearman_with_permutation(pred: Sequence[float], truth: Sequence[float], n_perm: int = 20000,
                              seed: int = 0) -> dict:
    from scipy.stats import spearmanr

    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[ok], truth[ok]
    if len(pred) < 3:
        return {"n": int(len(pred)), "rho": float("nan"), "p_perm": float("nan")}
    rho = float(spearmanr(pred, truth).statistic)
    rng = np.random.default_rng(seed)
    perm = np.array([spearmanr(rng.permutation(pred), truth).statistic for _ in range(n_perm)])
    p = float((np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (n_perm + 1))
    return {"n": int(len(pred)), "rho": rho, "p_perm_two_sided": p}


def bootstrap_pairwise_ci(pred: Sequence[float], truth: Sequence[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """IC 95% dell'accuratezza a coppie con bootstrap sugli *item* (non sulle coppie)."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    rng = np.random.default_rng(seed)
    accs = []
    n = len(pred)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = pairwise_accuracy(pred[idx], truth[idx])
        if r["n_pairs"]:
            accs.append(r["accuracy"])
    if not accs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(path: os.PathLike | str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(p)


# ================================================ synthetic seams for (c)
# Etichette: 1 = giunta plausibile (continuazione reale o salto pulito su
# inizio frase), 0 = giunta con un difetto noto. Ogni esempio dura
# ``EXAMPLE_S`` secondi con la giunta al centro.
EXAMPLE_S = 12.0
EXAMPLE_LONG_S = 24.0  # sottoinsieme per estetica/sorpresa (servono lati di 10 s)
POSITIVE_KINDS = ("natural", "natural_vocoder", "jump_clean", "x_clean")
NEGATIVE_KINDS = ("phase", "gain", "key", "word_cut", "double_bass", "off_phrase",
                  "x_phase", "x_gain", "x_key", "x_word_cut", "x_double_bass", "x_off_phrase")
CAMELOT_OK = 1          # distanza Camelot considerata compatibile (stessa, adiacente, relativa)
CAMELOT_BAD = 3


def camelot_distance(a: Optional[str], b: Optional[str]) -> Optional[int]:
    """Distanza sul cerchio Camelot (0 stessa; 1 adiacente o relativa; ...)."""
    if not a or not b:
        return None
    try:
        na, la = int(a[:-1]), a[-1].upper()
        nb, lb = int(b[:-1]), b[-1].upper()
    except ValueError:
        return None
    d = min((na - nb) % 12, (nb - na) % 12)
    return d + (0 if la == lb else 1) if not (d == 0 and la != lb) else 1


@dataclass
class TrackInfo:
    slug: str
    wav: str
    track_id: str
    duration_s: float
    bpm: float
    camelot: Optional[str]
    downbeats_s: list
    phrase_level: list          # per battuta (0, 4, 8, 16)
    words: list                 # [(start_s, end_s)]
    source: str
    fold: int = -1


def load_track_info(analysis_path: os.PathLike | str, root: Path = ROOT) -> Optional[TrackInfo]:
    """Da un'analisi ``dj_select`` (tracks/<slug>.json) a ``TrackInfo``.

    Restituisce None se mancano audio o griglia (file incompleti non usati).
    """
    try:
        d = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    wav = (d.get("inputs") or {}).get("canonical_wav")
    grid = (d.get("grid") or {}).get("downbeats_s")
    if not wav or not grid or not Path(wav).is_file():
        return None
    phrases = d.get("phrases") or []
    levels = [int(p.get("phrase_level", 0)) for p in phrases] if len(phrases) == len(grid) else [16 if i % 8 == 0 else 0 for i in range(len(grid))]
    words = []
    vpath = (d.get("inputs") or {}).get("vocals")
    if vpath and Path(vpath).is_file():
        try:
            v = json.loads(Path(vpath).read_text(encoding="utf-8"))
            sr = float(v.get("sample_rate", 48000))
            words = [(w["start_sample"] / sr, w["end_sample"] / sr) for w in v.get("words", [])]
        except (OSError, ValueError, KeyError):
            words = []
    return TrackInfo(slug=d.get("slug") or Path(analysis_path).stem, wav=str(wav), track_id=d.get("track_id", ""),
                     duration_s=float(d.get("duration_s", 0.0)), bpm=float((d.get("tempo") or {}).get("bpm", 0.0)),
                     camelot=(d.get("key") or {}).get("camelot"), downbeats_s=[float(x) for x in grid],
                     phrase_level=levels, words=words, source=str(analysis_path))


def word_active(words: Sequence[tuple], t: float, guard: float = 0.05) -> bool:
    return any(s + guard < t < e - guard for s, e in words)


def assign_folds(tracks: Sequence[TrackInfo], k: int = 5) -> None:
    """Fold deterministico per brano (hash dello slug)."""
    order = sorted(tracks, key=lambda tr: hashlib.sha256(tr.slug.encode()).hexdigest())
    for i, tr in enumerate(order):
        tr.fold = i % k


def _tempo_compatible(a: float, b: float, tol: float = 0.03) -> bool:
    for f in (0.5, 1.0, 2.0):
        if b > 0 and abs(a / (b * f) - 1.0) <= tol:
            return True
    return False


def plan_examples(tracks: Sequence[TrackInfo], *, seed: int = 0, margin_s: float = EXAMPLE_LONG_S / 2 + 1.0,
                  per_track: Optional[dict] = None, per_cross: Optional[dict] = None) -> list[dict]:
    """Ricette deterministiche degli esempi sintetici (nessun audio scritto).

    Coppie cross-track solo dentro lo stesso fold: nessun brano compare in due
    fold, così la validazione per brano resta pulita.
    """
    rng = np.random.default_rng(seed)
    per_track = per_track or {"natural": 4, "natural_vocoder": 2, "jump_clean": 3, "phase": 2, "gain": 2,
                              "key": 2, "word_cut": 2, "double_bass": 2, "off_phrase": 2}
    per_cross = per_cross or {"x_clean": 4, "x_key": 2, "x_phase": 2, "x_gain": 2, "x_word_cut": 2,
                              "x_double_bass": 2, "x_off_phrase": 2}
    out: list[dict] = []
    by_fold: dict[int, list[TrackInfo]] = {}
    for tr in tracks:
        by_fold.setdefault(tr.fold, []).append(tr)

    def ok_time(tr, t):
        return margin_s <= t <= tr.duration_s - margin_s

    def bars(tr, cond):
        return [i for i, t in enumerate(tr.downbeats_s) if ok_time(tr, t) and cond(i, t)]

    def pick(lst, n):
        if not lst:
            return []
        idx = rng.choice(len(lst), size=min(n, len(lst)), replace=False)
        return [lst[i] for i in idx]

    for tr in tracks:
        phrase = bars(tr, lambda i, t: tr.phrase_level[i] >= 8)
        phrase_free = [i for i in phrase if not word_active(tr.words, tr.downbeats_s[i])]
        mid = bars(tr, lambda i, t: tr.phrase_level[i] == 0)
        worded = bars(tr, lambda i, t: word_active(tr.words, t, guard=0.08))
        far = lambda i: [j for j in phrase_free if abs(j - i) >= 16]
        others = [o for o in tracks if o.slug != tr.slug and o.fold == tr.fold]
        for kind, n in per_track.items():
            for _ in range(n):
                fade = "cut" if kind == "word_cut" else str(rng.choice(["cut", "bar"]))
                rec = {"kind": kind, "domain": "within", "a": tr.slug, "fold": tr.fold, "fade": fade}
                if kind in ("natural", "natural_vocoder", "phase", "gain", "key", "double_bass"):
                    if not phrase:
                        break
                    i = int(pick(phrase, 1)[0])
                    rec.update(a_bar=i, a_t=tr.downbeats_s[i], b=tr.slug, b_t=tr.downbeats_s[i])
                    if kind == "phase":
                        rec["delta_ms"] = float(rng.choice([-1, 1]) * rng.uniform(30, 150))
                    elif kind == "gain":
                        rec["gain_db"] = float(rng.choice([-1, 1]) * rng.uniform(4, 8))
                    elif kind == "key":
                        rec["semitones"] = int(rng.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6]))
                    elif kind == "double_bass":
                        if not others:
                            break
                        o = others[int(rng.integers(len(others)))]
                        ob = bars(o, lambda i, t: o.phrase_level[i] >= 8)
                        if not ob:
                            break
                        j = int(pick(ob, 1)[0])
                        rec.update(bass_from=o.slug, bass_t=o.downbeats_s[j])
                elif kind in ("jump_clean", "off_phrase", "word_cut"):
                    src = {"jump_clean": phrase_free, "off_phrase": [i for i in mid if not word_active(tr.words, tr.downbeats_s[i])],
                           "word_cut": worded}[kind]
                    if not src:
                        break
                    i = int(pick(src, 1)[0])
                    tgt = far(i)
                    if not tgt:
                        break
                    j = int(pick(tgt, 1)[0])
                    rec.update(a_bar=i, a_t=tr.downbeats_s[i], b=tr.slug, b_bar=j, b_t=tr.downbeats_s[j])
                out.append(rec)
        # cross-track (stesso fold, tempo compatibile senza stretch)
        # Cross-track senza stretch: solo cut (niente sovrapposizione che
        # renderebbe udibile la deriva di tempo), tolleranza +-6%.
        compat = [o for o in others if _tempo_compatible(tr.bpm, o.bpm, tol=0.06)]
        for kind, n in per_cross.items():
            for _ in range(n):
                if kind == "x_key":
                    cands = [o for o in compat if (camelot_distance(tr.camelot, o.camelot) or 0) >= CAMELOT_BAD]
                else:
                    cands = [o for o in compat if camelot_distance(tr.camelot, o.camelot) is not None and camelot_distance(tr.camelot, o.camelot) <= CAMELOT_OK]
                if not cands:
                    break
                o = cands[int(rng.integers(len(cands)))]
                src = {"x_word_cut": worded, "x_off_phrase": [i for i in mid if not word_active(tr.words, tr.downbeats_s[i])]}.get(kind, phrase_free)
                ob = bars(o, lambda i, t: o.phrase_level[i] >= 8)
                if not src or not ob:
                    break
                i = int(pick(src, 1)[0])
                j = int(pick(ob, 1)[0])
                fade = "cut"
                rec = {"kind": kind, "domain": "cross", "a": tr.slug, "fold": tr.fold, "fade": fade,
                       "a_bar": i, "a_t": tr.downbeats_s[i], "b": o.slug, "b_bar": j, "b_t": o.downbeats_s[j],
                       "camelot_distance": camelot_distance(tr.camelot, o.camelot), "match_loudness": True}
                if kind == "x_phase":
                    rec["delta_ms"] = float(rng.choice([-1, 1]) * rng.uniform(30, 150))
                elif kind == "x_gain":
                    rec["gain_db"] = float(rng.choice([-1, 1]) * rng.uniform(4, 8))
                elif kind == "x_double_bass":
                    rec["bass_from"] = tr.slug
                    rec["bass_t"] = tr.downbeats_s[i]
                out.append(rec)
    for n, rec in enumerate(out):
        rec["label"] = 1 if rec["kind"] in POSITIVE_KINDS else 0
        rec["id"] = f"s{n:05d}_{rec['kind']}_{rec['a'][:16]}"
    return out


def _lowpass(x: np.ndarray, sr: int, hz: float = 150.0) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(4, hz, btype="low", fs=sr, output="sos"), x).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def synth_example(rec: dict, tracks: dict[str, TrackInfo], *, sr: int = 48000, length_s: float = EXAMPLE_S) -> np.ndarray:
    """Audio mono di un esempio: [giunta - L/2, giunta + L/2] con la giunta al centro.

    Il lato "pre" viene da A fino a ``a_t``; il lato "post" da ``b`` a partire
    da ``b_t`` (+ eventuale sfasamento), con dissolvenza ``cut`` (10 ms) o
    ``bar`` (una battuta lineare a guadagno costante: identità se pre e post
    sono lo stesso segnale continuo).
    """
    half = length_s / 2
    a = tracks[rec["a"]]
    b = tracks[rec["b"]]
    bar_s = 240.0 / a.bpm if a.bpm > 0 else 2.0
    fade_s = 0.010 if rec.get("fade", "cut") == "cut" else min(bar_s, 3.0)
    ext = fade_s / 2
    pre, _ = load_audio(a.wav, start_s=rec["a_t"] - half, duration_s=half + ext, sr=sr)
    b_start = rec["b_t"] + rec.get("delta_ms", 0.0) / 1000.0
    post, _ = load_audio(b.wav, start_s=b_start - ext, duration_s=half + ext, sr=sr)
    n_half, n_ext = int(round(half * sr)), int(round(ext * sr))
    if len(pre) < n_half + n_ext or len(post) < n_half + n_ext:
        raise ValueError(f"contesto insufficiente per {rec.get('id')}")
    pre, post = pre[: n_half + n_ext], post[: n_half + n_ext]
    kind = rec["kind"]
    if kind in ("natural_vocoder", "key") or rec.get("vocoder"):
        import librosa

        pre = librosa.effects.pitch_shift(pre, sr=sr, n_steps=0.0).astype(np.float32)
        post = librosa.effects.pitch_shift(post, sr=sr, n_steps=float(rec.get("semitones", 0))).astype(np.float32)
    if rec.get("match_loudness"):
        ref = _rms(pre[n_half - int(3 * sr): n_half])
        cur = _rms(post[n_ext: n_ext + int(3 * sr)])
        post = post * (ref / cur if cur > 1e-6 else 1.0)
    if "gain_db" in rec:
        post = post * (10 ** (rec["gain_db"] / 20.0))
    if "bass_from" in rec:
        src = tracks[rec["bass_from"]]
        # x_double_bass: il basso di A continua sotto B (niente bass swap);
        # double_bass: basso di un altro brano aggiunto dopo la giunta.
        start = rec["bass_t"] if kind == "double_bass" else rec["a_t"]
        bx, _ = load_audio(src.wav, start_s=start - ext, duration_s=half + ext, sr=sr)
        bx = _lowpass(bx[: len(post)], sr)
        if len(bx) < len(post):
            bx = np.pad(bx, (0, len(post) - len(bx)))
        own = _rms(_lowpass(post, sr))
        post = post + bx * (own / max(_rms(bx), 1e-6))
    # dissolvenza a guadagno costante centrata sulla giunta
    nf = max(2, int(round(fade_s * sr)))
    w = np.linspace(0.0, 1.0, nf, dtype=np.float32)
    head = pre[: n_half - nf // 2]
    xa = pre[n_half - nf // 2: n_half - nf // 2 + nf]
    xb = post[n_ext - nf // 2: n_ext - nf // 2 + nf]
    tail = post[n_ext - nf // 2 + nf: n_ext + n_half]
    if len(xa) != len(xb):  # arrotondamento di un campione ai bordi della lettura
        m = min(len(xa), len(xb))
        xa, xb, w = xa[:m], xb[:m], np.linspace(0.0, 1.0, m, dtype=np.float32)
    y = np.concatenate([head, xa * (1 - w) + xb * w, tail]).astype(np.float32)
    n = int(round(length_s * sr))
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


# ===================================================== dose-response (sensibilità)
DOSE_LEVELS = {
    # sfasamento di B (ms) con dissolvenza di una battuta: battiti che "flammano"
    "phase": [0.0, 25.0, 50.0, 100.0, 150.0, 200.0],
    # salto di livello (dB, |valore|; segno alternato per base) con cut
    "gain": [0.0, 1.5, 3.0, 4.5, 6.0, 9.0],
    # distanza Camelot 0..6 ottenuta con pitch shift (semitoni) del lato B, dissolvenza di una battuta;
    # tutti i livelli passano dal phase vocoder (anche 0) per non confondere artefatto e tonalità
    "key": [0, 1, 2, 3, 4, 5, 6],
}
CAMELOT_STEP_SEMITONES = {0: 0, 1: -5, 2: 2, 3: -3, 4: 4, 5: 1, 6: 6}


def plan_dose(tracks: Sequence[TrackInfo], n_bases: int = 8, seed: int = 5, margin_s: float = EXAMPLE_LONG_S / 2 + 1.0) -> list[dict]:
    """Varianti con difetti di intensità crescente su giunte reali buone.

    Base = continuazione naturale di un brano su un downbeat di inizio frase
    senza parola attiva. Più una coppia "salto pulito" / "salto con parola
    tagliata" sullo stesso brano (difetto binario).
    """
    rng = np.random.default_rng(seed)
    cands = []
    for tr in tracks:
        ph = [i for i, t in enumerate(tr.downbeats_s) if tr.phrase_level[i] >= 8 and margin_s <= t <= tr.duration_s - margin_s
              and not word_active(tr.words, t)]
        wd = [i for i, t in enumerate(tr.downbeats_s) if margin_s <= t <= tr.duration_s - margin_s and word_active(tr.words, t, guard=0.08)]
        if ph and wd and len(tr.words) > 50:
            cands.append((tr, ph, wd))
    cands.sort(key=lambda c: hashlib.sha256(c[0].slug.encode()).hexdigest())
    # una base per fold quando possibile
    chosen, seen_f = [], set()
    for c in cands:
        if c[0].fold not in seen_f:
            chosen.append(c)
            seen_f.add(c[0].fold)
    for c in cands:
        if len(chosen) >= n_bases:
            break
        if c not in chosen:
            chosen.append(c)
    out = []
    for b, (tr, ph, wd) in enumerate(chosen[:n_bases]):
        i = int(ph[int(rng.integers(len(ph)))])
        t = tr.downbeats_s[i]
        common = {"a": tr.slug, "b": tr.slug, "a_t": t, "b_t": t, "fold": tr.fold, "base": b, "a_bar": i}
        out.append({**common, "kind": "natural", "family": "base", "level": 0.0, "fade": "bar"})
        sign = 1.0 if b % 2 == 0 else -1.0
        for lv in DOSE_LEVELS["phase"][1:]:
            out.append({**common, "kind": "dose_phase", "family": "phase", "level": lv, "fade": "bar", "delta_ms": lv})
        for lv in DOSE_LEVELS["gain"][1:]:
            out.append({**common, "kind": "dose_gain", "family": "gain", "level": lv, "fade": "cut", "gain_db": sign * lv})
        for lv in DOSE_LEVELS["key"]:
            out.append({**common, "kind": "dose_key", "family": "key", "level": float(lv), "fade": "bar", "vocoder": True,
                        "semitones": CAMELOT_STEP_SEMITONES[lv]})
        far = [j for j in ph if abs(j - i) >= 16]
        w = int(wd[int(rng.integers(len(wd)))])
        if far:
            j = int(far[int(rng.integers(len(far)))])
            out.append({**common, "kind": "jump_clean", "family": "word", "level": 0.0, "fade": "cut", "b_t": tr.downbeats_s[j], "b_bar": j})
            out.append({**common, "kind": "word_cut", "family": "word", "level": 1.0, "fade": "cut", "a_t": tr.downbeats_s[w],
                        "a_bar": w, "b_t": tr.downbeats_s[j], "b_bar": j})
    for r in out:
        r["id"] = f"d{r['base']:02d}_{r['family']}_{r['level']:g}_{r['a'][:12]}"
        r["label"] = 1 if r["kind"] in ("natural", "jump_clean") else 0
        r["long"] = True
    return out


def monotonicity(levels: Sequence[float], goodness: Sequence[float]) -> dict:
    """Spearman tra intensità e bontà (atteso negativo) e quota di passi che non migliorano."""
    from scipy.stats import spearmanr

    lv = np.asarray(levels, dtype=float)
    g = np.asarray(goodness, dtype=float)
    ok = np.isfinite(g)
    lv, g = lv[ok], g[ok]
    if len(g) < 3:
        return {"n": int(len(g)), "rho": float("nan"), "frac_non_increasing_steps": float("nan")}
    o = np.argsort(lv)
    d = np.diff(g[o])
    return {"n": int(len(g)), "rho": float(spearmanr(lv, g).statistic),
            "frac_non_increasing_steps": float(np.mean(d <= 0)), "drop_0_to_max": float(g[o][0] - g[o][-1])}


# ============================================ (e) mestiere DJ: misure DSP (v3)
# Misure dirette delle regole del mestiere (REVISIONE_DJ_2026-10-20), senza
# addestramento: salto di loudness, fase dei battiti attraverso la giunta,
# coerenza tonale (chroma), voce attiva sul taglio (Silero VAD). Ogni misura è
# una *distanza* (più alta = peggio); la bontà è 1 - percentile rispetto alle
# giunte naturali sintetiche. Sono rilevatori dei difetti che definiscono il
# mestiere, quindi la loro AUC sintetica è attesa alta per costruzione: non è
# evidenza di gusto.
SR_CRAFT = 22050
CRAFT_VERSION = 2  # 2: fase dei battiti da Beat This! (1: solo DSP, troppo rumorosa)
CRAFT_SIDE_S = 9.5
CRAFT_GUARD_S = 0.12
CRAFT_FEATURES = ("loud_jump_db", "beat_phase_ms", "chroma_shift_gain", "chroma_dist", "vad_at_cut")
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_SILERO: dict = {}


def _st_loudness(x: np.ndarray, sr: int) -> float:
    """Loudness K-pesata non gated di tutto il segmento (BS.1770, pyloudnorm)."""
    import pyloudnorm

    if len(x) < int(0.4 * sr):
        return float("nan")
    meter = pyloudnorm.Meter(sr, block_size=len(x) / sr)
    with np.errstate(divide="ignore"):
        v = float(meter.integrated_loudness(np.asarray(x, dtype=np.float64)))
    return v if np.isfinite(v) else -70.0


def _beat_phase(env: np.ndarray, fps: float, t0: float, period: float) -> tuple[float, float]:
    """Fase (rad) e coerenza del pettine di periodo ``period`` sull'inviluppo d'attacco."""
    t = t0 + np.arange(len(env)) / fps
    e = env - env.mean()
    z = np.sum(np.maximum(e, 0) * np.exp(-2j * np.pi * t / period))
    return float(np.angle(z)), float(np.abs(z) / (np.sum(np.maximum(e, 0)) + 1e-9))


def _beat_period(env: np.ndarray, fps: float, lo: float = 0.42, hi: float = 1.0) -> float:
    """Periodo del battito (s) dall'autocorrelazione dell'inviluppo (60-143 bpm, rap)."""
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    l0, l1 = int(lo * fps), min(int(hi * fps) + 1, len(ac) - 1)
    if l1 <= l0 + 2:
        return float("nan")
    k = l0 + int(np.argmax(ac[l0:l1]))
    if 1 <= k < len(ac) - 1:  # interpolazione parabolica
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        k = k + (0.5 * (a - c) / den if den != 0 else 0.0)
    return float(k / fps)


def _refine_period(env: np.ndarray, fps: float, period: float, span: float = 0.04, steps: int = 81) -> float:
    """Affina il periodo massimizzando la coerenza del pettine (griglia fine ±4%)."""
    if not np.isfinite(period):
        return period
    grid = period * np.linspace(1 - span, 1 + span, steps)
    coh = [_beat_phase(env, fps, 0.0, p)[1] for p in grid]
    return float(grid[int(np.argmax(coh))])


def _key_profile_corr(chroma: np.ndarray) -> np.ndarray:
    """Correlazione con i 24 profili Krumhansl-Kessler (12 maggiori, 12 minori)."""
    out = []
    for prof in (_KK_MAJOR, _KK_MINOR):
        for k in range(12):
            out.append(np.corrcoef(chroma, np.roll(prof, k))[0, 1])
    return np.nan_to_num(np.array(out))


def _silero_probs(x16: np.ndarray) -> np.ndarray:
    """Probabilità di voce per frame di 32 ms (Silero VAD, pesi del pacchetto)."""
    import torch

    if "m" not in _SILERO:
        from silero_vad import load_silero_vad

        _SILERO["m"] = load_silero_vad(onnx=False)
    m = _SILERO["m"]
    m.reset_states()
    n = 512
    probs = []
    with torch.inference_mode():
        for i in range(0, len(x16) - n + 1, n):
            probs.append(float(m(torch.from_numpy(np.ascontiguousarray(x16[i:i + n], dtype=np.float32)), 16000).item()))
    return np.array(probs)


def craft_features(x: np.ndarray, sr: int, start_s: float, end_s: Optional[float] = None, *,
                   side_s: float = CRAFT_SIDE_S, guard_s: float = CRAFT_GUARD_S, vad: bool = True,
                   beat_tracker: Any = "auto") -> dict:
    """Distanze del mestiere DJ attorno a una giunta (mono). Più alto = peggio.

    ``start_s``/``end_s``: inizio e fine della sovrapposizione (uguali per un
    cut). Lato "prima" = [start - side, start - guard] (solo A), lato "dopo" =
    [end + guard, end + side] (solo B).

    * ``loud_jump_db``: |loudness K-pesata dopo − prima| (BS.1770, 3 s).
    * ``beat_phase_ms``: |sfasamento| del pettine dei battiti dopo la giunta
      rispetto a quello prima, con il periodo stimato sul lato prima
      (inviluppo d'attacco librosa, 86 fps); ridotto a ±metà battito.
    * ``chroma_shift_gain``: quanto una trasposizione del chroma medio "dopo"
      spiega il "prima" meglio della trasposizione nulla (0 = stessa tonalità).
    * ``chroma_dist``: 1 − correlazione dei profili tonali (Krumhansl-Kessler)
      prima/dopo.
    * ``vad_at_cut``: probabilità di voce (Silero VAD) negli ultimi 160 ms
      prima dell'uscita di A (parola tagliata se alta).
    """
    import librosa

    end_s = start_s if end_s is None else end_s
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = resample(x, sr, SR_CRAFT) if sr != SR_CRAFT else x
    srr = SR_CRAFT
    a0, a1 = max(0.0, start_s - side_s), start_s - guard_s
    b0, b1 = end_s + guard_s, min(len(y) / srr, end_s + side_s)
    if a1 - a0 < 1.5 or b1 - b0 < 1.5:
        raise ValueError("lati troppo corti per le misure del mestiere")
    pre = y[int(a0 * srr): int(a1 * srr)]
    post = y[int(b0 * srr): int(b1 * srr)]
    out: dict[str, Any] = {"craft_version": CRAFT_VERSION}
    # loudness (3 s accanto alla giunta)
    lp = _st_loudness(pre[-int(3 * srr):], srr)
    lq = _st_loudness(post[: int(3 * srr)], srr)
    out["loud_pre_lufs"], out["loud_post_lufs"] = lp, lq
    out["loud_jump_db"] = float(abs(lq - lp))
    # fase dei battiti: periodo di ciascun lato stimato sul lato lungo (autocorrelazione + affinamento),
    # fase di ciascun lato misurata vicino alla giunta (3 s) con il proprio periodo e riferita all'istante
    # della giunta; sfasamento = differenza delle due griglie estrapolate alla giunta (come la retta di
    # Beat This! in dj_qa, ma solo DSP). Se un lato è a tempo doppio/metà, si confronta al periodo minore.
    hop = 256
    fps = srr / hop
    env_pre = librosa.onset.onset_strength(y=pre, sr=srr, hop_length=hop)
    env_post = librosa.onset.onset_strength(y=post, sr=srr, hop_length=hop)
    pa = _refine_period(env_pre, fps, _beat_period(env_pre, fps))
    pb = _refine_period(env_post, fps, _beat_period(env_post, fps))
    out["beat_period_s"] = pa
    if np.isfinite(pa) and np.isfinite(pb):
        r = pb / pa
        p = pa
        if abs(r / 2 - 1) < 0.08 or abs(r * 2 - 1) < 0.08:
            p = min(pa, pb)
        near = int(3.0 * fps)
        # tempi rispetto alla giunta: il lato prima finisce a start - guard, il dopo inizia a end + guard
        t_pre0 = (a1 - start_s) - min(near, len(env_pre)) / fps
        ph_a, coh_a = _beat_phase(env_pre[-near:], fps, t_pre0, pa if p == pa else p)
        ph_b, coh_b = _beat_phase(env_post[:near], fps, b0 - end_s, pb if p == pb else p)
        # fase in secondi della griglia di ciascun lato all'istante della giunta (0)
        ta = (-ph_a / (2 * np.pi)) * (pa if p == pa else p)
        tb = (-ph_b / (2 * np.pi)) * (pb if p == pb else p)
        d = ((tb - ta) / p + 0.5) % 1.0 - 0.5
        out["beat_phase_ms"] = float(abs(d) * p * 1000.0)
        out["beat_coherence"] = float(min(coh_a, coh_b))
        out["tempo_ratio"] = float(r)
    else:
        out["beat_phase_ms"] = float("nan")
    out["beat_phase_source"] = "onset_dsp"
    # misura principale: Beat This! (Foscarin et al., ISMIR 2024; checkpoint del progetto, via dj_qa)
    # sulla clip, rette dei battiti nei lati A-solo e B-solo estrapolate alla giunta (come dj_qa
    # beat_this_mix). Il valore DSP resta come ripiego se il tracker manca o le rette non sono dritte.
    if beat_tracker:
        try:
            from musicmix import dj_qa

            tr = dj_qa._get_tracker("auto") if beat_tracker == "auto" else beat_tracker
            if tr is not None:
                # contesto del tracker: 6,5 s per lato (costo lineare nella durata)
                c0 = max(0.0, start_s - 6.5)
                c1 = min(len(y) / srr, end_s + 6.5)
                bt = tr(np.asarray(y[int(c0 * srr): int(c1 * srr)], dtype=np.float32), srr)
                beats = np.asarray(bt["beats"], float) + c0
                fa = dj_qa._fit_beat_line(beats[(beats >= a0) & (beats <= a1)])
                fb = dj_qa._fit_beat_line(beats[(beats >= b0) & (beats <= b1)])
                if fa and fb and max(fa["rms_resid_ms"], fb["rms_resid_ms"]) <= 20.0:
                    ka, ca, kb, cb = fa["period_s"], fa["intercept_s"], fb["period_s"], fb["intercept_s"]
                    rr = max(ka, kb) / min(ka, kb)
                    folded = abs(rr / 2.0 - 1.0) <= 0.08
                    pp = min(ka, kb) if folded else ka
                    pha = ((start_s - ca) / (pp if folded else ka)) % 1.0
                    phb = ((end_s - cb) / (pp if folded else kb)) % 1.0
                    off = ((phb - pha + 0.5) % 1.0 - 0.5) * pp * 1000.0
                    out["beat_phase_ms"] = float(abs(off))
                    out["beat_phase_source"] = "beat_this"
                    out["bt_tempo_change_pct"] = float(100.0 * (ka / kb - 1.0)) if not folded else None
                    out["bt_rms_resid_ms"] = float(max(fa["rms_resid_ms"], fb["rms_resid_ms"]))
                else:
                    out["beat_phase_source"] = "onset_dsp (Beat This! rette non affidabili)"
        except Exception as e:  # noqa: BLE001
            out["beat_phase_source"] = f"onset_dsp (Beat This! errore {type(e).__name__})"
    # tonalità
    c_pre = librosa.feature.chroma_stft(y=pre, sr=srr, n_fft=4096, hop_length=2048).mean(axis=1)
    c_post = librosa.feature.chroma_stft(y=post, sr=srr, n_fft=4096, hop_length=2048).mean(axis=1)
    corr = np.array([np.corrcoef(c_pre, np.roll(c_post, k))[0, 1] for k in range(12)])
    corr = np.nan_to_num(corr)
    out["chroma_corr0"] = float(corr[0])
    out["chroma_shift_gain"] = float(max(0.0, corr.max() - corr[0]))
    out["chroma_best_shift"] = int(np.argmax(corr) if np.argmax(corr) <= 6 else np.argmax(corr) - 12)
    kp, kq = _key_profile_corr(c_pre), _key_profile_corr(c_post)
    out["chroma_dist"] = float(1.0 - np.corrcoef(kp, kq)[0, 1])
    # melodia continua su entrambi i lati (Brewster & Broughton, How to DJ Right, cap. 11 "How to cut":
    # più melodia continua c'è ai due lati della giunta, meno è probabile che un cut suoni bene).
    # Quota armonica HPSS (librosa) sui 3 s accanto alla giunta; solo diagnostica (nessuna etichetta sintetica).
    def _harm_ratio(seg):
        sp = np.abs(librosa.stft(seg, n_fft=2048, hop_length=512)) ** 2
        h, pc = librosa.decompose.hpss(sp)
        return float(h.sum() / (h.sum() + pc.sum() + 1e-12))
    out["harm_ratio_pre"] = _harm_ratio(pre[-int(3 * srr):])
    out["harm_ratio_post"] = _harm_ratio(post[: int(3 * srr)])
    out["cut_melodic_risk"] = float(min(out["harm_ratio_pre"], out["harm_ratio_post"])) if end_s - start_s < 0.5 else None
    # voce sul taglio (lato A, ultimi 160 ms prima dell'uscita)
    if vad:
        try:
            s0 = max(0.0, start_s - 2.0)
            x16 = resample(y[int(s0 * srr): int(start_s * srr)], srr, 16000)
            pr = _silero_probs(x16)
            k = max(1, int(round(0.16 / 0.032)))
            out["vad_at_cut"] = float(pr[-k:].max()) if len(pr) else float("nan")
            out["vad_pre_mean"] = float(pr.mean()) if len(pr) else float("nan")
        except Exception as e:  # noqa: BLE001
            out["vad_at_cut"] = float("nan")
            out["vad_error"] = f"{type(e).__name__}: {e}"
    return out


# Soglie dei flag (veto tecnico). Origine: dj_qa (loudness 2/3 dB [H]; ritmo
# 15 ms [U/H] per i deck, qui allargata a 40 ms perché la misura solo-clip ha
# un errore di circa 25 ms sulle giunte naturali); voce: probabilità Silero.
CRAFT_FLAG_RULES = {
    "loud_jump_db": (3.0, "salto di loudness > 3 dB attorno alla giunta"),
    "beat_phase_ms": (40.0, "battiti di B sfasati di oltre 40 ms rispetto alla griglia di A"),
    "chroma_dist": (1.0, "profili tonali prima/dopo poco compatibili (possibile scontro di tonalità)"),
    "vad_at_cut": (0.9, "voce probabilmente attiva sul taglio di A (indicativo: Silero non è informativo nella dose-risposta)"),
}


def craft_flags(feats: dict, rules: Optional[dict] = None, *, is_cut: bool = False) -> list[dict]:
    """Flag di veto dalle misure del mestiere (solo segnalazioni, non un voto).

    Con un cut netto il flag tonale non si applica: lo scontro di tonalità si
    sente quando i due brani suonano insieme (Brewster & Broughton, How to DJ
    Right, cap. 15: "Cut them, don't blend them"; Webber, DJ Skills p. 238).
    """
    rules = rules or CRAFT_FLAG_RULES
    out = []
    for k, (thr, msg) in rules.items():
        if is_cut and k == "chroma_dist":
            continue
        v = feats.get(k)
        if v is not None and np.isfinite(v) and v > thr:
            out.append({"flag": k, "value": float(v), "threshold": thr, "message": msg})
    return out


def craft_goodness(feats: dict, nulls: dict) -> dict:
    """Bontà per misura = 1 − percentile della distanza tra le giunte naturali sintetiche."""
    out = {}
    for k in CRAFT_FEATURES:
        v = feats.get(k)
        out[k] = 1.0 - empirical_percentile(v, nulls.get(k, [])) if v is not None and np.isfinite(v) and nulls.get(k) else float("nan")
    return out


def fit_nonneg_logistic(x: np.ndarray, y: np.ndarray, *, c: float = 1.0) -> LinearSeamModel:
    """Logistica con coefficienti >= 0 su feature standardizzate di "bontà".

    Vincolo: ogni critico può solo aumentare la probabilità di giunta
    plausibile quando la sua bontà aumenta (nessun critico entra "invertito"
    per adattamento ai dati). L2 con peso 1/C, classi bilanciate.
    """
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = x.mean(0), x.std(0)
    sd[sd < 1e-9] = 1.0
    z = (x - mu) / sd
    w_pos = 0.5 / max(y.mean(), 1e-9)
    w_neg = 0.5 / max(1 - y.mean(), 1e-9)
    sw = np.where(y > 0.5, w_pos, w_neg)
    d = z.shape[1]

    def f(th):
        w, b = th[:d], th[d]
        m = z @ w + b
        lo = np.logaddexp(0, -m) * y + np.logaddexp(0, m) * (1 - y)
        p = 1 / (1 + np.exp(-m))
        g = (p - y) * sw
        return float(np.sum(sw * lo) / len(y) + 0.5 / (c * len(y)) * w @ w), \
            np.concatenate([z.T @ g / len(y) + w / (c * len(y)), [g.sum() / len(y)]])

    r = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", bounds=[(0, None)] * d + [(None, None)])
    return LinearSeamModel(mu, sd, r.x[:d], float(r.x[d]), None, None, {"C": c, "nonneg": True})


# ======================================================================= API
def _item_from_request(req: dict, idx: int, workdir: Path) -> TransitionItem:
    """Costruisce un item da una richiesta dell'API (audio come percorso o array)."""
    import soundfile as sf

    clip = req["clip"]
    if isinstance(clip, (str, os.PathLike)):
        wav = str(clip)
    else:
        arr = np.asarray(clip, dtype=np.float32)
        wav = str(workdir / f"clip_{idx:03d}.wav")
        sf.write(wav, arr, int(req.get("sample_rate", 48000)), subtype="FLOAT")
    meta = req.get("metadata") or {}
    sources = req.get("sources") or {}
    item_id = str(req.get("id", f"req_{idx:03d}")).replace("/", "_")
    if "overlap" in meta or "switch" in meta:
        it = item_from_render(meta, item_id=f"api/{item_id}", dataset="api", wav=wav, qa=req.get("qa"))
    else:
        dur = sf.info(wav).duration
        sw = float(meta.get("switch_s", dur / 2))
        it = TransitionItem(item_id=f"api/{item_id}", dataset="api", wav=wav, clip_offset_s=0.0, clip_dur_s=dur,
                            start_s=float(meta.get("start_s", sw)), switch_s=sw, end_s=float(meta.get("end_s", sw)),
                            kind=str(meta.get("kind", "unknown")), qa=req.get("qa"))
    for side in ("a", "b"):
        src = sources.get(side) or {}
        if src.get("audio"):
            setattr(it, f"{side}_audio", str(src["audio"]))
        if src.get("switch_s") is not None:
            setattr(it, f"{side}_switch_s", float(src["switch_s"]))
        if src.get("v12_track") is not None:
            it.meta[f"{side}_v12_track"] = bool(src["v12_track"])
    it.wav_sha256 = sha256_file(wav)
    return it


def _api_result(item_id: str, r: dict, mode: str, timings: dict) -> dict:
    return {"id": item_id, "score": r["score"], "uncertainty": r["uncertainty"], "mode": mode,
            "parts": {"goodness": r.get("goodness", {}), "craft": r.get("craft", {}),
                      "contributions_logit": r.get("contributions_logit"), "score_fast": r.get("score_fast"),
                      "missing": r.get("missing", [])},
            "flags": r.get("flags", []), "timings_s": timings, "version": DJ_CRITIC_VERSION + "+v3",
            "semantics": ("P(giunta plausibile) da un ensemble con pesi >= 0 stimato su giunte sintetiche (CV per brano); "
                          "affidabile come veto tecnico (volume, fase, voce), non come voto di gusto")}


def _v4_block(it: TransitionItem, req: dict, r: dict, *, nll_dir: Optional[Path] = None) -> dict:
    """Canali v4 dei libri (``musicmix.dj_critic_books``) per una richiesta, additivi al risultato v3."""
    from musicmix import dj_critic_books as B

    x, sr = load_audio(it.wav, start_s=it.clip_offset_s, duration_s=it.clip_dur_s, sr=SR_CRAFT)
    decks = None
    if req.get("decks"):
        dsr = int(req.get("sample_rate", 48000))
        decks = {k: (resample(np.asarray(v, dtype=np.float32), dsr, SR_CRAFT) if isinstance(v, np.ndarray) else v)
                 for k, v in req["decks"].items() if k != "gains"}
    nll = ent = None
    off = 0.0
    if nll_dir is not None:
        p = nll_dir / (it.item_id.replace("/", "__") + ".npz")
        if p.exists():
            z = np.load(p)
            nll, ent, off = z["nll"], z["ent"], float(z["offset_s"])
    v4 = B.score_v4(x, sr, req.get("metadata") or {}, start_s=it.start_s, switch_s=it.switch_s, end_s=it.end_s,
                    craft=r.get("craft") or None, qa=req.get("qa"), decks=decks, nll=nll, entropy=ent, nll_offset_s=off)
    v4["summary"] = B.summary_v4(v4)
    return v4


def score_transitions(requests: Sequence[dict], *, models_dir: Optional[os.PathLike] = None, workdir: Optional[os.PathLike] = None,
                      threads: int = 2, keep: bool = False, mode: str = "full", version: str = "v3",
                      content_surprise: bool = False) -> list[dict]:
    """Valuta un lotto di transizioni con il critico v3.

    Ogni richiesta: ``{"clip": path | array (frames, ch), "sample_rate": int,
    "metadata": metadati dj_render/dj_effects oppure {"start_s","switch_s","end_s"},
    "sources": {"a": {"audio": path, "switch_s": s}, "b": {...}}, "id": str}``.

    ``mode="fast"``: solo misure DSP del mestiere nel processo corrente (circa
    1 s per clip, nessuna rete neurale). ``mode="full"``: anche la giunta (c)
    su embedding (e la sorpresa se l'ensemble la usa) tramite
    ``tools/run_dj_critic.py api`` in due venv; ricarica i pesi a ogni
    chiamata (decine di secondi), quindi conviene passare lotti.
    Restituisce ``{score, uncertainty, parts, flags, timings_s, ...}`` per
    richiesta, nello stesso ordine. Non è un giudizio di gusto: vedi
    runs/dj_critic_v1/REPORT.md.

    ``version="v4"`` (additivo, 21/10/2026): punteggio, flag e campi v3 identici, più un campo ``v4`` con i
    canali dei libri (``musicmix.dj_critic_books``) in due blocchi separati, ``technical`` (veto/integrità)
    e ``musical`` (diagnostica, mai veto). Facoltativi nella richiesta: ``"decks"`` (deck separati alla
    ``sample_rate``, per backbeat/tessitura) e ``"qa"`` (report dj_qa, per il dropout). Con
    ``content_surprise=True`` calcola anche NLL ed entropia di MusicGen (venv-np1, >= 6 GB liberi).
    """
    if version not in ("v3", "v4"):
        raise ValueError("version: 'v3' o 'v4'")
    import shutil
    import subprocess
    import tempfile
    import time as _time

    models_dir = Path(models_dir or ROOT / "runs/dj_critic_v1")
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="dj_critic_api_"))
    wd.mkdir(parents=True, exist_ok=True)
    t0 = _time.time()
    items = [_item_from_request(r, i, wd) for i, r in enumerate(requests)]
    out = []
    if mode == "fast":
        v3 = json.loads((models_dir / "models/v3_models.json").read_text())
        fast = v3["fast"]
        models = [LinearSeamModel.from_json(m) for m in fast["models"]]
        for it in items:
            t1 = _time.time()
            x, sr = load_audio(it.wav, start_s=it.clip_offset_s, duration_s=it.clip_dur_s, sr=SR_CRAFT)
            side = min(CRAFT_SIDE_S, it.start_s - 0.2, it.clip_dur_s - it.end_s - 0.2)
            cf = craft_features(x, sr, it.start_s, it.end_s, side_s=side)
            g = {f"craft_{k}": v for k, v in craft_goodness(cf, v3["craft_nulls"]).items()}
            vals = [g.get(f) for f in fast["features"]]
            if any(v is None or not np.isfinite(v) for v in vals):
                score, unc = float("nan"), float("nan")
            else:
                ps = [float(m.proba(np.array([vals]))[0]) for m in models]
                score, unc = ps[0], float(np.std(ps[1:]))
            m0 = models[0]
            contrib = {f: float(m0.coef[j] * (g[f] - m0.mean[j]) / m0.scale[j]) for j, f in enumerate(fast["features"])
                       if np.isfinite(g.get(f, np.nan))}
            r = {"score": score, "uncertainty": unc, "goodness": g, "craft": cf, "contributions_logit": contrib,
                 "flags": craft_flags(cf, {k: tuple(v) for k, v in v3.get("flag_rules", CRAFT_FLAG_RULES).items()},
                                      is_cut=(it.end_s - it.start_s) < 0.5)}
            out.append(_api_result(it.item_id.split("/", 1)[1], r, "fast", {"total": _time.time() - t1}))
    else:
        write_json(wd / "items.json", {"schema_version": DJ_CRITIC_VERSION, "items": [i.to_json() for i in items]})
        py = python_main()
        subprocess.run([py, str(ROOT / "tools/run_dj_critic.py"), "api", "--out", str(wd), "--threads", str(threads),
                        "--models-dir", str(models_dir)], check=True)
        sc = json.loads((wd / "scores_v3.json").read_text())["items"]
        tim = json.loads((wd / "api_timings.json").read_text()) if (wd / "api_timings.json").exists() else {}
        tot = _time.time() - t0
        for it in items:
            out.append(_api_result(it.item_id.split("/", 1)[1], sc[it.item_id], "full",
                                   {**tim, "total_batch": tot, "per_item": tot / max(1, len(items))}))
    if version == "v4":
        nll_dir = None
        if content_surprise:
            nll_dir = wd / "musicgen_ent"
            write_json(wd / "items_v4.json", {"schema_version": DJ_CRITIC_VERSION, "items": [i.to_json() for i in items]})
            py1 = python_np1()
            subprocess.run([py1, str(ROOT / "tools/run_dj_critic_v4.py"), "musicgen", "--out", str(wd), "--threads", str(threads),
                            "--items-json", str(wd / "items_v4.json"), "--cache-dir", str(nll_dir)], check=True)
        for it, req, res in zip(items, requests, out):
            t1 = _time.time()
            res["v4"] = _v4_block(it, req, {"craft": (res.get("parts") or {}).get("craft")}, nll_dir=nll_dir)
            res["version_v4"] = res["v4"]["version"]
            res.setdefault("timings_s", {})["v4"] = _time.time() - t1
    if not keep and not workdir:
        shutil.rmtree(wd, ignore_errors=True)
    return out


def score_transition(clip_audio, metadata: dict, sources: Optional[dict] = None, *, mode: str = "full", **kw) -> dict:
    """``score_transition(clip_audio, metadata, sources) -> {score, uncertainty, parts, flags, timings_s}``.

    ``clip_audio``: percorso WAV o array (frames, canali) a ``kw['sample_rate']``
    (default 48000). ``metadata``: metadati di ``dj_render``/``dj_effects`` o
    ``{"start_s", "switch_s", "end_s"}`` (secondi nella clip). ``mode``:
    "fast" (solo DSP, ~1 s) o "full" (con la giunta su embedding). Per
    confrontare più tecniche usare ``score_transitions`` con un solo lotto.
    ``version="v4"`` aggiunge il campo ``v4`` (canali dei libri; ``decks=``, ``qa=`` facoltativi); il
    default ``"v3"`` restituisce esattamente quanto prima.
    """
    req = {"clip": clip_audio, "metadata": metadata, "sources": sources or {}, "sample_rate": kw.pop("sample_rate", 48000),
           "id": kw.pop("id", "single")}
    for k in ("decks", "qa"):  # facoltativi, usati solo da version="v4"
        if kw.get(k) is not None:
            req[k] = kw.pop(k)
        else:
            kw.pop(k, None)
    return score_transitions([req], mode=mode, **kw)[0]
