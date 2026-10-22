"""Supervisione dai brani stessi (insong_v1): dataset di punti naturali/spostati, cache, modelli, controlli.

Fasi (``python tools/build_insong_dataset.py <fase>``); output in ``runs/insong_v1/``:

  points    [venv]      brani ammessi (catalogo del ponte + V12, track_filter, esclusioni), punti e giunte
  dsp       [venv]      cache DSP per brano (energia per banda, flusso, voce dallo stem; float16, 50 Hz)
  mert      [venv-np1]  cache MERT-v1-95M per brano (layer 6 e 12, media su 0,16 s, float16)
  cocola    [venv-np1]  COCOLA_HP_v1 (Ciranni et al. 2024): coerenza voce/strumentale allineata e spostata
  train     [venv]      modelli dei punti, leave-one-artist-out, AUC per tipo di negativo; modelli salvati
  pairs     [venv]      giunte tra brani e controllo "stesso brano vs brani diversi"
  external  [venv]      controllo esterno sui voti ciechi v3 (mai usati per addestrare)
  cues      [venv]      cache ``cue_scores/<slug>.json`` per l'API ``self_supervision.cue_naturalness``
  fetch-cocola          scarica il checkpoint ufficiale (Google Drive del README) e verifica lo sha256

Vincoli della macchina condivisa: un modello alla volta, solo con >= 6 GB disponibili; niente audio intermedio salvato;
fermarsi sotto 2 GB di disco libero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix import self_supervision as S  # noqa: E402
from musicmix import track_filter as TF  # noqa: E402
from musicmix.paths import install_home, localize, models_dir  # noqa: E402

OUT = S.RUN_DIR
CATALOG = ROOT / "runs/pool_bridge_v1/catalog_ready.json"
MODELS = models_dir()
MERT_DIR = MODELS / "hf/hub/models--m-a-p--MERT-v1-95M/snapshots/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
COCOLA_CKPT = MODELS / "cocola" / "COCOLA_HP_v1.ckpt"
COCOLA_SHA256 = "951b70bc787fcac489208a6b665d0d4e1375d6c0511a91eedad2814986bfb0aa"
COCOLA_URL = "https://drive.usercontent.google.com/download?id=1S-_OvnDwNFLNZD5BmI1Ouck_prutRVWZ&export=download&confirm=t"
COCOLA_PYDEPS = Path(os.environ.get("MUSICMIX_COCOLA_PYDEPS", str(install_home() / "pydeps/cocola")))  # efficientnet_pytorch 0.7.1
COCOLA_REPO = {"url": "https://github.com/gladia-research-group/cocola", "revision": "b3ac9480cd1594d990a1d2dfabbab7cf15af8dfa",
               "license": "MIT (codice); checkpoint senza licenza separata, addestrato su MoisesDB (licenza non commerciale), "
                          "Slakh2100 e CocoChorales (CC BY 4.0): uso di ricerca"}
#: brani esclusi dal dataset (oltre a track_filter/excluded_tracks): griglia non affidabile o modello dj_phrase assente
EXCLUDE = {"antidote": "tempo ambiguo d'ottava (65,6 vs 131 BPM): con la griglia a metà tempo i negativi ±battito sarebbero sbagliati "
                       "(stessa esclusione del set v3)"}
MIN_FREE_DISK_GB = 2.0
MIN_FREE_RAM_GB = 6.0


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p: Path, obj) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_js), encoding="utf-8")


def _js(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save_npz_atomic(path: Path, compressed: bool = False, **arrays) -> None:
    """Scrive l'npz in un file temporaneo e lo rinomina: un processo interrotto non lascia cache troncate."""
    tmp = Path(path).with_suffix(".tmp.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_disk() -> float:
    free = shutil.disk_usage("/").free / 1e9
    if free < MIN_FREE_DISK_GB:
        raise SystemExit(f"disco libero {free:.2f} GB < {MIN_FREE_DISK_GB} GB: mi fermo")
    return free


def avail_ram_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def wait_ram(need: float = MIN_FREE_RAM_GB, max_wait_s: float = 3600) -> None:
    t0 = time.time()
    while avail_ram_gb() < need:
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"RAM disponibile {avail_ram_gb():.1f} GB < {need} GB da {max_wait_s:.0f} s")
        log(f"RAM disponibile {avail_ram_gb():.1f} GB < {need} GB: attendo")
        time.sleep(60)


# =========================================================================== brani e punti
def select_tracks() -> tuple[list[S.TrackData], list[dict]]:
    cat = load(CATALOG)
    entries = {t["slug"]: t for t in cat["tracks"]}
    tf = TF.default_filter()
    kept, dropped = tf.split(list(entries), entries)
    skipped = [{"slug": d["slug"], "why": d["why"]} for d in dropped]
    tracks = []
    for slug in kept:
        e = entries[slug]
        if slug in EXCLUDE:
            skipped.append({"slug": slug, "why": EXCLUDE[slug]})
            continue
        if e.get("origin") not in ("pool_v12", "local_library") or e.get("status") != "ready":
            skipped.append({"slug": slug, "why": f"origine/stato {e.get('origin')}/{e.get('status')}"})
            continue
        pj = S.SOURCE_RUN / "phrase" / f"{slug}.json"
        tj = S.SOURCE_RUN / "dj_plan" / "tracks" / f"{slug}.json"
        if not (pj.is_file() and tj.is_file()):
            skipped.append({"slug": slug, "why": "analisi DJ o modello dj_phrase assenti in runs/dj_set_v3_bridge (stem vocale "
                                                 "assente sulla macchina di analisi o brano arrivato dopo il congelamento)"})
            continue
        tr = S.load_track(slug, artist=e.get("artist"))
        if not tr.meta.get("asr_reliable"):
            skipped.append({"slug": slug, "why": "trascrizione non affidabile per dj_phrase"})
            continue
        if not (tr.audio_path and Path(tr.audio_path).is_file() and tr.vocal_path and Path(tr.vocal_path).is_file()):
            skipped.append({"slug": slug, "why": "audio canonico o stem vocale assenti"})
            continue
        tracks.append(tr)
    return tracks, skipped


def tracks_by_slug() -> dict[str, S.TrackData]:
    doc = load(OUT / "tracks.json")
    return {r["slug"]: S.load_track(r["slug"], artist=r["artist"]) for r in doc["tracks"]}


def stage_points(seed: int = 0) -> None:
    tracks, skipped = select_tracks()
    rows = [{"slug": t.slug, "artist": t.artist, "group": t.group, "bpm": round(t.bpm, 2), "camelot": t.camelot,
             "duration_s": round(t.duration_s, 2), "audio": str(Path(t.audio_path).relative_to(ROOT)) if
             str(t.audio_path).startswith(str(ROOT)) else t.audio_path, "vocal_stem": t.vocal_path, "lang": t.meta.get("lang"),
             "n_lines": len(t.lines), "n_words": len(t.words)} for t in tracks]
    dump(OUT / "tracks.json", {"schema_version": S.SCHEMA + ".tracks", "catalog": str(CATALOG.relative_to(ROOT)),
                               "catalog_sha256": sha256(CATALOG), "source_run": str(S.SOURCE_RUN.relative_to(ROOT)),
                               "track_filter": TF.default_filter().describe(), "tracks": rows, "skipped": skipped})
    pts = S.build_points(tracks, seed=seed)
    pairs = S.build_pairs(pts, tracks, seed=seed)
    counts = {}
    for r in pts:
        k = "pos:" + r["kind"] if r["label"] else "neg:" + r["kind"]
        counts[k] = counts.get(k, 0) + 1
    pc = {}
    for r in pairs:
        k = f"{r['scope']}:{'pos' if r['label'] else r['side'] + ':' + r['kind']}"
        pc[k] = pc.get(k, 0) + 1
    dump(OUT / "points.json", {"schema_version": S.SCHEMA + ".points", "params": S.PARAMS, "neg_types": S.NEG_TYPES,
                               "seed": seed, "counts": counts, "points": pts})
    dump(OUT / "pairs.json", {"schema_version": S.SCHEMA + ".pairs", "seed": seed, "counts": pc, "pairs": pairs})
    groups = sorted({t.group for t in tracks})
    log(f"{len(tracks)} brani, {len(groups)} gruppi d'artista {groups}; esclusi {[s['slug'] for s in skipped]}")
    log(f"punti {len(pts)}: {json.dumps(counts, sort_keys=True)}")
    log(f"giunte {len(pairs)}: {json.dumps(pc, sort_keys=True)}")


# =========================================================================== audio
def read_mono(path: str, sr_out: int) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != sr_out:
        g = np.gcd(int(sr), int(sr_out))
        x = resample_poly(x, sr_out // g, sr // g).astype(np.float32)
    return x


def stage_dsp() -> None:
    import librosa

    check_disk()
    (OUT / "cache/dsp").mkdir(parents=True, exist_ok=True)
    for slug, tr in tracks_by_slug().items():
        p = OUT / "cache/dsp" / f"{slug}.npz"
        if p.is_file():
            continue
        sr, hop = 16000, int(16000 / S.DSP_FPS)
        x = read_mono(tr.audio_path, sr)
        v = read_mono(tr.vocal_path, sr)
        n = min(len(x), len(v))
        X = np.abs(librosa.stft(x[:n], n_fft=1024, hop_length=hop, center=True)) ** 2
        f = librosa.fft_frequencies(sr=sr, n_fft=1024)
        cols = []
        for lo, hi in S.DSP_BANDS:
            band = X[(f >= lo) & (f < hi)].sum(0)
            cols.append(10 * np.log10(band + 1e-10))
        logm = np.log1p(1000 * np.sqrt(X))
        flux = np.r_[0.0, np.maximum(np.diff(logm, axis=1), 0).sum(0)]
        cols.append(flux / (np.median(flux) + 1e-6))
        V = np.abs(librosa.stft(v[:n], n_fft=1024, hop_length=hop, center=True)) ** 2
        vdb = 10 * np.log10(V.sum(0) + 1e-10)
        cols.append(vdb - np.percentile(vdb, 95))
        mixdb = np.stack(cols[:4])
        ref = np.percentile(mixdb.max(0), 95)
        for i in range(4):
            cols[i] = cols[i] - ref
        arr = np.stack(cols, axis=1).astype(np.float16)
        save_npz_atomic(p, compressed=True, x=arr, fps=np.float32(S.DSP_FPS), cols=np.asarray(S.DSP_COLS))
        log(f"dsp {slug}: {arr.shape}")


def stage_chroma() -> None:
    """Croma (librosa chroma_stft, 10 fps, float16) per il "ritorno armonico" del compito tensione -> risoluzione."""
    import librosa

    check_disk()
    (OUT / "cache/chroma").mkdir(parents=True, exist_ok=True)
    for slug, tr in tracks_by_slug().items():
        p = OUT / "cache/chroma" / f"{slug}.npz"
        if p.is_file():
            continue
        x = read_mono(tr.audio_path, 16000)
        c = librosa.feature.chroma_stft(y=x, sr=16000, n_fft=4096, hop_length=1600).T
        save_npz_atomic(p, compressed=True, x=c.astype(np.float16), fps=np.float32(10.0))
        log(f"chroma {slug}: {c.shape}")


def stage_mert(threads: int = 3) -> None:
    import torch
    from transformers import AutoModel

    check_disk()
    todo = [s for s in tracks_by_slug() if not (OUT / "cache/mert" / f"{s}.npz").is_file()]
    if not todo:
        log("mert: tutte le cache presenti (riprendibile: si saltano i brani già fatti)")
        return
    wait_ram()
    torch.set_num_threads(threads)
    model = AutoModel.from_pretrained(str(MERT_DIR), trust_remote_code=True, local_files_only=True).eval()
    (OUT / "cache/mert").mkdir(parents=True, exist_ok=True)
    chunk, ctx, sr, fps = 30.0, 3.0, 24000, 75
    for slug, tr in tracks_by_slug().items():
        p = OUT / "cache/mert" / f"{slug}.npz"
        if p.is_file():
            continue
        check_disk()
        t0 = time.time()
        x = read_mono(tr.audio_path, sr)
        n_frames = int(len(x) / sr * fps)
        feats = {L: np.zeros((n_frames, 768), np.float32) for L in S.MERT_LAYERS}
        s = 0.0
        while s < len(x) / sr:
            a, b = max(0.0, s - ctx), min(len(x) / sr, s + chunk + ctx)
            seg = x[int(a * sr): int(b * sr)]
            seg = (seg - seg.mean()) / (seg.std() + 1e-7)
            with torch.inference_mode():
                hs = model(torch.from_numpy(np.ascontiguousarray(seg))[None], output_hidden_states=True).hidden_states
            k0 = int(round((s - a) * fps))
            i0, i1 = int(round(s * fps)), min(n_frames, int(round((s + chunk) * fps)))
            for L in S.MERT_LAYERS:
                h = hs[L][0].numpy()
                take = h[k0: k0 + (i1 - i0)]
                feats[L][i0: i0 + len(take)] = take
            s += chunk
        pool = 12
        m = n_frames // pool
        out = {f"l{L}": feats[L][: m * pool].reshape(m, pool, 768).mean(1).astype(np.float16) for L in S.MERT_LAYERS}
        save_npz_atomic(p, **out, hop_s=np.float32(S.MERT_HOP_S), model=np.asarray("m-a-p/MERT-v1-95M@12af15fe"))
        log(f"mert {slug}: {m} frame in {time.time() - t0:.0f} s")
        del feats, x


# =========================================================================== COCOLA
def fetch_cocola() -> None:
    COCOLA_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if not COCOLA_CKPT.is_file():
        import subprocess

        subprocess.run(["curl", "-sS", "-L", "-o", str(COCOLA_CKPT), COCOLA_URL, "--max-time", "600"], check=True)
    h = sha256(COCOLA_CKPT)
    if h != COCOLA_SHA256:
        raise SystemExit(f"sha256 di COCOLA diverso: {h}")
    log(f"COCOLA ok {COCOLA_CKPT} {h}")


class Cocola:
    """COCOLA_HP_v1 ricostruito dal codice ufficiale (EfficientNet-b0 su HPSS mel 64, proiezione 512, LayerNorm, tanh,
    similarità bilineare). Il checkpoint Lightning si legge con ``weights_only=True`` registrando solo gli enum."""

    def __init__(self, threads: int = 3):
        import enum
        import types

        import torch
        from torch import nn

        if str(COCOLA_PYDEPS) not in sys.path:
            sys.path.insert(0, str(COCOLA_PYDEPS))
        from efficientnet_pytorch import EfficientNet

        torch.set_num_threads(threads)
        self.torch = torch
        mod = types.ModuleType("contrastive_model.constants")

        # enum del modulo ufficiale contrastive_model/constants.py (solo valori: nessun codice del repository eseguito)
        spec = {
            "Dataset": {"CCS": "coco_chorales_contrastive/*", "CCS_RANDOM": "coco_chorales_contrastive/random",
                        "CCS_STRING": "coco_chorales_contrastive/string", "CCS_BRASS": "coco_chorales_contrastive/brass",
                        "CCS_WOODWIND": "coco_chorales_contrastive/woodwind", "SLAKH2100": "slakh2100_contrastive",
                        "MOISESDB": "moisesdb_contrastive", "MIXED": "mixed_contrastive"},
            "ModelInputType": {"SINGLE_CHANNEL_MEL_SPECTROGRAM": "single_channel_mel_spectrogram",
                               "DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE": "double_channel_harmonic_percussive"},
            "ModelFeatureExtractorType": {"HPSS": "hpss", "MEL_SPECTROGRAM": "mel_spectrogram"},
            "FeatureExtractionTime": {"OFFLINE": "offline", "ONLINE": "online"},
            "EmbeddingMode": {"HARMONIC": "harmonic", "PERCUSSIVE": "percussive", "BOTH": "both", "RANDOM": "random"},
        }
        enums = []
        for name, members in spec.items():
            E = enum.Enum(name, members, module="contrastive_model.constants", qualname=name)
            setattr(mod, name, E)
            enums.append(E)
        sys.modules.setdefault("contrastive_model", types.ModuleType("contrastive_model"))
        sys.modules["contrastive_model.constants"] = mod
        torch.serialization.add_safe_globals(enums)
        ck = torch.load(str(COCOLA_CKPT), map_location="cpu", weights_only=True)
        sd = ck["state_dict"]

        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.encoder = nn.Module()
                self.encoder.encoder.model = nn.Sequential(EfficientNet.from_name("efficientnet-b0", include_top=False, in_channels=2),
                                                           nn.Dropout(0.1), nn.Flatten())
                self.encoder.projection = nn.Linear(1280, 512)
                self.layer_norm = nn.LayerNorm(512)
                self.similarity = nn.Module()
                self.similarity.w = nn.Parameter(torch.zeros(512, 512))

        self.net = Enc()
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"COCOLA: chiavi mancanti {missing[:5]} inattese {unexpected[:5]}")
        self.net.eval()
        self.hparams = {k: (v.value if hasattr(v, "value") else v) for k, v in (ck.get("hyper_parameters") or {}).items()}

    @staticmethod
    def hpss_features(x16: np.ndarray) -> np.ndarray:
        import librosa

        st = librosa.stft(x16, n_fft=1024, win_length=400, hop_length=160)
        h, p = librosa.decompose.hpss(st)
        out = []
        for comp in (h, p):
            mel = librosa.feature.melspectrogram(S=np.abs(comp) ** 2, sr=16000, fmin=60.0, fmax=7800.0, n_mels=64)
            out.append(librosa.power_to_db(mel, ref=np.max))
        return np.stack(out).astype(np.float32)

    def embed(self, feats: np.ndarray, mode: str = "both") -> np.ndarray:
        """Embedding (tanh(LayerNorm)) per un batch di feature HPSS [B, 2, 64, W]; ``mode`` azzera un canale."""
        t = self.torch.from_numpy(np.ascontiguousarray(feats)).clone()
        if mode == "harmonic":
            t[:, 1] = 0
        elif mode == "percussive":
            t[:, 0] = 0
        with self.torch.inference_mode():
            e = self.net.encoder.encoder.model(t)
            e = self.net.encoder.projection(e)
            e = self.torch.tanh(self.net.layer_norm(e))
        return e.numpy()

    def score(self, ex: np.ndarray, ey: np.ndarray) -> np.ndarray:
        w = self.net.similarity.w.detach().numpy()
        return np.einsum("bi,ij,bj->b", ex, w, ey)


def stage_cocola(threads: int = 3, per_track: int = 20, cross_pairs: int = 60, seed: int = 0) -> None:
    """Coerenza voce/strumentale (stem): finestra di 5 s dello strumentale (mix − voce) da un downbeat con voce; la voce
    della stessa finestra (positivo) contro la voce spostata di ±½ battito, ±1 battito, ±2 battiti, ±1 battuta,
    +2 battute (negativi). Dentro lo stesso brano e tra brani diversi (tempo entro ±3 %, voce di Y allineata al
    downbeat di X). Registra anche i punteggi per la distanza Camelot (compatibilità armonica)."""
    check_disk()
    wait_ram(3.0)
    cc = Cocola(threads)
    rng = np.random.default_rng(seed)
    tracks = tracks_by_slug()
    sr, W = 16000, 5.0
    shifts = {"offbeat": 0.5, "beat": 1.0, "halfbar": 2.0, "bar": 4.0, "two_bars": 8.0}
    audio: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(slug):
        if slug not in audio:
            if len(audio) >= 3:
                audio.pop(next(iter(audio)))
            tr = tracks[slug]
            mix, voc = read_mono(tr.audio_path, sr), read_mono(tr.vocal_path, sr)
            n = min(len(mix), len(voc))
            audio[slug] = (mix[:n] - voc[:n], voc[:n])
        return audio[slug]

    def win(x, t):
        a = int(round(t * sr))
        seg = x[a: a + int(W * sr)]
        return seg if len(seg) == int(W * sr) else None

    def anchors(tr: S.TrackData):
        voc = get(tr.slug)[1]
        out = []
        for d in tr.downbeats_s:
            if d < 3 * tr.period_s * 4 or d + W + 8 * tr.period_s + 1 > tr.duration_s:
                continue
            seg = win(voc, d)
            if seg is None:
                continue
            fr = seg[: len(seg) // 400 * 400].reshape(-1, 400)
            rms = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
            if (rms > np.percentile(rms, 90) - 25).mean() >= 0.5 and np.percentile(rms, 90) > -45:
                out.append(float(d))
        return out

    rows = []
    t0 = time.time()
    anc = {}
    part = OUT / "cache" / "cocola_partial.jsonl"  # riprendibile: righe già calcolate per brano
    done: dict[str, list] = {}
    if part.is_file():
        for line in part.read_text().splitlines():
            d = json.loads(line)
            done[d["key"]] = d["rows"]
    for slug, tr in tracks.items():
        rng_t = np.random.default_rng([seed, sum(map(ord, slug))])
        if f"within:{slug}" in done and f"anchors:{slug}" in done:
            rows += done[f"within:{slug}"]
            anc[slug] = done[f"anchors:{slug}"]
            continue
        a = anchors(tr)
        anc[slug] = a
        pick = sorted(rng_t.choice(a, size=min(per_track, len(a)), replace=False)) if a else []
        n_before = len(rows)
        inst, voc = get(slug)
        for d in pick:
            wi = win(inst, d)
            feats, labels = [Cocola.hpss_features(wi)], []
            cand = [("aligned", 0.0)] + [(f"{k}{'+' if s > 0 else '-'}", s * v * tr.period_s) for k, v in shifts.items()
                                         for s in ((1,) if k == "two_bars" else (-1, 1))]
            keep = []
            for name, dt in cand:
                wv = win(voc, d + dt)
                if wv is None:
                    continue
                feats.append(Cocola.hpss_features(wv))
                keep.append((name, dt))
            F = np.stack(feats)
            sc = {}
            for mode in ("both", "harmonic", "percussive"):
                E = cc.embed(F, mode)
                sc[mode] = cc.score(np.repeat(E[:1], len(E) - 1, 0), E[1:])
            for i, (name, dt) in enumerate(keep):
                rows.append({"scope": "within", "x": slug, "y": slug, "t_x": d, "t_y": d + dt, "shift": name.rstrip("+-"),
                             "shift_s": dt, **{f"s_{m}": float(sc[m][i]) for m in sc}})
        with open(part, "a") as fh:
            fh.write(json.dumps({"key": f"within:{slug}", "rows": rows[n_before:]}, default=_js) + "\n")
            fh.write(json.dumps({"key": f"anchors:{slug}", "rows": a}, default=_js) + "\n")
        log(f"cocola within {slug}: {len(pick)} ancore ({time.time() - t0:.0f} s)")
    # tra brani
    slugs = sorted(tracks)
    cand_pairs = [(x, y) for x in slugs for y in slugs if x != y and tracks[x].group != tracks[y].group
                  and abs(tracks[y].bpm / tracks[x].bpm - 1) <= 0.03 and anc.get(x) and anc.get(y)]
    for k in rng.permutation(len(cand_pairs))[:cross_pairs]:
        x, y = cand_pairs[int(k)]
        key = f"cross:{x}>{y}:{int(k)}"
        if key in done:
            rows += done[key]
            continue
        n_before = len(rows)
        tx, ty = tracks[x], tracks[y]
        rk = np.random.default_rng([seed, int(k)])
        dx, dy = float(rk.choice(anc[x])), float(rk.choice(anc[y]))
        wi = win(get(x)[0], dx)
        vy = get(y)[1]
        feats, keep = [Cocola.hpss_features(wi)], []
        for name, dt in [("aligned", 0.0)] + [(f"{k2}{'+' if s > 0 else '-'}", s * v * ty.period_s) for k2, v in shifts.items()
                                              for s in ((1,) if k2 == "two_bars" else (-1, 1))]:
            wv = win(vy, dy + dt)
            if wv is None:
                continue
            feats.append(Cocola.hpss_features(wv))
            keep.append((name, dt))
        F = np.stack(feats)
        sc = {}
        for mode in ("both", "harmonic", "percussive"):
            E = cc.embed(F, mode)
            sc[mode] = cc.score(np.repeat(E[:1], len(E) - 1, 0), E[1:])
        for i, (name, dt) in enumerate(keep):
            rows.append({"scope": "cross", "x": x, "y": y, "t_x": dx, "t_y": dy + dt, "shift": name.rstrip("+-"), "shift_s": dt,
                         "camelot_x": tx.camelot, "camelot_y": ty.camelot, "tempo_ratio": ty.bpm / tx.bpm,
                         **{f"s_{m}": float(sc[m][i]) for m in sc}})
        with open(part, "a") as fh:
            fh.write(json.dumps({"key": key, "rows": rows[n_before:]}, default=_js) + "\n")
    log(f"cocola cross: {len(cand_pairs)} coppie candidate, {min(cross_pairs, len(cand_pairs))} usate ({time.time() - t0:.0f} s)")
    dump(OUT / "cocola_stems.json", {"schema_version": S.SCHEMA + ".cocola", "model": {"checkpoint": str(COCOLA_CKPT),
                                     "sha256": COCOLA_SHA256, "source": COCOLA_REPO, "hparams": cc.hparams},
                                     "window_s": W, "shifts_beats": shifts, "rows": rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "points":
        stage_points(a.seed)
    elif a.stage == "dsp":
        stage_dsp()
    elif a.stage == "chroma":
        stage_chroma()
    elif a.stage == "mert":
        stage_mert(a.threads)
    elif a.stage == "fetch-cocola":
        fetch_cocola()
    elif a.stage == "cocola":
        stage_cocola(a.threads, seed=a.seed)
    else:
        import insong_eval  # tools/insong_eval.py: train, pairs, external, cues

        insong_eval.main(a.stage, a)
