"""Critico automatico di transizioni DJ: run completa (runs/dj_critic_v1).

Fasi (ognuna riprendibile, risultati in ``--out``):

  inventory   [venv]      item da valutare (gallerie, set, V12, ricostruzioni
                          bit-esatte di V10/V8 con ``anchor_handoff_v1``)
  plan        [venv]      brani, fold per brano, ricette degli esempi sintetici
  muq         [venv-np1]  MuQ (embedding per (c)) + testa SongEval (a)
  musicgen    [venv-np1]  sorpresa MusicGen-small (b)
  audiobox    [venv]      Audiobox Aesthetics (a)
  qa          [venv]      dj_qa solo-clip per gli item senza report QA (d)
  train       [venv]      classificatore della giunta (c), CV per brano
  score       [venv]      punteggi dei critici + combinazione dichiarata
  validate    [venv]      confronto con i giudizi umani (n minuscolo)
  report      [venv]      REPORT.md

Esempio: ``python tools/run_dj_critic.py inventory plan`` poi
``$MUSICMIX_PY_NP1 tools/run_dj_critic.py muq musicgen`` (o ``all``, che
lancia ogni fase con l'interprete giusto).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from musicmix import dj_critic as C  # noqa: E402
from musicmix.paths import python_main, python_np1  # noqa: E402

PY_MAIN = python_main()
PY_NP1 = python_np1()
NP1_STAGES = {"muq", "musicgen", "emb"}
STAGES = ["inventory", "plan", "doseplan", "muq", "musicgen", "emb", "audiobox", "qa", "qa2", "craft", "train", "score", "validate", "train2", "score2", "dose", "validate2", "train3", "score3", "validate3", "report"]
GIUDIZI_STORICI = ROOT.parent / "giudizi json"
V12_TRACKS = ("5_percent_tint", "after_hours", "creature", "heartless", "mafia", "reminder", "too_many_nights")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# ============================================================ track registry
def analysis_files() -> list[Path]:
    files = sorted(glob.glob(str(ROOT / "runs/dj_plan_v1/tracks/*.json")))
    files += sorted(glob.glob(str(ROOT / "runs/dj_plan_cc_v1/tracks/*.json")))
    files += sorted(glob.glob(str(ROOT / ".cache/dj_set_v1/tracks/*.json")))
    seen, out = set(), []
    for f in files:
        if f.endswith(".key.json"):
            continue
        slug = Path(f).stem
        if slug in seen:
            continue
        seen.add(slug)
        out.append(Path(f))
    return out


def load_tracks() -> dict[str, C.TrackInfo]:
    tracks = {}
    for f in analysis_files():
        tr = C.load_track_info(f)
        if tr is not None and len(tr.downbeats_s) >= 32:
            tracks[tr.slug] = tr
    return tracks


def track_wav_by_id() -> dict[str, str]:
    """track_id (sha256 del WAV canonico) -> percorso locale."""
    out = {}
    for f in analysis_files():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        wav = (d.get("inputs") or {}).get("canonical_wav")
        if d.get("track_id") and wav and Path(wav).is_file():
            out[d["track_id"]] = wav
    for f in glob.glob(str(ROOT / "runs/*/audio/*.ingest.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        wav = f.replace(".ingest.json", ".wav")
        if d.get("track_id") and Path(wav).is_file():
            out.setdefault(d["track_id"], wav)
    for wav in glob.glob(str(ROOT / "runs/reminder_to_too_many_nights/audio/*.wav")):
        out.setdefault(C.sha256_file(wav), wav)
    return out


# =============================================================== inventory
def _gallery_items(run: str, dataset: str, labels: dict) -> list[C.TransitionItem]:
    items = []
    for sub in ("clips", "rejected"):
        for meta in sorted((ROOT / "runs" / run / sub).glob("*.json")):
            wav = meta.with_suffix(".wav")
            if not wav.is_file():
                continue
            d = json.loads(meta.read_text(encoding="utf-8"))
            r = d["render"]
            lab = labels.get(meta.stem)
            it = C.item_from_render(r, item_id=f"{run}/{meta.stem}", dataset=dataset if sub == "clips" else dataset + "_rejected",
                                    wav=str(wav), qa=d.get("qa"), group=(d.get("gallery") or {}).get("group", ""), label=lab)
            it.wav_sha256 = C.sha256_file(wav)
            it.meta["published"] = sub == "clips"
            items.append(it)
    return items


def _gallery_v1_labels() -> dict:
    out = {}
    for f in sorted((ROOT / "giudizi_cloud/dj_gallery_v1/judgments").glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        out[j["clip_id"]] = {"score": float(j["score"]), "scale": "1-10", "session": "dj_gallery_v1",
                             "source": rel(f), "note": j.get("note", ""), "source_sha256": j.get("source_sha256")}
    return out


def _set_items() -> list[C.TransitionItem]:
    run = ROOT / "runs/dj_set_v1b"
    s = json.loads((run / "set.json").read_text(encoding="utf-8"))
    wav = run / "set.wav"
    files = sorted((run / "transitions").glob("*.json"))
    items = []
    for t in s["transitions"]:
        idx = int(t["index"])
        clip = t["clip"] if isinstance(t["clip"], dict) else eval(t["clip"])  # noqa: S307 (repr di dict salvato)
        splice = t["splice"] if isinstance(t["splice"], dict) else eval(t["splice"])  # noqa: S307
        # file del tentativo finale: stesso indice e tipo, verdetto non fail
        cand = [f for f in files if f.name.startswith(f"t{idx:02d}_") and f.stem.endswith(t["type"])]
        chosen = None
        for f in cand:
            d = json.loads(f.read_text(encoding="utf-8"))
            if (d.get("qa") or {}).get("verdict") != "fail":
                chosen = (f, d)
        if chosen is None:
            continue
        f, d = chosen
        it = C.item_from_render(d["render"], item_id=f"dj_set_v1b/{f.stem}", dataset="dj_set_v1b", wav=str(wav), qa=d.get("qa"),
                                clip_offset_s=splice["clip_origin_set_sample"] / 48000.0, clip_dur_s=float(clip["duration_s"]),
                                group=f"set_t{idx:02d}")
        it.meta["set_times_s"] = t.get("set_times_s")
        it.meta["rule"] = t.get("rule")
        items.append(it)
    return items


def _v12_items(wav_by_id: dict) -> list[C.TransitionItem]:
    run = ROOT / "runs/psychology_v12_release"
    review = json.loads((run / "review_manifest.json").read_text(encoding="utf-8"))
    sel = {}
    for f in run.glob("selected_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        sel[d["candidate"]["candidate_id"]] = d["candidate"]
    items = []
    for it in review["items"]:
        cand = sel.get(it["candidate_id"])
        wav = ROOT / it["audio_path"]
        if cand is None or not wav.is_file():
            continue
        natural = it["intent"] == "native_reference"
        a_wav = wav_by_id.get(cand["track_a"])
        b_wav = a_wav if natural else wav_by_id.get(cand["track_b"])
        label = None
        if it["id"].endswith("_main"):
            label = {"ordinal": 1.0, "scale": "ordinale (1 = molto insufficiente)", "session": "psychology_v12_release",
                     "source": "dichiarazione dell'utente riportata dall'orchestratore: tutte le V12 'molto insufficienti'"}
        item = C.TransitionItem(
            item_id=f"v12/{it['id']}", dataset="v12_release" if not natural else "v12_natural_reference",
            wav=str(wav), clip_offset_s=0.0, clip_dur_s=20.0, start_s=10.0, switch_s=10.0, end_s=10.0,
            kind="handoff" if not natural else "natural_continuation", group=it["candidate_id"],
            a_audio=a_wav, a_switch_s=cand["cue_out_sample"] / 48000.0, a_track_id=cand["track_a"],
            b_audio=b_wav, b_switch_s=(cand["cue_out_sample"] if natural else cand["cue_in_sample"]) / 48000.0,
            b_track_id=cand["track_a"] if natural else cand["track_b"], label=label,
            meta={"intent": it["intent"], "title": it["title"]})
        item.wav_sha256 = C.sha256_file(wav)
        items.append(item)
    return items


V10_ORDINAL = {
    # Codifica *nostra* [H] dei commenti verbali dell'utente (feedback_verbale_V10):
    # 3 = "molto sufficiente"; 2.5 = "mid"; 1.5 = "non va bene"/"non ci sta"/"neanche";
    # 1 = "pessima, non carica e non risolve, spezza".
    "audio_001": 2.5, "audio_002": 1.5, "audio_003": 3.0, "audio_004": 1.5,
    "audio_005": 1.0, "audio_006": 1.5, "audio_007": 1.5, "audio_008": 1.5,
}


def _reconstruct(out_dir: Path, wav_by_id: dict) -> tuple[list[C.TransitionItem], list[dict]]:
    """Ricostruisce bit per bit le clip storiche ``anchor_handoff_v1`` e verifica lo sha256."""
    from musicmix.audio import load_audio
    from musicmix.expectation_pipeline import candidate_from_dict
    from musicmix.musical_preview import render_handoff

    rec_dir = out_dir / "reconstructed"
    rec_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Any] = {}

    def track(tid):
        if tid not in cache:
            cache[tid] = load_audio(wav_by_id[tid])
        return cache[tid]

    # candidati da manifest/proposals
    cands: dict[str, dict] = {}
    expected: dict[str, dict] = {}
    for m in ("runs/pretrained_v10_pool6/manifest.json", "runs/pretrained_v10_linkseg6/manifest.json",
              "runs/pool_experiment_v8_final/manifest.json"):
        d = json.loads((ROOT / m).read_text(encoding="utf-8"))
        for a in d.get("artifacts", []):
            if a.get("render_version") != "anchor_handoff_v1":
                continue
            expected[a["candidate_id"]] = {"sha256": a["sha256"], "manifest": m, "system": a.get("system")}
            if a.get("candidate"):
                cands[a["candidate_id"]] = a["candidate"]
    for f in (ROOT / "runs/pool_experiment_v8_final").glob("proposal_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        cands.setdefault(d["candidate"]["candidate_id"], d["candidate"])
    report = []
    ok_ids = {}
    for cid, exp in sorted(expected.items()):
        cand = cands.get(cid)
        entry = {"candidate_id": cid, "manifest": exp["manifest"], "expected_sha256": exp["sha256"]}
        if cand is None or cand["track_a"] not in wav_by_id or cand["track_b"] not in wav_by_id:
            entry["status"] = "sorgenti o candidato mancanti"
            report.append(entry)
            continue
        path = rec_dir / f"{cid}.wav"
        if path.exists():
            path.unlink()
        meta = render_handoff(track(cand["track_a"]), track(cand["track_b"]), candidate_from_dict(cand), path)
        entry.update(actual_sha256=meta["sha256"], status="bit-identico" if meta["sha256"] == exp["sha256"] else "DIVERSO")
        if meta["sha256"] != exp["sha256"]:
            path.unlink()
        else:
            ok_ids[cid] = (cand, path)
        report.append(entry)
    items = []
    # V10: giudizi verbali (8 audio)
    v10 = json.loads((GIUDIZI_STORICI / "feedback_verbale_V10_review_8_audio.json").read_text(encoding="utf-8"))
    for o in v10["observations"]:
        cid = o["candidate_id"]
        if cid not in ok_ids:
            continue
        cand, path = ok_ids[cid]
        items.append(_handoff_item(f"v10/{o['audio_id']}", "v10_reconstructed", path, cand, wav_by_id, o["audio_sha256"],
                                   {"ordinal": V10_ORDINAL[o["audio_id"]], "scale": "ordinale [H] codificato da commento verbale",
                                    "session": "pretrained_v10_review", "comment": o["user_comment"], "system": o["system"]}))
    # V8: confronti A/B (4 prove)
    fb = json.loads((GIUDIZI_STORICI / "feedback_Pool_V8_3bdeea23988e22af.json").read_text(encoding="utf-8"))
    for o in fb["observations"]:
        for lab, c in o["candidates"].items():
            cid = Path(c["source"].replace("\\", "/")).stem
            if cid not in ok_ids:
                continue
            cand, path = ok_ids[cid]
            items.append(_handoff_item(f"v8/{o['pair_id']}_{lab}", "v8_reconstructed", path, cand, wav_by_id, None,
                                       {"pair_id": o["pair_id"], "preference": o["preference_label"], "candidate_label": lab,
                                        "word_cut_reported": c.get("word_cut"), "session": "pool_experiment_v8", "system": c["system"]}))
    return items, report


def _handoff_item(item_id, dataset, path, cand, wav_by_id, sha_expected, label) -> C.TransitionItem:
    it = C.TransitionItem(item_id=item_id, dataset=dataset, wav=str(path), clip_offset_s=0.0, clip_dur_s=20.0,
                          start_s=10.0, switch_s=10.0, end_s=10.0, kind="handoff", group=item_id.split("/")[0],
                          a_audio=wav_by_id[cand["track_a"]], a_switch_s=cand["cue_out_sample"] / 48000.0, a_track_id=cand["track_a"],
                          b_audio=wav_by_id[cand["track_b"]], b_switch_s=cand["cue_in_sample"] / 48000.0, b_track_id=cand["track_b"],
                          label=label, meta={"candidate_id": cand["candidate_id"], "renderer": "anchor_handoff_v1"})
    it.wav_sha256 = C.sha256_file(path)
    if sha_expected and it.wav_sha256 != sha_expected:
        raise RuntimeError(f"{item_id}: sha ricostruito != sha registrato nel giudizio")
    return it


def stage_inventory(out: Path):
    wav_by_id = track_wav_by_id()
    items = _gallery_items("dj_gallery_v1", "dj_gallery_v1", _gallery_v1_labels())
    items += _gallery_items("dj_gallery_v2", "dj_gallery_v2", {})
    items += _set_items()
    items += _v12_items(wav_by_id)
    rec_items, rec_report = _reconstruct(out, wav_by_id)
    items += rec_items
    # stesse clip pubblicate in due gallerie: segnala il duplicato (stesso PCM)
    by_sha = {}
    for it in items:
        by_sha.setdefault(it.wav_sha256, []).append(it.item_id)
    for it in items:
        if it.wav_sha256 and it.clip_offset_s == 0.0 and len(by_sha.get(it.wav_sha256, [])) > 1:
            it.meta["same_pcm_as"] = [x for x in by_sha[it.wav_sha256] if x != it.item_id]
    C.write_json(out / "items.json", {"schema_version": C.DJ_CRITIC_VERSION, "items": [i.to_json() for i in items]})
    C.write_json(out / "reconstruction.json", {"renderer": "musicmix.musical_preview.render_handoff (anchor_handoff_v1)",
                                               "results": rec_report})
    log(f"inventory: {len(items)} item; ricostruzioni {sum(r['status']=='bit-identico' for r in rec_report)}/{len(rec_report)} bit-identiche")


def load_items(out: Path) -> list[C.TransitionItem]:
    d = json.loads((out / "items.json").read_text(encoding="utf-8"))
    return [C.TransitionItem.from_json(x) for x in d["items"]]


# ==================================================================== plan
LONG_SUBSET = {"natural": 40, "natural_vocoder": 8, "jump_clean": 12, "x_clean": 12, "phase": 6, "gain": 6, "key": 6,
               "word_cut": 6, "double_bass": 6, "off_phrase": 6, "x_phase": 4, "x_gain": 4, "x_key": 4,
               "x_word_cut": 4, "x_double_bass": 4, "x_off_phrase": 4}


def stage_plan(out: Path):
    tracks = load_tracks()
    tl = list(tracks.values())
    C.assign_folds(tl)
    recs = C.plan_examples(tl, seed=20261021)
    rng = np.random.default_rng(7)
    by_kind: dict[str, list] = {}
    for r in recs:
        by_kind.setdefault(r["kind"], []).append(r)
    for kind, lst in by_kind.items():
        n = LONG_SUBSET.get(kind, 0)
        for i in rng.permutation(len(lst))[:n]:
            lst[int(i)]["long"] = True
    C.write_json(out / "synth_plan.json", {
        "schema_version": C.DJ_CRITIC_VERSION,
        "tracks": {k: {"slug": v.slug, "wav": rel(v.wav), "fold": v.fold, "bpm": v.bpm, "camelot": v.camelot,
                       "n_bars": len(v.downbeats_s), "n_words": len(v.words), "analysis": rel(v.source),
                       "v12_track": v.slug in V12_TRACKS} for k, v in tracks.items()},
        "examples": recs,
        "counts": {k: len(v) for k, v in by_kind.items()},
        "long_counts": {k: sum(1 for r in v if r.get("long")) for k, v in by_kind.items()},
    })
    log(f"plan: {len(tracks)} brani, {len(recs)} esempi; per tipo {({k: len(v) for k, v in by_kind.items()})}")


def load_plan(out: Path) -> tuple[dict, list[dict]]:
    d = json.loads((out / "synth_plan.json").read_text(encoding="utf-8"))
    tracks = load_tracks()
    for slug, t in d["tracks"].items():
        if slug in tracks:
            tracks[slug].fold = t["fold"]
    return tracks, d["examples"]


def stage_doseplan(out: Path):
    tracks, _ = load_plan(out)
    tl = [t for t in tracks.values() if t.fold >= 0]
    recs = C.plan_dose(tl)
    C.write_json(out / "dose_plan.json", {"schema_version": C.DJ_CRITIC_VERSION, "levels": C.DOSE_LEVELS,
                                          "camelot_step_semitones": C.CAMELOT_STEP_SEMITONES, "examples": recs})
    log(f"doseplan: {len(recs)} varianti su {len(set(r['base'] for r in recs))} basi")


def load_dose(out: Path) -> list[dict]:
    p = out / "dose_plan.json"
    recs = json.loads(p.read_text())["examples"] if p.exists() else []
    # DJC_REVERSE=1: un secondo processo percorre le varianti dalla fine (niente lavoro doppio)
    return recs[::-1] if os.environ.get("DJC_REVERSE") == "1" else recs


def long_windows() -> dict:
    """Finestre (s) dentro un esempio lungo di 24 s con giunta a 12 s."""
    return {"seam@-1": (6.0, 10.0), "seam@+0": (7.0, 10.0), "seam@+1": (8.0, 10.0), "pre": (2.0, 10.0), "post": (12.0, 10.0)}


def long_item(rec: dict) -> C.TransitionItem:
    """Geometria di un esempio lungo come item (giunta a 12 s di 24 s)."""
    return C.TransitionItem(item_id=rec["id"], dataset="synthetic", wav="", clip_offset_s=0.0, clip_dur_s=C.EXAMPLE_LONG_S,
                            start_s=12.0, switch_s=12.0, end_s=12.0, kind=rec["kind"])


# ===================================================================== muq
class MuQRunner:
    def __init__(self, threads: int):
        import torch
        from safetensors.torch import load_file

        sys.path.insert(0, str(ROOT / ".cache/song_eval_runtime/SongEval"))
        from model import Generator  # type: ignore
        from muq import MuQ

        torch.set_num_threads(threads)
        self.torch = torch
        head_path = ROOT / C.PROVENANCE["songeval"]["file"]
        muq_dir = ROOT / Path(C.PROVENANCE["muq"]["file"]).parent
        for key, p in (("songeval", head_path), ("muq", muq_dir / "model.safetensors")):
            if C.sha256_file(p) != C.PROVENANCE[key]["sha256"]:
                raise RuntimeError(f"hash {key} diverso dal pin")
        self.head = Generator(in_features=1024, ffd_hidden_size=4096, num_classes=5, attn_layer_num=4)
        self.head.load_state_dict(load_file(str(head_path), device="cpu"), strict=True)
        self.head.eval()
        self.muq = MuQ.from_pretrained(str(muq_dir), local_files_only=True).eval()

    def hidden(self, mono24k: np.ndarray, layers=(6, 12)) -> dict:
        with self.torch.inference_mode():
            outp = self.muq(self.torch.from_numpy(np.ascontiguousarray(mono24k))[None, :], output_hidden_states=True)
            hs = outp["hidden_states"]
            return {l: hs[l][0].float().numpy() for l in layers}

    def songeval(self, hidden6: np.ndarray) -> dict:
        with self.torch.inference_mode():
            v = self.head(self.torch.from_numpy(hidden6)[None]).squeeze(0).numpy()
        return {a: float(v[i]) for i, a in enumerate(C.SONGEVAL_AXES)}


def item_priority(items: list) -> list:
    """Prima gli item con giudizi, poi le applicazioni, per ultimi gli scartati dal QA."""
    order = {"dj_gallery_v1": 0, "v12_release": 0, "v12_natural_reference": 0, "v10_reconstructed": 0, "v8_reconstructed": 1,
             "dj_gallery_v2": 2, "dj_set_v1b": 2, "dj_gallery_v1_rejected": 3, "dj_gallery_v2_rejected": 3}
    return sorted(items, key=lambda it: order.get(it.dataset, 4))


def stage_muq(out: Path, threads: int, limit: Optional[int] = None, part: str = "all"):
    cache = out / "cache/muq"
    cache.mkdir(parents=True, exist_ok=True)
    runner = MuQRunner(threads)
    tracks, recs = load_plan(out)
    if part in ("all", "items", "items_dose"):
        _muq_items(out, runner, limit)
    if part in ("all", "dose", "items_dose"):
        _muq_dose(out, runner, tracks)
    if part in ("items", "dose", "items_dose"):
        return
    # 1) esempi sintetici (12 s): feature della giunta, in ordine casuale
    #    (così un sottoinsieme bilanciato è pronto anche se la macchina si ferma)
    todo = [r for r in recs if not (cache / f"{r['id']}.npz").exists() and not (cache / f"{r['id']}.skip").exists()]
    todo = [todo[i] for i in np.random.default_rng(11).permutation(len(todo))]
    # gli esempi lunghi prima: servono anche ai critici a/b
    todo.sort(key=lambda r: not r.get("long"))
    if limit:
        todo = todo[:limit]
    log(f"muq: {len(todo)} esempi sintetici da calcolare")
    se_path = out / "cache/songeval_synth.jsonl"
    done = set()
    if se_path.exists():
        done = {json.loads(l)["id"] for l in se_path.read_text().splitlines() if l.strip()}
    t0 = time.time()
    with se_path.open("a") as fh:
        for n, r in enumerate(todo):
            try:
                x = C.synth_example(r, tracks, sr=C.SR_MUQ, length_s=C.EXAMPLE_S)
            except (ValueError, KeyError) as e:
                (cache / f"{r['id']}.skip").write_text(str(e))
                continue
            hid = runner.hidden(x)
            blocks = {l: C.seam_feature_block(h, fps=C.MUQ_FPS, seam_s=C.EXAMPLE_S / 2) for l, h in hid.items()}
            compact, full = C.seam_design_row(blocks)
            np.savez(cache / f"{r['id']}.npz", compact=compact, full=full.astype(np.float16))
            # 2) esempi lunghi (24 s): SongEval su finestre di 10 s ritagliate
            #    dagli hidden state di un solo passaggio MuQ sui 24 s
            if r.get("long") and r["id"] not in done:
                try:
                    xl = C.synth_example(r, tracks, sr=C.SR_MUQ, length_s=C.EXAMPLE_LONG_S)
                    h6 = runner.hidden(xl, layers=(6,))[6]
                    scores = {name: runner.songeval(h6[int(s0 * C.MUQ_FPS): int((s0 + d) * C.MUQ_FPS)])
                              for name, (s0, d) in long_windows().items()}
                    fh.write(json.dumps({"id": r["id"], "scores": scores, "method": SONGEVAL_METHOD}) + "\n")
                    fh.flush()
                except (ValueError, KeyError):
                    pass
            if n % 25 == 0:
                log(f"muq synth {n}/{len(todo)} ({(time.time()-t0)/(n+1):.1f} s/es)")


def _muq_dose(out: Path, runner, tracks):
    cache = out / "cache/muq_dose"
    cache.mkdir(parents=True, exist_ok=True)
    se_path = out / "cache/songeval_dose.jsonl"
    done = set()
    if se_path.exists():
        done = {json.loads(l)["id"] for l in se_path.read_text().splitlines() if l.strip()}
    recs = load_dose(out)
    with se_path.open("a") as fh:
        for n, r in enumerate(recs):
            if (cache / f"{r['id']}.npz").exists() and r["id"] in done:
                continue
            if r["a"] not in tracks or r["b"] not in tracks:
                continue  # sorgente non presente su questa macchina (es. dopo un riavvio)
            x = C.synth_example(r, tracks, sr=C.SR_MUQ, length_s=C.EXAMPLE_S)
            hid = runner.hidden(x)
            blocks = {l: C.seam_feature_block(h, fps=C.MUQ_FPS, seam_s=C.EXAMPLE_S / 2) for l, h in hid.items()}
            compact, full = C.seam_design_row(blocks)
            np.savez(cache / f"{r['id']}.npz", compact=compact, full=full.astype(np.float16))
            if r["id"] not in done:
                xl = C.synth_example(r, tracks, sr=C.SR_MUQ, length_s=C.EXAMPLE_LONG_S)
                h6 = runner.hidden(xl, layers=(6,))[6]
                scores = {name: runner.songeval(h6[int(s0 * C.MUQ_FPS): int((s0 + d) * C.MUQ_FPS)])
                          for name, (s0, d) in long_windows().items()}
                fh.write(json.dumps({"id": r["id"], "scores": scores, "method": SONGEVAL_METHOD}) + "\n")
                fh.flush()
            if n % 10 == 0:
                log(f"muq dose {n}/{len(recs)}")


def _muq_items(out: Path, runner, limit=None):
    """Item reali: SongEval sulle finestre estetiche + feature della giunta."""
    items = item_priority(load_items(out))
    it_path = out / "cache/muq_items.json"
    res = json.loads(it_path.read_text()) if it_path.exists() else {}
    if limit:
        items = items[:limit]
    for it in items:
        if it.item_id in res:
            continue
        entry: dict[str, Any] = {"songeval": {}, "seam": {}, "songeval_method": SONGEVAL_METHOD}
        full_clip = it.read_clip(C.SR_MUQ)
        h6 = runner.hidden(full_clip, layers=(6,))[6]
        for w in C.aesthetic_windows(it):
            if w["source"] == "clip":
                f0, f1 = int(w["start_s"] * C.MUQ_FPS), int((w["start_s"] + w["dur_s"]) * C.MUQ_FPS)
                entry["songeval"][w["name"]] = runner.songeval(h6[f0:f1])
            else:
                seg = C.read_window(it, w, C.SR_MUQ)
                entry["songeval"][w["name"]] = runner.songeval(runner.hidden(seg, layers=(6,))[6])
        for name, t in seam_points(it).items():
            s0 = t - C.EXAMPLE_S / 2
            if s0 < 0 or t + C.EXAMPLE_S / 2 > it.clip_dur_s + 1e-6:
                s0 = min(max(0.0, s0), max(0.0, it.clip_dur_s - C.EXAMPLE_S))
            seg = it.read_clip(C.SR_MUQ, s0, C.EXAMPLE_S)
            hid = runner.hidden(seg)
            try:
                blocks = {l: C.seam_feature_block(h, fps=C.MUQ_FPS, seam_s=t - s0) for l, h in hid.items()}
            except ValueError:
                continue
            compact, full = C.seam_design_row(blocks)
            entry["seam"][name] = {"t_s": t, "compact": compact.tolist(), "full": full.astype(np.float32).tolist()}
        res[it.item_id] = entry
        C.write_json(it_path, res)
        log(f"muq item {it.item_id}")


SONGEVAL_METHOD = ("testa SongEval ufficiale su hidden_states[6] di MuQ; finestre di 10 s ritagliate dai frame (25 fps) "
                   "di un solo passaggio sulla clip intera (24 s per gli esempi sintetici); sorgenti A/B: passaggio separato di 10 s")


def seam_points(it: C.TransitionItem) -> dict:
    pts = {"switch": it.switch_s}
    if it.start_s < it.switch_s - 1.0:
        pts["start"] = it.start_s
    if it.end_s > it.switch_s + 1.0:
        pts["end"] = it.end_s
    return pts


# ======================================================= altri embedding (v2)
MERT_DIR = C.MODELS / "hf/hub/models--m-a-p--MERT-v1-95M/snapshots/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
CLAP_DIR = C.MODELS / "hf/hub/models--laion--larger_clap_music/snapshots/a0b4534a14f58e20944452dff00a22a06ce629d1"


class MERTRunner:
    """MERT-v1-95M (m-a-p, ICLR 2024): hidden state a 75 Hz, input 24 kHz normalizzato."""
    fps = 75.0
    sr = 24000

    def __init__(self, threads):
        import torch
        from transformers import AutoModel

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = AutoModel.from_pretrained(str(MERT_DIR), trust_remote_code=True, local_files_only=True).eval()

    def features(self, x: np.ndarray, seam_s: float):
        x = (x - x.mean()) / (x.std() + 1e-7)
        with self.torch.inference_mode():
            hs = self.model(self.torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None], output_hidden_states=True).hidden_states
        blocks = {l: C.seam_feature_block(hs[l][0].numpy(), fps=self.fps, seam_s=seam_s) for l in (6, 12)}
        return C.seam_design_row(blocks)


class CLAPRunner:
    """LAION larger_clap_music (CLAP, Wu et al. 2023): embedding audio di finestre prima/dopo/attraverso."""
    sr = 48000

    def __init__(self, threads):
        import torch
        from transformers import ClapModel, ClapProcessor

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = ClapModel.from_pretrained(str(CLAP_DIR), local_files_only=True).eval()
        self.proc = ClapProcessor.from_pretrained(str(CLAP_DIR), local_files_only=True)

    def features(self, x: np.ndarray, seam_s: float):
        wins = {"pre": (seam_s - 3.0, seam_s - 0.25), "post": (seam_s + 0.25, seam_s + 3.0), "across": (seam_s - 1.5, seam_s + 1.5)}
        segs = [x[int(a * self.sr): int(b * self.sr)] for a, b in wins.values()]
        inp = self.proc(audios=segs, sampling_rate=self.sr, return_tensors="pt")
        with self.torch.inference_mode():
            e = self.model.get_audio_features(**inp).numpy()
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        pre, post, acr = e
        compact = np.array([float(pre @ post), float(np.linalg.norm(pre - post)), float(acr @ pre), float(acr @ post)])
        full = np.concatenate([pre, post, np.abs(pre - post), acr]).astype(np.float32)
        return compact, full


def stage_emb(out: Path, threads: int, model: str = "mert", limit: Optional[int] = None):
    runner = MERTRunner(threads) if model == "mert" else CLAPRunner(threads)
    cache = out / f"cache/emb_{model}"
    cache.mkdir(parents=True, exist_ok=True)
    tracks, recs = load_plan(out)
    # stesso ordine della fase muq: confronto sugli stessi esempi
    todo = [recs[i] for i in np.random.default_rng(11).permutation(len(recs))]
    todo.sort(key=lambda r: not r.get("long"))
    todo = load_dose(out) + todo
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    for n, r in enumerate(todo):
        p = cache / f"{r['id']}.npz"
        if p.exists():
            continue
        try:
            x = C.synth_example(r, tracks, sr=runner.sr, length_s=C.EXAMPLE_S)
        except (ValueError, KeyError):
            continue
        compact, full = runner.features(x, C.EXAMPLE_S / 2)
        np.savez(p, compact=compact, full=full.astype(np.float16))
        if n % 25 == 0:
            log(f"emb {model} {n}/{len(todo)} ({(time.time()-t0)/(n+1):.1f} s/es)")
    # item reali: punti della giunta
    ip = out / f"cache/emb_{model}_items.json"
    res = json.loads(ip.read_text()) if ip.exists() else {}
    for it in item_priority(load_items(out)):
        if it.item_id in res:
            continue
        entry = {}
        for name, t in seam_points(it).items():
            s0 = min(max(0.0, t - C.EXAMPLE_S / 2), max(0.0, it.clip_dur_s - C.EXAMPLE_S))
            seg = it.read_clip(runner.sr, s0, C.EXAMPLE_S)
            try:
                compact, full = runner.features(seg, t - s0)
            except ValueError:
                continue
            entry[name] = {"t_s": t, "compact": compact.tolist(), "full": full.astype(np.float32).tolist()}
        res[it.item_id] = entry
        C.write_json(ip, res)
    log(f"emb {model}: fatto")


# ================================================================ musicgen
def stage_musicgen(out: Path, threads: int, limit: Optional[int] = None, part: str = "all"):
    cache = out / "cache/musicgen"
    cache.mkdir(parents=True, exist_ok=True)
    mg = C.MusicGenSurprisal(threads=threads)
    items = item_priority(load_items(out))
    if limit:
        items = items[:limit]
    if part not in ("all", "items"):
        items = []
    for it in items:
        p = cache / (it.item_id.replace("/", "__") + ".npz")
        if p.exists():
            continue
        dur = min(it.clip_dur_s, C.SURPRISAL_MAX_S)
        # finestra di al massimo 30 s che contiene tutta la transizione e lati simmetrici
        span = it.end_s - it.start_s
        side = max(4.0, (dur - span) / 2)
        s0 = max(0.0, min(it.start_s - side, it.clip_dur_s - dur))
        x = it.read_clip(C.SR_MUSICGEN, s0, dur)
        nll = mg.frame_nll(x)
        np.savez(p, nll=nll.astype(np.float32), offset_s=s0, dtype=mg.dtype)
        log(f"musicgen item {it.item_id} ({dur:.1f} s)")
    tracks, recs = load_plan(out)
    if part == "all":
        # prima le giunte naturali lunghe: servono come distribuzione nulla a tutti
        for r in [r for r in recs if r.get("long") and r["kind"] in ("natural", "natural_vocoder")]:
            p = cache / f"{r['id']}.npz"
            if p.exists():
                continue
            try:
                x = C.synth_example(r, tracks, sr=C.SR_MUSICGEN, length_s=C.EXAMPLE_LONG_S)
            except (ValueError, KeyError):
                continue
            np.savez(p, nll=mg.frame_nll(x).astype(np.float32), offset_s=0.0, dtype=mg.dtype)
            log(f"musicgen null {r['id']}")
    if part in ("all", "dose"):
        for r in load_dose(out):
            p = cache / f"{r['id']}.npz"
            if p.exists() or r["a"] not in tracks or r["b"] not in tracks:
                continue
            x = C.synth_example(r, tracks, sr=C.SR_MUSICGEN, length_s=C.EXAMPLE_LONG_S)
            np.savez(p, nll=mg.frame_nll(x).astype(np.float32), offset_s=0.0, dtype=mg.dtype)
            log(f"musicgen dose {r['id']}")
    if part not in ("all", "synth"):
        return
    longs = [r for r in recs if r.get("long") and not (cache / f"{r['id']}.npz").exists()]
    longs = [longs[i] for i in np.random.default_rng(13).permutation(len(longs))]
    if limit:
        longs = longs[:limit]
    t0 = time.time()
    for n, r in enumerate(longs):
        try:
            x = C.synth_example(r, tracks, sr=C.SR_MUSICGEN, length_s=C.EXAMPLE_LONG_S)
        except (ValueError, KeyError):
            continue
        nll = mg.frame_nll(x)
        np.savez(cache / f"{r['id']}.npz", nll=nll.astype(np.float32), offset_s=0.0, dtype=mg.dtype)
        if n % 10 == 0:
            log(f"musicgen synth {n}/{len(longs)} ({(time.time()-t0)/(n+1):.1f} s/es)")


# ================================================================ audiobox
def audiobox_predictor(threads: int):
    from musicmix.aesthetic_critic import AudioboxSeamCritic

    c = AudioboxSeamCritic(ROOT, threads=threads)
    c._load()
    if c.provenance["sha256"] != C.PROVENANCE["audiobox"]["sha256"]:
        raise RuntimeError("hash Audiobox diverso dal pin")
    return c


def _ab_score(c, segs: list[np.ndarray]) -> list[dict]:
    import torch

    with torch.inference_mode():
        return c.predictor.forward([{"path": torch.from_numpy(np.ascontiguousarray(s))[None, :], "sample_rate": C.SR_AUDIOBOX} for s in segs])


def stage_audiobox(out: Path, threads: int):
    c = audiobox_predictor(threads)
    items = load_items(out)
    path = out / "cache/audiobox_items.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for it in items:
        if it.item_id in res:
            continue
        ws = C.aesthetic_windows(it)
        segs = [C.read_window(it, w, C.SR_AUDIOBOX) for w in ws]
        res[it.item_id] = {w["name"]: s for w, s in zip(ws, _ab_score(c, segs))}
        C.write_json(path, res)
    log(f"audiobox: {len(res)} item")
    tracks, recs = load_plan(out)
    spath = out / "cache/audiobox_synth.jsonl"
    done = set()
    if spath.exists():
        done = {json.loads(l)["id"] for l in spath.read_text().splitlines() if l.strip()}
    longs = [r for r in recs if r.get("long") and r["id"] not in done]
    with spath.open("a") as fh:
        for r in longs:
            try:
                x = C.synth_example(r, tracks, sr=C.SR_AUDIOBOX, length_s=C.EXAMPLE_LONG_S)
            except (ValueError, KeyError):
                continue
            names = list(long_windows())
            segs = [x[int(s * C.SR_AUDIOBOX): int((s + d) * C.SR_AUDIOBOX)] for s, d in long_windows().values()]
            fh.write(json.dumps({"id": r["id"], "scores": dict(zip(names, _ab_score(c, segs)))}) + "\n")
            fh.flush()
    log("audiobox: esempi lunghi fatti")
    dpath = out / "cache/audiobox_dose.jsonl"
    done = set()
    if dpath.exists():
        done = {json.loads(l)["id"] for l in dpath.read_text().splitlines() if l.strip()}
    with dpath.open("a") as fh:
        for r in load_dose(out):
            if r["id"] in done:
                continue
            x = C.synth_example(r, tracks, sr=C.SR_AUDIOBOX, length_s=C.EXAMPLE_LONG_S)
            names = list(long_windows())
            segs = [x[int(s0 * C.SR_AUDIOBOX): int((s0 + d) * C.SR_AUDIOBOX)] for s0, d in long_windows().values()]
            fh.write(json.dumps({"id": r["id"], "scores": dict(zip(names, _ab_score(c, segs)))}) + "\n")
            fh.flush()
    log("audiobox: dose fatta")


# ====================================================================== qa
def stage_qa(out: Path):
    """dj_qa (versione corrente) solo-clip per gli item senza report QA."""
    from musicmix import dj_qa
    import soundfile as sf

    items = load_items(out)
    path = out / "cache/qa_clip_only.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    tracks = load_tracks()
    words_by_wav = {t.wav: t.words for t in tracks.values()}
    for it in items:
        if it.qa or it.item_id in res:
            continue
        x, sr = sf.read(it.wav, dtype="float32", always_2d=True)
        sw = int(round(it.switch_s * sr))
        meta = {"kind": "cut", "sample_rate": sr, "overlap": {"output_start_sample": sw, "output_end_sample": sw},
                "switch": {"output_sample": sw}, "clip": {"length_samples": len(x)}}
        words = None
        if it.a_audio in words_by_wav and it.a_switch_s is not None:
            off = it.a_switch_s - it.switch_s
            words = [{"start_s": s - off, "end_s": e - off, "text": ""} for s, e in words_by_wav[it.a_audio]
                     if 0 <= s - off and e - off <= it.switch_s + 1.0]
        try:
            res[it.item_id] = dj_qa.evaluate_render(x, meta, sample_rate=sr, words_a=words, tracker="auto")
            res[it.item_id]["critic_note"] = "dj_qa solo-clip (niente deck, stem o griglie): ritmo da Beat This! sul mix, verse_cut dalle parole ASR di A"
        except Exception as e:  # noqa: BLE001
            res[it.item_id] = {"error": f"{type(e).__name__}: {e}"}
        C.write_json(path, res)
    log(f"qa: {len(res)} item solo-clip")


def qa_clip_only(x: np.ndarray, sr: int, start_s: float, switch_s: float, end_s: float, words=None) -> dict:
    """dj_qa sulla sola clip (stessa procedura per item reali e sintetici, v2)."""
    from musicmix import dj_qa

    st, sw, en = (int(round(v * sr)) for v in (start_s, switch_s, end_s))
    meta = {"kind": "cut" if en - st < int(0.05 * sr) else "blend", "sample_rate": sr,
            "overlap": {"output_start_sample": st, "output_end_sample": en}, "switch": {"output_sample": sw},
            "clip": {"length_samples": len(x)}}
    r = dj_qa.evaluate_render(x, meta, sample_rate=sr, words_a=words, tracker="auto")
    r["critic_note"] = "dj_qa solo-clip: ritmo da Beat This! sul mix, loudness/true peak sulla clip, verse_cut dalle parole ASR di A"
    return r


def stage_qa2(out: Path):
    """v2: dj_qa solo-clip per tutti gli item, gli esempi lunghi e le varianti dose (feature tecnica comparabile)."""
    path = out / "cache/qa2.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    tracks, recs = load_plan(out)
    words_by_wav = {t.wav: t.words for t in tracks.values()}
    n_new = 0
    for it in item_priority(load_items(out)):
        if it.item_id in res:
            continue
        x, _ = C.load_audio(it.wav, start_s=it.clip_offset_s, duration_s=it.clip_dur_s, sr=48000, mono=False)
        words = None
        if it.a_audio in words_by_wav and it.a_switch_s is not None:
            off = it.a_switch_s - it.switch_s
            words = [{"start_s": a - off, "end_s": b - off, "text": ""} for a, b in words_by_wav[it.a_audio]
                     if 0 <= a - off and b - off <= it.switch_s + 1.0]
        try:
            res[it.item_id] = qa_clip_only(x, 48000, it.start_s, it.switch_s, it.end_s, words)
        except Exception as e:  # noqa: BLE001
            res[it.item_id] = {"error": f"{type(e).__name__}: {e}"}
        n_new += 1
        if n_new % 5 == 0:
            C.write_json(path, res)
    C.write_json(path, res)
    log(f"qa2: item fatti ({len(res)})")
    syn = load_dose(out) + [r for r in recs if r.get("long")]
    for r in syn:
        if r["id"] in res:
            continue
        try:
            x = C.synth_example(r, tracks, sr=48000, length_s=C.EXAMPLE_LONG_S)
            a = tracks[r["a"]]
            off = r["a_t"] - C.EXAMPLE_LONG_S / 2
            words = [{"start_s": s0 - off, "end_s": e0 - off, "text": ""} for s0, e0 in a.words if 0 <= s0 - off and e0 - off <= 13.0]
            res[r["id"]] = qa_clip_only(x, 48000, 12.0, 12.0, 12.0, words)
        except Exception as e:  # noqa: BLE001
            res[r["id"]] = {"error": f"{type(e).__name__}: {e}"}
        n_new += 1
        if n_new % 10 == 0:
            C.write_json(path, res)
            log(f"qa2: {len(res)}")
    C.write_json(path, res)
    log(f"qa2: fatto ({len(res)})")


# ============================================================ craft (v3)
def stage_craft(out: Path):
    """v3: misure del mestiere (loudness, fase dei battiti Beat This!, tonalità, voce sul taglio)
    su tutti gli esempi sintetici (24 s), le varianti dose e gli item reali.
    ``DJC_SHARD=i/n``: questo processo calcola solo la sua parte (file cache/craft_shard{i}.json);
    senza shard, unisce i file shard in cache/craft.json e completa ciò che manca."""
    import torch

    torch.set_num_threads(1)
    path = out / "cache/craft.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    shard = os.environ.get("DJC_SHARD")
    si, sn = (int(x) for x in shard.split("/")) if shard else (0, 1)
    res = _json(path)
    for f in sorted(path.parent.glob("craft_shard*.json")):
        res.update({k: v for k, v in _json(f).items() if v.get("craft_version") == C.CRAFT_VERSION})
    own_path = path.parent / f"craft_shard{si}.json" if shard else path
    own = _json(own_path) if shard else res

    def fresh(k):
        v = res.get(k) or own.get(k)
        return v is not None and v.get("craft_version") == C.CRAFT_VERSION

    def mine(k):
        return int(hashlib.sha256(k.encode()).hexdigest(), 16) % sn == si

    n_new = 0
    t0 = time.time()
    for it in item_priority(load_items(out)):
        if fresh(it.item_id) or not mine(it.item_id):
            continue
        try:
            x, sr = C.load_audio(it.wav, start_s=it.clip_offset_s, duration_s=it.clip_dur_s, sr=C.SR_CRAFT)
            side = min(C.CRAFT_SIDE_S, it.start_s - 0.2, it.clip_dur_s - it.end_s - 0.2)
            own[it.item_id] = C.craft_features(x, sr, it.start_s, it.end_s, side_s=side)
        except Exception as e:  # noqa: BLE001
            own[it.item_id] = {"error": f"{type(e).__name__}: {e}", "craft_version": C.CRAFT_VERSION}
        n_new += 1
    C.write_json(own_path, own)
    log(f"craft: item ({len(own)})")
    if (out / "synth_plan.json").exists():
        tracks, recs = load_plan(out)
        syn = load_dose(out) + [recs[i] for i in np.random.default_rng(11).permutation(len(recs))]
        for r in syn:
            if fresh(r["id"]) or not mine(r["id"]):
                continue
            if r["a"] not in tracks or r["b"] not in tracks or (r.get("bass_from") and r["bass_from"] not in tracks):
                continue
            try:
                x = C.synth_example(r, tracks, sr=C.SR_CRAFT, length_s=C.EXAMPLE_LONG_S)
                own[r["id"]] = C.craft_features(x, C.SR_CRAFT, C.EXAMPLE_LONG_S / 2, side_s=C.CRAFT_SIDE_S)
            except Exception as e:  # noqa: BLE001
                own[r["id"]] = {"error": f"{type(e).__name__}: {e}", "craft_version": C.CRAFT_VERSION}
            n_new += 1
            if n_new % 25 == 0:
                C.write_json(own_path, own)
                log(f"craft: {len(own)} ({(time.time()-t0)/n_new:.1f} s/es)")
    C.write_json(own_path, own)
    if not shard:
        C.write_json(path, {k: v for k, v in own.items()})
    log(f"craft: fatto ({len(own)})")


# =================================================================== train
def _load_synth_features(out: Path, recs: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows, comp, full = [], [], []
    for r in recs:
        p = out / "cache/muq" / f"{r['id']}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        rows.append(r)
        comp.append(z["compact"])
        full.append(z["full"].astype(np.float32))
    return rows, np.array(comp), np.array(full)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def stage_train(out: Path):
    tracks, recs = load_plan(out)
    rows, comp, full = _load_synth_features(out, recs)
    y = np.array([r["label"] for r in rows])
    folds = np.array([r["fold"] for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    v12 = np.array([(r["a"] in V12_TRACKS) or (r["b"] in V12_TRACKS) or (r.get("bass_from") in V12_TRACKS) for r in rows])
    configs = {"compact": (comp, None, 1.0), "full_pca32": (full, 32, 0.1), "full_pca64": (full, 64, 0.1)}
    cv: dict[str, Any] = {}
    oof_all = {}
    for name, (x, pca, cval) in configs.items():
        oof = np.full(len(y), np.nan)
        for f in sorted(set(folds.tolist())):
            tr, te = folds != f, folds == f
            m = C.fit_linear_seam_model(x[tr], y[tr], c=cval, pca=pca)
            oof[te] = m.proba(x[te])
        oof_all[name] = oof
        per_kind = {}
        for k in sorted(set(kinds.tolist())):
            if k in C.POSITIVE_KINDS:
                continue
            msk = (kinds == k) | np.isin(kinds, [p for p in C.POSITIVE_KINDS if (p.startswith("x_") == k.startswith("x_"))])
            per_kind[k] = {"auc_vs_positives_same_domain": _auc(y[msk], oof[msk]), "n_neg": int((kinds == k).sum())}
        pos_by_kind = {k: float(np.mean(oof[kinds == k])) for k in C.POSITIVE_KINDS if (kinds == k).any()}
        cv[name] = {"auc": _auc(y, oof), "auc_within": _auc(y[np.char.find(kinds.astype(str), "x_") != 0], oof[np.char.find(kinds.astype(str), "x_") != 0]),
                    "auc_cross": _auc(y[np.char.startswith(kinds.astype(str), "x_")], oof[np.char.startswith(kinds.astype(str), "x_")]),
                    "per_negative_kind": per_kind, "mean_p_positive_kinds": pos_by_kind,
                    "brier": float(np.mean((oof - y) ** 2))}
    # modello scelto con la CV sintetica (mai con i giudizi umani)
    best = max(cv, key=lambda k: cv[k]["auc"])
    x, pca, cval = configs[best]
    models = {
        "all": C.fit_linear_seam_model(x, y, c=cval, pca=pca),
        "no_v12_tracks": C.fit_linear_seam_model(x[~v12], y[~v12], c=cval, pca=pca),
    }
    fold_models = {}
    for f in sorted(set(folds.tolist())):
        fold_models[str(f)] = C.fit_linear_seam_model(x[folds != f], y[folds != f], c=cval, pca=pca).to_json()
    C.write_json(out / "models/seam_models.json", {
        "feature_set": best, "configs": {k: {"pca": v[1], "C": v[2]} for k, v in configs.items()},
        "models": {k: m.to_json() for k, m in models.items()}, "fold_models": fold_models,
        "n_train": int(len(y)), "n_train_no_v12": int((~v12).sum()),
        "muq_layers": [6, 12], "selection": "massima AUC in CV per brano (GroupKFold a 5 fold) sugli esempi sintetici",
    })
    np.savez(out / "cache/seam_oof.npz", ids=np.array([r["id"] for r in rows]), **{k: v for k, v in oof_all.items()})
    C.write_json(out / "seam_cv.json", {"n": int(len(y)), "n_pos": int(y.sum()), "n_tracks": len(set(r["a"] for r in rows)),
                                        "cv": cv, "chosen": best,
                                        "note": "fold = brani disgiunti; cross-track solo dentro lo stesso fold"})
    log(f"train: n={len(y)} scelto {best} AUC {cv[best]['auc']:.3f}")


# =================================================================== score
def _songeval_rel(scores: dict) -> dict:
    return C.aesthetic_relative(scores, C.SONGEVAL_AXES)


def _audiobox_rel(scores: dict) -> dict:
    return C.aesthetic_relative(scores, C.AUDIOBOX_QUALITY_AXES)


def _jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    return {d["id"]: d for d in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def build_nulls(out: Path) -> dict:
    """Distribuzioni nulle dalle giunte naturali sintetiche (esempi lunghi)."""
    tracks, recs = load_plan(out)
    byid = {r["id"]: r for r in recs}
    ab = _jsonl(out / "cache/audiobox_synth.jsonl")
    se = _jsonl(out / "cache/songeval_synth.jsonl")
    nulls = {"audiobox_delta": [], "songeval_delta": [], "surprisal_peak_z": []}
    synth_rows = []
    for rid, r in byid.items():
        if not r.get("long"):
            continue
        row = {"id": rid, "kind": r["kind"], "label": r["label"]}
        if rid in ab:
            row["audiobox_delta"] = _audiobox_rel(ab[rid]["scores"]).get("delta_vs_sides")
        if rid in se:
            row["songeval_delta"] = _songeval_rel(se[rid]["scores"]).get("delta_vs_sides")
        p = out / "cache/musicgen" / f"{rid}.npz"
        if p.exists():
            z = np.load(p)
            st = C.surprisal_seam_stats(z["nll"].sum(0), fps=C.MUSICGEN_FPS, start_s=12.0, switch_s=12.0, end_s=12.0)
            row["surprisal_peak_z"] = st.get("peak_z")
            row["surprisal_excess"] = st.get("excess_post2s")
        synth_rows.append(row)
        if r["kind"] in ("natural", "natural_vocoder"):
            for k in nulls:
                if row.get(k) is not None:
                    nulls[k].append(row[k])
    return {"nulls": nulls, "synth_rows": synth_rows}


def critic_scores_for_item(it: C.TransitionItem, *, ab: dict, mq: dict, mg_path: Path, qa_extra: dict,
                           seam_model: Optional[C.LinearSeamModel], fold_models: list, nulls: dict, feature_set: str) -> dict:
    out: dict[str, Any] = {}
    # (a) estetico
    a_parts = {}
    if it.item_id in ab:
        rel_ab = _audiobox_rel(ab[it.item_id])
        a_parts["audiobox"] = rel_ab
    if it.item_id in mq and mq[it.item_id].get("songeval"):
        a_parts["songeval"] = _songeval_rel(mq[it.item_id]["songeval"])
    pcts, sds = [], []
    for name, rel_ in a_parts.items():
        d = rel_.get("delta_vs_sides")
        null = nulls[f"{name}_delta"]
        if d is None or not null:
            continue
        pct = C.empirical_percentile(d, null)
        rel_["percentile_vs_natural"] = pct
        pcts.append(pct)
        sds.append(C.percentile_sd(d, rel_.get("seam_sd_jitter", 0.05), null))
    out["aesthetic"] = {"score": float(np.mean(pcts)) if pcts else float("nan"),
                        "sd": float(np.sqrt(np.mean(np.square(sds)))) if sds else float("nan"), "parts": a_parts}
    # (b) sorpresa
    if mg_path.exists():
        z = np.load(mg_path)
        nll = z["nll"].sum(0)
        st = C.surprisal_seam_stats(nll, fps=C.MUSICGEN_FPS, start_s=it.start_s, switch_s=it.switch_s, end_s=it.end_s,
                                    offset_s=float(z["offset_s"]))
        if st.get("status") == "ok":
            null = nulls["surprisal_peak_z"]
            pct = C.empirical_percentile(st["peak_z"], null) if null else float("nan")
            st["percentile_vs_natural"] = pct
            out["surprisal"] = {"score": 1.0 - pct, "sd": C.percentile_sd(st["peak_z"], st["peak_z_sd"], null), "stats": st}
        else:
            out["surprisal"] = {"score": float("nan"), "sd": float("nan"), "stats": st}
    # (c) giunta
    if it.item_id in mq and mq[it.item_id].get("seam") and seam_model is not None:
        per_point = {}
        for name, s in mq[it.item_id]["seam"].items():
            x = np.array(s["compact"] if feature_set == "compact" else s["full"], dtype=np.float64)
            p = float(seam_model.proba(x)[0])
            ens = [float(m.proba(x)[0]) for m in fold_models]
            per_point[name] = {"p": p, "fold_sd": float(np.std(ens)), "t_s": s["t_s"]}
        worst = min(per_point, key=lambda k: per_point[k]["p"])
        out["seam"] = {"score": per_point[worst]["p"], "sd": max(per_point[worst]["fold_sd"], 0.02), "weakest_point": worst,
                       "points": per_point}
    # (d) tecnico
    qa = it.qa or qa_extra.get(it.item_id)
    tech = C.technical_from_qa(qa if qa and "checks" in qa else None)
    tech["source"] = "stored (render)" if it.qa else ("clip-only dj_qa" if qa else None)
    out["technical"] = tech if tech.get("status") == "ok" else {"score": float("nan"), "sd": float("nan"), **tech}
    out["combined"] = C.combine({k: out[k] for k in ("aesthetic", "surprisal", "seam", "technical") if k in out})
    return out


def stage_score(out: Path):
    items = load_items(out)
    ab = json.loads((out / "cache/audiobox_items.json").read_text()) if (out / "cache/audiobox_items.json").exists() else {}
    mq = json.loads((out / "cache/muq_items.json").read_text()) if (out / "cache/muq_items.json").exists() else {}
    qa_extra = json.loads((out / "cache/qa_clip_only.json").read_text()) if (out / "cache/qa_clip_only.json").exists() else {}
    sm = json.loads((out / "models/seam_models.json").read_text())
    fs = sm["feature_set"]
    model_all = C.LinearSeamModel.from_json(sm["models"]["all"])
    model_val = C.LinearSeamModel.from_json(sm["models"]["no_v12_tracks"])
    fold_models = [C.LinearSeamModel.from_json(m) for m in sm["fold_models"].values()]
    nb = build_nulls(out)
    nulls = nb["nulls"]
    results = {}
    for it in items:
        # Per gli item che coinvolgono i 7 brani V12 (tutti quelli con giudizi
        # umani) si usa il modello addestrato senza quei brani.
        uses_v12 = any(s in (it.a_audio or "") + (it.b_audio or "") for s in ("pool_v8", "pool_v10", "pool_v12", "reminder_to_too_many_nights"))
        model = model_val if uses_v12 else model_all
        mg = out / "cache/musicgen" / (it.item_id.replace("/", "__") + ".npz")
        r = critic_scores_for_item(it, ab=ab, mq=mq, mg_path=mg, qa_extra=qa_extra, seam_model=model,
                                   fold_models=fold_models, nulls=nulls, feature_set=fs)
        r["seam_model"] = "no_v12_tracks" if uses_v12 else "all"
        results[it.item_id] = {"item": {k: getattr(it, k) for k in ("dataset", "kind", "group", "label", "wav_sha256")}, **r}
    # benchmark sintetico dei critici a/b: AUC giunte naturali vs difetti (esempi lunghi)
    bench = {}
    rows = nb["synth_rows"]
    oof = np.load(out / "cache/seam_oof.npz") if (out / "cache/seam_oof.npz").exists() else None
    oof_map = dict(zip(oof["ids"].tolist(), oof[fs].tolist())) if oof is not None else {}
    for key, sign in (("audiobox_delta", 1), ("songeval_delta", 1), ("surprisal_peak_z", -1), ("surprisal_excess", -1), ("seam_oof", 1)):
        vals = [(r["label"], r["kind"], r.get(key) if key != "seam_oof" else oof_map.get(r["id"])) for r in rows]
        vals = [v for v in vals if v[2] is not None and np.isfinite(v[2])]
        if not vals:
            continue
        y = np.array([v[0] for v in vals])
        s = sign * np.array([v[2] for v in vals])
        kinds = np.array([v[1] for v in vals])
        per = {}
        for k in sorted(set(kinds.tolist()) - set(C.POSITIVE_KINDS)):
            m = (kinds == k) | np.isin(kinds, ["natural", "natural_vocoder"])
            per[k] = _auc(y[m], s[m])
        bench[key] = {"auc_all": _auc(y, s), "n": int(len(y)), "n_pos": int(y.sum()), "auc_vs_natural_per_kind": per}
    C.write_json(out / "scores.json", {"schema_version": C.DJ_CRITIC_VERSION, "combination": C.COMBINATION,
                                       "null_sizes": {k: len(v) for k, v in nulls.items()}, "items": results,
                                       "synthetic_benchmark": bench})
    log(f"score: {len(results)} item; null {({k: len(v) for k, v in nulls.items()})}")


# ================================================================ validate
CRITIC_KEYS = {
    "combined": lambda r: r["combined"].get("score"),
    "aesthetic": lambda r: r["aesthetic"].get("score"),
    "audiobox_delta_sides": lambda r: r["aesthetic"]["parts"].get("audiobox", {}).get("delta_vs_sides"),
    "audiobox_delta_sources": lambda r: r["aesthetic"]["parts"].get("audiobox", {}).get("delta_vs_sources"),
    "audiobox_seam_abs": lambda r: r["aesthetic"]["parts"].get("audiobox", {}).get("seam"),
    "songeval_delta_sides": lambda r: r["aesthetic"]["parts"].get("songeval", {}).get("delta_vs_sides"),
    "songeval_delta_sources": lambda r: r["aesthetic"]["parts"].get("songeval", {}).get("delta_vs_sources"),
    "songeval_seam_abs": lambda r: r["aesthetic"]["parts"].get("songeval", {}).get("seam"),
    "surprisal": lambda r: (r.get("surprisal") or {}).get("score"),
    "seam": lambda r: (r.get("seam") or {}).get("score"),
    "technical": lambda r: r["technical"].get("score"),
}


def _val(r, key):
    try:
        v = CRITIC_KEYS[key](r)
    except (KeyError, TypeError, AttributeError):
        return float("nan")
    return float("nan") if v is None else float(v)


def stage_validate(out: Path):
    sc = json.loads((out / "scores.json").read_text())["items"]
    sets: dict[str, list] = {}
    # S1: galleria v1, voti 1-10 (n=6, un ascoltatore, una sessione)
    s1 = [(k, v["item"]["label"]["score"]) for k, v in sc.items() if v["item"]["dataset"] == "dj_gallery_v1" and v["item"]["label"]]
    sets["S1_gallery_v1_scores"] = s1
    # S2: V10 ricostruite, ordinale codificato da commenti verbali [H]
    sets["S2_v10_verbal_ordinal"] = [(k, v["item"]["label"]["ordinal"]) for k, v in sc.items() if v["item"]["dataset"] == "v10_reconstructed"]
    # S3: tra sessioni: galleria v1 (voto >= 7) contro V12 main ("molto insufficienti")
    s3 = [(k, 2.0) for k, s in s1 if s >= 7] + [(k, 1.0) for k, v in sc.items() if v["item"]["dataset"] == "v12_release" and v["item"]["label"]]
    sets["S3_gallery_ge7_vs_v12_main"] = s3
    res: dict[str, Any] = {"sets": {}, "v8_pairs": {}, "notes": [
        "Nessun peso o soglia è stato stimato su questi giudizi: la combinazione è dichiarata prima (PREREGISTRATION.json).",
        "Il classificatore (c) per gli item con brani V12 è addestrato senza quei 7 brani.",
        "n minuscolo: gli intervalli sono larghi e le coppie non sono indipendenti (bootstrap sugli item).",
        "S2 usa una codifica ordinale nostra dei commenti verbali V10 [H]."]}
    for name, pairs in sets.items():
        ids = [k for k, _ in pairs]
        truth = np.array([t for _, t in pairs], dtype=float)
        per = {}
        for key in CRITIC_KEYS:
            pred = np.array([_val(sc[k], key) for k in ids])
            pa = C.pairwise_accuracy(pred, truth)
            k_ok = sum(p[2] for p in pa["pairs"])
            entry = {"n_items": int(np.isfinite(pred).sum()), "n_pairs": pa["n_pairs"], "pairwise_accuracy": pa["accuracy"],
                     "wilson95_pairs": C.wilson_interval(k_ok, pa["n_pairs"]),
                     "bootstrap95_items": C.bootstrap_pairwise_ci(pred, truth) if pa["n_pairs"] else None}
            if name != "S3_gallery_ge7_vs_v12_main":
                entry["spearman"] = C.spearman_with_permutation(pred, truth, n_perm=5000)
            per[key] = entry
        res["sets"][name] = {"n_items": len(ids), "items": ids, "truth": truth.tolist(), "critics": per}
    # V8: unica preferenza netta (pool_04: A > B) + pareggi/entrambi scartati
    v8 = {}
    for k, v in sc.items():
        if v["item"]["dataset"] == "v8_reconstructed":
            lab = v["item"]["label"]
            v8.setdefault(lab["pair_id"], {})[lab["candidate_label"]] = (k, lab["preference"])
    for pid, d in sorted(v8.items()):
        if set(d) != {"A", "B"}:
            continue
        pref = d["A"][1]
        row = {"preference": pref}
        for key in CRITIC_KEYS:
            a, b = _val(sc[d["A"][0]], key), _val(sc[d["B"][0]], key)
            row[key] = {"A": a, "B": b, "agrees": (None if pref not in ("A", "B") or not (np.isfinite(a) and np.isfinite(b))
                                                   else bool((a > b) == (pref == "A")))}
        res["v8_pairs"][pid] = row
    # controllo: continuazione naturale V12 contro la transizione main sulla stessa coppia
    nat = {}
    for k, v in sc.items():
        if v["item"]["dataset"] in ("v12_release", "v12_natural_reference"):
            nat.setdefault(v["item"]["group"], {})[v["item"]["dataset"]] = k
    res["v12_natural_vs_main"] = {g: {key: {"natural": _val(sc[d["v12_natural_reference"]], key), "main": _val(sc[d["v12_release"]], key)}
                                      for key in CRITIC_KEYS}
                                  for g, d in nat.items() if len(d) == 2}
    C.write_json(out / "validation.json", res)
    log("validate: fatto")


# ======================================================================= v2
# v2 (indicazioni dell'utente, notte 20->21/10): confronto embedding MuQ/MERT/CLAP,
# ensemble calibrato su dati sintetici (stacking con CV per brano), incertezza da
# bootstrap per brano, test dose-risposta, LOGO per coppia di brani sui giudizi.
EMB_SOURCES = {"muq": "cache/muq", "mert": "cache/emb_mert", "clap": "cache/emb_clap"}
DOSE_EMB_DIR = {"muq": "cache/muq_dose", "mert": "cache/emb_mert", "clap": "cache/emb_clap"}
N_BOOT_SEAM = 40
N_BOOT_STACK = 200
STACK_FEATURES = ("audiobox_delta", "songeval_delta", "surprisal_peak_z", "seam_logit", "technical")


def _emb_table(out: Path, ids: list, src: str, dose: bool = False) -> dict:
    d = out / (DOSE_EMB_DIR[src] if dose else EMB_SOURCES[src])
    res = {}
    for i in ids:
        p = d / f"{i}.npz"
        if p.exists():
            z = np.load(p)
            res[i] = (z["compact"].astype(np.float64), z["full"].astype(np.float32))
    return res


def _feat(tab_by_src: dict, srcs: tuple, ids: list, kind: str) -> np.ndarray:
    rows = []
    for i in ids:
        parts = [tab_by_src[s][i][0 if kind == "compact" else 1] for s in srcs]
        rows.append(np.concatenate(parts))
    return np.array(rows)


def _cv_oof(x, y, folds, c, pca):
    oof = np.full(len(y), np.nan)
    for f in sorted(set(folds.tolist())):
        tr, te = folds != f, folds == f
        if len(set(y[tr].tolist())) < 2:
            continue
        oof[te] = C.fit_linear_seam_model(x[tr], y[tr], c=c, pca=pca).proba(x[te])
    return oof


def _boot_models(x, y, groups, c, pca, n_boot, seed=0) -> list:
    """Bootstrap a grappolo per brano: si ricampionano i brani (con le loro giunte)."""
    rng = np.random.default_rng(seed)
    base = C.fit_linear_seam_model(x, y, c=c, pca=pca)
    xt = x if base.pca_components is None else (np.asarray(x, dtype=np.float64) - base.pca_mean) @ base.pca_components.T
    ug = np.array(sorted(set(groups.tolist())))
    models = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        m = C.fit_linear_seam_model(xt[idx], y[idx], c=c, pca=None)
        m.pca_mean, m.pca_components = base.pca_mean, base.pca_components
        models.append(m)
    return [base] + models


def _seam_proba_ens(models: list, x) -> tuple:
    ps = np.array([m.proba(x) for m in models])  # (B, n)
    return ps[0], ps[1:].std(0) if len(ps) > 1 else np.zeros(ps.shape[1])


# Giunte "ambigue" escluse da addestramento e AUC (v3, dai libri): un cut tra
# tonalità lontane non è un errore per Brewster & Broughton (How to DJ Right,
# cap. 15 "How to avoid key clashes": "Cut them, don't blend them ... use the
# cut to purposefully change key"); lo scontro di tonalità conta nei blend.
AMBIGUOUS_KINDS = ("x_key",)


def stage_train2(out: Path):
    tracks, recs = load_plan(out)
    recs = [r for r in recs if r["kind"] not in AMBIGUOUS_KINDS]
    byid = {r["id"]: r for r in recs}
    all_ids = [r["id"] for r in recs]
    tabs = {s: _emb_table(out, all_ids, s) for s in EMB_SOURCES}
    tabs = {s: t for s, t in tabs.items() if len(t) >= int(os.environ.get("DJC_MIN_EMB", "60"))}
    log("train2: embedding disponibili " + ", ".join(f"{s}={len(t)}" for s, t in tabs.items()))
    common = sorted(set.intersection(*[set(t) for t in tabs.values()])) if tabs else []
    combos = [(s,) for s in tabs]
    if "muq" in tabs and "mert" in tabs:
        combos.append(("muq", "mert"))
    if len(tabs) == 3:
        combos.append(("muq", "mert", "clap"))
    if "mert" in tabs and "clap" in tabs:
        combos.append(("mert", "clap"))
    cfgs = {"compact": (1.0, None), "full_pca32": (0.1, 32)}
    comp = {}
    y_c = np.array([byid[i]["label"] for i in common])
    f_c = np.array([byid[i]["fold"] for i in common])
    common_set = set(common)
    for srcs in combos:
        own = sorted(set.intersection(*[set(tabs[s_]) for s_ in srcs]))
        y_o = np.array([byid[i]["label"] for i in own])
        f_o = np.array([byid[i]["fold"] for i in own])
        is_c = np.array([i in common_set for i in own])
        for kind, (cval, pca) in cfgs.items():
            if len(common) < int(os.environ.get("DJC_MIN_EMB", "60")):
                continue
            x = _feat(tabs, srcs, common, kind)
            oof = _cv_oof(x, y_c, f_c, cval, pca)
            # stessi esempi di test (comuni), ma addestramento su tutti gli esempi
            # disponibili per quella sorgente nei fold di training
            xo = _feat(tabs, srcs, own, kind)
            oof_o = _cv_oof(xo, y_o, f_o, cval, pca)
            comp["+".join(srcs) + ":" + kind] = {"auc_common": _auc(y_c, oof), "n_common": len(common),
                                                  "auc_common_trainall": _auc(y_o[is_c], oof_o[is_c]),
                                                  "auc_own": _auc(y_o, oof_o), "n_own": len(own)}
    # selezionabili solo le combinazioni con copertura >= 80% della massima (lo stacker v3 e il test
    # dose-risposta richiedono la giunta su quasi tutti gli esempi)
    max_own = max((v["n_own"] for v in comp.values()), default=0)
    selectable = {k: v for k, v in comp.items() if v["n_own"] >= 0.8 * max_own} or comp
    for k in comp:
        comp[k]["selectable"] = k in selectable
    best = max(selectable, key=lambda k: selectable[k]["auc_common_trainall"]) if comp else "muq:full_pca32"
    srcs = tuple(best.split(":")[0].split("+"))
    kind = best.split(":")[1]
    cval, pca = cfgs[kind]
    ids = sorted(set.intersection(*[set(tabs[s]) for s in srcs]))
    x = _feat(tabs, srcs, ids, kind)
    y = np.array([byid[i]["label"] for i in ids])
    folds = np.array([byid[i]["fold"] for i in ids])
    tr_of = np.array([byid[i]["a"] for i in ids])
    kinds = np.array([byid[i]["kind"] for i in ids])
    v12 = np.array([(byid[i]["a"] in V12_TRACKS) or (byid[i]["b"] in V12_TRACKS) or (byid[i].get("bass_from") in V12_TRACKS) for i in ids])
    oof = _cv_oof(x, y, folds, cval, pca)
    per_kind = {}
    for k in sorted(set(kinds.tolist()) - set(C.POSITIVE_KINDS)):
        m = (kinds == k) | np.isin(kinds, [p for p in C.POSITIVE_KINDS if p.startswith("x_") == k.startswith("x_")])
        per_kind[k] = _auc(y[m], oof[m])
    ens_all = _boot_models(x, y, tr_of, cval, pca, N_BOOT_SEAM, seed=1)
    ens_nov12 = _boot_models(x[~v12], y[~v12], tr_of[~v12], cval, pca, N_BOOT_SEAM, seed=2)
    fold_models = {str(f): C.fit_linear_seam_model(x[folds != f], y[folds != f], c=cval, pca=pca).to_json()
                   for f in sorted(set(folds.tolist()))}
    seam_oof = dict(zip(ids, oof.tolist()))
    # ---- stacking sul sottoinsieme lungo (tutti i critici disponibili)
    nb = build_nulls(out)
    qa2 = json.loads((out / "cache/qa2.json").read_text()) if (out / "cache/qa2.json").exists() else {}
    rows = []
    for r in nb["synth_rows"]:
        if r["id"] not in byid:
            continue
        p = seam_oof.get(r["id"])
        tq = C.technical_from_qa(qa2.get(r["id"]) if "checks" in (qa2.get(r["id"]) or {}) else None, only=C.TECHNICAL_V2_CHECKS)
        feat = {"audiobox_delta": r.get("audiobox_delta"), "songeval_delta": r.get("songeval_delta"),
                "surprisal_peak_z": r.get("surprisal_peak_z"),
                "seam_logit": None if p is None or not np.isfinite(p) else float(np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))),
                "technical": tq.get("score") if tq.get("status") == "ok" else None}
        rows.append({"id": r["id"], "label": r["label"], "kind": r["kind"], "fold": byid[r["id"]]["fold"], "a": byid[r["id"]]["a"], **feat})
    avail = [f for f in STACK_FEATURES if sum(rw[f] is not None and np.isfinite(rw[f]) for rw in rows) >= 0.8 * max(1, len(rows))]
    full_rows = [rw for rw in rows if all(rw[f] is not None and np.isfinite(rw[f]) for f in avail)]
    stack: dict[str, Any] = {"features": avail, "n": len(full_rows)}
    if len(full_rows) >= 30 and avail:
        xs = np.array([[rw[f] for f in avail] for rw in full_rows])
        ys = np.array([rw["label"] for rw in full_rows])
        fs = np.array([rw["fold"] for rw in full_rows])
        gs = np.array([rw["a"] for rw in full_rows])
        oof_s = _cv_oof(xs, ys, fs, 1.0, None)
        stack["auc_oof_stacked"] = _auc(ys, oof_s)
        stack["auc_single"] = {f: _auc(ys, xs[:, j] * (-1 if f == "surprisal_peak_z" else 1)) for j, f in enumerate(avail)}
        stack["mean_equal_weight_auc"] = _auc(ys, np.mean([(xs[:, j] - xs[:, j].mean()) / (xs[:, j].std() + 1e-9) * (-1 if f == "surprisal_peak_z" else 1)
                                                           for j, f in enumerate(avail)], axis=0))
        # calibrazione: affidabilità in 5 bin dell'OOF
        bins = np.clip((oof_s * 5).astype(int), 0, 4)
        stack["reliability"] = [{"bin": b, "n": int((bins == b).sum()), "mean_pred": float(oof_s[bins == b].mean()) if (bins == b).any() else None,
                                 "frac_pos": float(ys[bins == b].mean()) if (bins == b).any() else None} for b in range(5)]
        stack["brier_oof"] = float(np.mean((oof_s - ys) ** 2))
        ens_stack = _boot_models(xs, ys, gs, 1.0, None, N_BOOT_STACK, seed=3)
        stack["models"] = [m.to_json() for m in ens_stack]
        stack["impute"] = {f: float(np.median(xs[:, j])) for j, f in enumerate(avail)}
        stack["coef_standardized"] = dict(zip(avail, ens_stack[0].coef.tolist()))
        stack["fold_models"] = {str(f): C.fit_linear_seam_model(xs[fs != f], ys[fs != f], c=1.0).to_json() for f in sorted(set(fs.tolist()))}
    C.write_json(out / "models/v2_models.json", {
        "embedding_comparison": comp, "chosen": best, "seam_cv_auc": _auc(y, oof), "seam_cv_per_kind": per_kind,
        "n_seam_train": int(len(y)), "seam_ensemble_all": [m.to_json() for m in ens_all],
        "seam_ensemble_no_v12": [m.to_json() for m in ens_nov12], "seam_fold_models": fold_models,
        "stack": stack, "track_fold": {t.slug: t.fold for t in tracks.values()},
        "selection": "solo CV per brano su dati sintetici; nessun giudizio umano usato"})
    C.write_json(out / "seam_oof_v2.json", seam_oof)
    log(f"train2: scelto {best} AUC {comp.get(best, {}).get('auc_common')} (n_common={len(common)}); seam n={len(y)} AUC {_auc(y, oof):.3f}; stacking {stack.get('auc_oof_stacked')}")


def _v2_load(models_dir: Path) -> dict:
    d = json.loads((models_dir / "models/v2_models.json").read_text())
    d["_ens_all"] = [C.LinearSeamModel.from_json(m) for m in d["seam_ensemble_all"]]
    d["_ens_nov12"] = [C.LinearSeamModel.from_json(m) for m in d["seam_ensemble_no_v12"]]
    d["_fold"] = {int(k): C.LinearSeamModel.from_json(m) for k, m in d["seam_fold_models"].items()}
    st = d["stack"]
    d["_stack"] = [C.LinearSeamModel.from_json(m) for m in st.get("models", [])]
    d["_stack_fold"] = {int(k): C.LinearSeamModel.from_json(m) for k, m in st.get("fold_models", {}).items()}
    return d


def _seam_x(v2: dict, per_src: dict) -> Optional[np.ndarray]:
    srcs = v2["chosen"].split(":")[0].split("+")
    kind = v2["chosen"].split(":")[1]
    try:
        return np.concatenate([np.asarray(per_src[s][0 if kind == "compact" else 1], dtype=np.float64) for s in srcs])
    except KeyError:
        return None


def _stack_predict(v2: dict, feats: dict, models: Optional[list] = None) -> tuple:
    st = v2["stack"]
    models = models if models is not None else v2["_stack"]
    if not models:
        return float("nan"), float("nan"), []
    imputed = [f for f in st["features"] if feats.get(f) is None or not np.isfinite(feats.get(f))]
    x = np.array([[feats[f] if f not in imputed else st["impute"][f] for f in st["features"]]])
    ps = np.array([m.proba(x)[0] for m in models])
    return float(ps[0]), float(ps[1:].std()) if len(ps) > 1 else float("nan"), imputed


def _logit(p):
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return float(np.log(p / (1 - p)))


def v2_item_scores(it: C.TransitionItem, *, ab: dict, mq: dict, emb_items: dict, mg_path: Path, qa2: dict, nulls: dict,
                   v2: dict, use_no_v12: bool) -> dict:
    """Punteggi v2 di un item: parti, stacking calibrato e incertezza bootstrap."""
    parts: dict[str, Any] = {}
    feats: dict[str, Any] = {f: None for f in STACK_FEATURES}
    if it.item_id in ab:
        r = _audiobox_rel(ab[it.item_id])
        feats["audiobox_delta"] = r.get("delta_vs_sides")
        r["percentile_vs_natural"] = C.empirical_percentile(r.get("delta_vs_sides"), nulls["audiobox_delta"]) if r.get("delta_vs_sides") is not None else None
        parts["audiobox"] = r
    if it.item_id in mq and mq[it.item_id].get("songeval"):
        r = _songeval_rel(mq[it.item_id]["songeval"])
        feats["songeval_delta"] = r.get("delta_vs_sides")
        r["percentile_vs_natural"] = C.empirical_percentile(r.get("delta_vs_sides"), nulls["songeval_delta"]) if r.get("delta_vs_sides") is not None else None
        parts["songeval"] = r
    if mg_path.exists():
        z = np.load(mg_path)
        stt = C.surprisal_seam_stats(z["nll"].sum(0), fps=C.MUSICGEN_FPS, start_s=it.start_s, switch_s=it.switch_s, end_s=it.end_s,
                                     offset_s=float(z["offset_s"]))
        if stt.get("status") == "ok":
            feats["surprisal_peak_z"] = stt["peak_z"]
            stt["percentile_vs_natural"] = C.empirical_percentile(stt["peak_z"], nulls["surprisal_peak_z"])
        parts["surprisal"] = stt
    # (c) per ogni punto della giunta: embedding scelti -> ensemble bootstrap
    pts = {}
    src_pts = {s: (emb_items.get(s, {}).get(it.item_id) or {}) for s in EMB_SOURCES}
    if it.item_id in mq:
        src_pts["muq"] = mq[it.item_id].get("seam", {})
    ens = v2["_ens_nov12"] if use_no_v12 else v2["_ens_all"]
    chosen_srcs = v2["chosen"].split(":")[0].split("+")
    for name in (src_pts.get(chosen_srcs[0]) or {}):
        per_src = {s: (src_pts[s][name]["compact"], src_pts[s][name]["full"]) for s in src_pts if name in src_pts[s]}
        x = _seam_x(v2, per_src)
        if x is None:
            continue
        p, sd = _seam_proba_ens(ens, x[None])
        pts[name] = {"p": float(p[0]), "boot_sd": float(sd[0])}
    if pts:
        worst = min(pts, key=lambda k: pts[k]["p"])
        feats["seam_logit"] = _logit(pts[worst]["p"])
        parts["seam"] = {"p": pts[worst]["p"], "boot_sd": pts[worst]["boot_sd"], "weakest_point": worst, "points": pts,
                         "model": "no_v12_tracks" if use_no_v12 else "all"}
    q = qa2.get(it.item_id)
    t = C.technical_from_qa(q if q and "checks" in q else None, only=C.TECHNICAL_V2_CHECKS)
    if t.get("status") == "ok":
        feats["technical"] = t["score"]
    parts["technical"] = t
    # v3: media a pesi uguali dei critici che superano il test dose-risposta
    # (giunta v2, sorpresa, tecnico ritmo+loudness); Audiobox/SongEval esclusi
    # perché premiano il salto di volume (vedi dose.json). Scelta su dati sintetici.
    v3parts = {}
    if "seam" in parts:
        v3parts["seam"] = (parts["seam"]["p"], max(parts["seam"]["boot_sd"], 0.02))
    sp = (parts.get("surprisal") or {})
    if sp.get("percentile_vs_natural") is not None and np.isfinite(sp["percentile_vs_natural"]):
        v3parts["surprisal"] = (1.0 - sp["percentile_vs_natural"], C.percentile_sd(sp["peak_z"], sp.get("peak_z_sd", 0.1), nulls["surprisal_peak_z"]))
    if parts["technical"].get("status") == "ok":
        v3parts["technical"] = (parts["technical"]["score"], parts["technical"]["sd"])
    v3 = C.combine({k: {"score": a, "sd": b} for k, (a, b) in v3parts.items()},
                   weights={"seam": 1.0, "surprisal": 1.0, "technical": 1.0})
    p, sd, imputed = _stack_predict(v2, feats)
    # incertezza totale: bootstrap dello stacker + propagazione del bootstrap di (c)
    sd_tot = float(np.sqrt(sd ** 2 + (parts.get("seam", {}).get("boot_sd", 0.0) * 0.25) ** 2)) if np.isfinite(sd) else float("nan")
    return {"score": v3.get("score"), "uncertainty": v3.get("sd"), "disagreement_sd": v3.get("disagreement_sd"),
            "score_definition": "v3: media di P(giunta) v2, 1-percentile sorpresa, tecnico (ritmo+loudness)", "v3_used": v3.get("used"),
            "stacked_v2": p, "stacked_v2_sd": sd_tot, "imputed_features": imputed, "features": feats, "parts": parts}


def stage_score2(out: Path, models_dir: Optional[Path] = None):
    models_dir = models_dir or out
    items = load_items(out)
    ab = _json(out / "cache/audiobox_items.json")
    mq = _json(out / "cache/muq_items.json")
    qa2 = _json(out / "cache/qa2.json")
    emb_items = {s: _json(out / f"cache/emb_{s}_items.json") for s in ("mert", "clap")}
    v2 = _v2_load(models_dir)
    nulls = build_nulls(models_dir)["nulls"]
    res = {}
    for it in items:
        uses_v12 = any(s in (it.a_audio or "") + (it.b_audio or "") for s in ("pool_v8", "pool_v10", "pool_v12", "reminder_to_too_many_nights"))
        r = v2_item_scores(it, ab=ab, mq=mq, emb_items=emb_items, mg_path=out / "cache/musicgen" / (it.item_id.replace("/", "__") + ".npz"),
                           qa2=qa2, nulls=nulls, v2=v2, use_no_v12=uses_v12)
        res[it.item_id] = {"item": {k: getattr(it, k) for k in ("dataset", "kind", "group", "label", "wav_sha256")}, **r}
    C.write_json(out / "scores_v2.json", {"schema_version": C.DJ_CRITIC_VERSION + "+v2", "chosen_embedding": v2["chosen"],
                                          "stack_features": v2["stack"].get("features"), "items": res})
    log(f"score2: {len(res)} item")


def _json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def stage_dose(out: Path):
    """Sensibilità dose-risposta: ogni critico deve peggiorare in modo monotono."""
    recs = load_dose(out)
    if not recs:
        return
    ab = _jsonl(out / "cache/audiobox_dose.jsonl")
    se = _jsonl(out / "cache/songeval_dose.jsonl")
    qa2 = _json(out / "cache/qa2.json")
    nulls = build_nulls(out)["nulls"]
    v1 = json.loads((out / "models/seam_models.json").read_text()) if (out / "models/seam_models.json").exists() else None
    v2 = _v2_load(out) if (out / "models/v2_models.json").exists() else None
    tabs = {s: _emb_table(out, [r["id"] for r in recs], s, dose=True) for s in EMB_SOURCES}
    craft = _json(out / "cache/craft.json")
    craft_nulls = craft_null_dists(out, craft)
    v3m = _v3_load(out) if (out / "models/v3_models.json").exists() else None
    rows = []
    for r in recs:
        g: dict[str, Any] = {"id": r["id"], "base": r["base"], "family": r["family"], "level": r["level"], "fold": r["fold"]}
        feats = {f: None for f in STACK_FEATURES}
        if r["id"] in ab:
            d = _audiobox_rel(ab[r["id"]]["scores"]).get("delta_vs_sides")
            feats["audiobox_delta"] = d
            g["audiobox"] = C.empirical_percentile(d, nulls["audiobox_delta"]) if d is not None else None
        if r["id"] in se:
            d = _songeval_rel(se[r["id"]]["scores"]).get("delta_vs_sides")
            feats["songeval_delta"] = d
            g["songeval"] = C.empirical_percentile(d, nulls["songeval_delta"]) if d is not None else None
        p = out / "cache/musicgen" / f"{r['id']}.npz"
        if p.exists():
            st = C.surprisal_seam_stats(np.load(p)["nll"].sum(0), fps=C.MUSICGEN_FPS, start_s=12.0, switch_s=12.0, end_s=12.0)
            if st.get("status") == "ok":
                feats["surprisal_peak_z"] = st["peak_z"]
                g["surprisal"] = 1.0 - C.empirical_percentile(st["peak_z"], nulls["surprisal_peak_z"])
        # (c) v1: modello del fold che esclude il brano della base
        if v1 and r["id"] in tabs["muq"]:
            m = C.LinearSeamModel.from_json(v1["fold_models"][str(r["fold"])])
            x = tabs["muq"][r["id"]][0 if v1["feature_set"] == "compact" else 1]
            g["seam_v1"] = float(m.proba(np.asarray(x, dtype=np.float64)[None])[0])
        if v2:
            per_src = {s: tabs[s][r["id"]] for s in tabs if r["id"] in tabs[s]}
            x = _seam_x(v2, per_src)
            if x is not None and r["fold"] in v2["_fold"]:
                pv = float(v2["_fold"][r["fold"]].proba(x[None])[0])
                g["seam_v2"] = pv
                feats["seam_logit"] = _logit(pv)
        q = qa2.get(r["id"])
        t = C.technical_from_qa(q if q and "checks" in q else None, only=C.TECHNICAL_V2_CHECKS)
        if t.get("status") == "ok":
            g["technical"] = t["score"]
            feats["technical"] = t["score"]
        if v2 and v2["_stack_fold"].get(r["fold"]) is not None:
            g["stack_v2"] = _stack_predict(v2, feats, [v2["_stack_fold"][r["fold"]]])[0]
        # v3: misure del mestiere (bontà = 1 - percentile tra le giunte naturali) e ritmo/loudness continui di dj_qa
        cf = craft.get(r["id"]) or {}
        if "error" not in cf and cf:
            for k, v in C.craft_goodness(cf, craft_nulls).items():
                g[f"craft_{k}"] = v
        if q and "checks" in q:
            bt = ((((q["checks"].get("rhythm") or {}).get("methods") or {}).get("beat_this_mix")) or {})
            if bt.get("beat_phase_offset_at_switch_ms") is not None:
                g["bt_beat_offset"] = -abs(float(bt["beat_phase_offset_at_switch_ms"]))
            lj = (q["checks"].get("loudness_jump") or {}).get("max_abs_jump_db")
            if lj is not None:
                g["qa_loud_jump"] = -float(lj)
        if v3m is not None:
            sp = _stack3_predict(v3m, g, fold=r["fold"])
            if sp is not None:
                g["stack_v3"] = sp
        v3p = [g.get(k) for k in ("seam_v2", "surprisal", "technical") if g.get(k) is not None and np.isfinite(g.get(k))]
        g["v3"] = float(np.mean(v3p)) if v3p else None
        v1parts = [g.get(k) for k in ("surprisal", "seam_v1", "technical")] + ([np.nanmean([v for v in (g.get("audiobox"), g.get("songeval")) if v is not None])] if (g.get("audiobox") is not None or g.get("songeval") is not None) else [])
        v1parts = [v for v in v1parts if v is not None and np.isfinite(v)]
        g["combined_v1"] = float(np.mean(v1parts)) if v1parts else None
        rows.append(g)
    critics = ["audiobox", "songeval", "surprisal", "seam_v1", "seam_v2", "technical", "bt_beat_offset", "qa_loud_jump"] + \
        [f"craft_{k}" for k in C.CRAFT_FEATURES] + ["combined_v1", "stack_v2", "v3", "stack_v3"]
    summary: dict[str, Any] = {}
    for fam in ("phase", "gain", "key"):
        summary[fam] = {}
        for cname in critics:
            per_base = []
            curves = {}
            for b in sorted(set(r["base"] for r in rows)):
                rr = [r for r in rows if r["base"] == b and (r["family"] == fam or r["family"] == "base")]
                if fam == "key":  # il livello 0 della tonalità è la versione con vocoder, non la base
                    rr = [r for r in rows if r["base"] == b and r["family"] == fam]
                lv = [r["level"] for r in rr]
                gv = [r.get(cname) if r.get(cname) is not None else np.nan for r in rr]
                mono = C.monotonicity(lv, gv)
                if mono["n"] >= 3:
                    per_base.append(mono)
                    for l, v in zip(lv, gv):
                        curves.setdefault(l, []).append(v)
            if per_base:
                summary[fam][cname] = {
                    "n_bases": len(per_base),
                    "mean_rho": float(np.nanmean([m["rho"] for m in per_base])),
                    "frac_bases_rho_neg": float(np.mean([m["rho"] < 0 for m in per_base])),
                    "frac_non_increasing_steps": float(np.nanmean([m["frac_non_increasing_steps"] for m in per_base])),
                    "mean_curve": {str(k): float(np.nanmean(v)) for k, v in sorted(curves.items())}}
    word = {}
    for cname in critics:
        diffs = []
        for b in sorted(set(r["base"] for r in rows)):
            c0 = [r.get(cname) for r in rows if r["base"] == b and r["family"] == "word" and r["level"] == 0.0]
            c1 = [r.get(cname) for r in rows if r["base"] == b and r["family"] == "word" and r["level"] == 1.0]
            if c0 and c1 and c0[0] is not None and c1[0] is not None:
                diffs.append(c0[0] - c1[0])
        if diffs:
            word[cname] = {"n_pairs": len(diffs), "frac_word_cut_worse": float(np.mean([d > 0 for d in diffs])), "mean_drop": float(np.mean(diffs))}
    summary["word_cut_binary"] = word
    C.write_json(out / "dose.json", {"levels": C.DOSE_LEVELS, "rows": rows, "summary": summary,
                                     "note": "bontà: percentile vs giunte naturali (a, b), P(plausibile) (c, stack), mappa QA (d); attesa: rho < 0"})
    log("dose: fatto")


def stage_validate2(out: Path):
    """Validazione v2: stessi insiemi della v1 + LOGO per coppia di brani."""
    sc2 = json.loads((out / "scores_v2.json").read_text())["items"]
    sc1 = json.loads((out / "scores.json").read_text())["items"] if (out / "scores.json").exists() else {}
    items = {it.item_id: it for it in load_items(out)}
    keys = {
        "v3_robust": lambda k: sc2[k]["score"],
        "v2_stacked": lambda k: sc2[k]["stacked_v2"],
        "v2_seam": lambda k: (sc2[k]["parts"].get("seam") or {}).get("p"),
        "v2_audiobox_pct": lambda k: (sc2[k]["parts"].get("audiobox") or {}).get("percentile_vs_natural"),
        "v2_songeval_pct": lambda k: (sc2[k]["parts"].get("songeval") or {}).get("percentile_vs_natural"),
        "v2_surprisal_good": lambda k: (lambda z: None if z is None else -z)((sc2[k]["parts"].get("surprisal") or {}).get("peak_z")),
        "v2_technical_cliponly": lambda k: (sc2[k]["parts"].get("technical") or {}).get("score"),
        "v1_combined": lambda k: sc1[k]["combined"]["score"] if k in sc1 else None,
    }

    def val(k, key):
        try:
            v = keys[key](k)
        except (KeyError, TypeError):
            return float("nan")
        return float("nan") if v is None else float(v)

    sets = {}
    s1 = [(k, v["item"]["label"]["score"]) for k, v in sc2.items() if v["item"]["dataset"] == "dj_gallery_v1" and v["item"]["label"]]
    sets["S1_gallery_v1_scores"] = s1
    sets["S2_v10_verbal_ordinal"] = [(k, v["item"]["label"]["ordinal"]) for k, v in sc2.items() if v["item"]["dataset"] == "v10_reconstructed"]
    sets["S3_gallery_ge7_vs_v12_main"] = [(k, 2.0) for k, s in s1 if s >= 7] + [(k, 1.0) for k, v in sc2.items() if v["item"]["dataset"] == "v12_release" and v["item"]["label"]]
    res: dict[str, Any] = {"sets": {}, "logo": {}}
    for name, pairs in sets.items():
        ids = [k for k, _ in pairs]
        truth = np.array([t for _, t in pairs], dtype=float)
        per = {}
        for key in keys:
            pred = np.array([val(k, key) for k in ids])
            pa = C.pairwise_accuracy(pred, truth)
            per[key] = {"n_pairs": pa["n_pairs"], "pairwise_accuracy": pa["accuracy"],
                        "bootstrap95_items": C.bootstrap_pairwise_ci(pred, truth) if pa["n_pairs"] else None,
                        "spearman": C.spearman_with_permutation(pred, truth, n_perm=5000) if name != "S3_gallery_ge7_vs_v12_main" else None}
        res["sets"][name] = {"items": ids, "truth": truth.tolist(), "critics": per}
    # ---- LOGO per coppia di brani: si sceglie (o si combina) sui gruppi restanti, si valuta sul gruppo escluso
    pool = []
    for name in ("S1_gallery_v1_scores", "S2_v10_verbal_ordinal"):
        tv = np.array([t for _, t in sets[name]], dtype=float)
        rk = (tv - tv.min()) / (tv.max() - tv.min() + 1e-9)
        for (k, _), r in zip(sets[name], rk):
            it = items[k]
            pool.append({"id": k, "set": name, "y": float(r), "group": "|".join(sorted([it.a_track_id or "", it.b_track_id or ""]))})
    for k, v in sc2.items():
        if v["item"]["dataset"] == "v12_release" and v["item"]["label"]:
            it = items[k]
            pool.append({"id": k, "set": "S3_v12", "y": 0.0, "group": "|".join(sorted([it.a_track_id or "", it.b_track_id or ""]))})
    cand = [k for k in keys if k != "v1_combined"]
    X = np.array([[val(p["id"], c) for c in cand] for p in pool])
    Y = np.array([p["y"] for p in pool])
    G = np.array([p["group"] for p in pool])
    S = np.array([p["set"] for p in pool])
    pred_sel = np.full(len(Y), np.nan)
    pred_ridge = np.full(len(Y), np.nan)
    chosen_log = []
    for g in sorted(set(G.tolist())):
        tr, te = G != g, G == g
        # (i) scegli il critico singolo con l'accuratezza a coppie migliore sui gruppi di training (dentro ogni insieme)
        best, best_acc = None, -1
        for j, c in enumerate(cand):
            accs = []
            for sname in set(S[tr].tolist()):
                m = tr & (S == sname)
                pa = C.pairwise_accuracy(X[m, j], Y[m])
                if pa["n_pairs"]:
                    accs.append(pa["accuracy"])
            a = np.mean(accs) if accs else -1
            if a > best_acc:
                best, best_acc = j, a
        pred_sel[te] = X[te, best]
        chosen_log.append({"held_out_group": g[:20], "chosen": cand[best], "train_acc": best_acc})
        # (ii) ridge sui critici standardizzati (valori mancanti -> media)
        xm = np.nanmean(X[tr], axis=0)
        xm = np.where(np.isfinite(xm), xm, 0.0)
        xtr = np.where(np.isfinite(X[tr]), X[tr], xm)
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
        from sklearn.linear_model import Ridge
        rg = Ridge(alpha=3.0).fit((xtr - mu) / sd, Y[tr])
        xte = np.where(np.isfinite(X[te]), X[te], xm)
        pred_ridge[te] = rg.predict((xte - mu) / sd)
    for nm, pred in (("logo_select_best_single", pred_sel), ("logo_ridge", pred_ridge)):
        out_sets = {}
        for sname in ("S1_gallery_v1_scores", "S2_v10_verbal_ordinal"):
            m = S == sname
            pa = C.pairwise_accuracy(pred[m], Y[m])
            out_sets[sname] = {"n_pairs": pa["n_pairs"], "pairwise_accuracy": pa["accuracy"],
                               "bootstrap95_items": C.bootstrap_pairwise_ci(pred[m], Y[m]) if pa["n_pairs"] else None}
        m3 = (S == "S3_v12") | ((S == "S1_gallery_v1_scores") & np.isin(np.array([p["id"] for p in pool]), [k for k, s in s1 if s >= 7]))
        y3 = np.where(S == "S3_v12", 0.0, 1.0)
        pa = C.pairwise_accuracy(pred[m3], y3[m3])
        out_sets["S3_gallery_ge7_vs_v12_main"] = {"n_pairs": pa["n_pairs"], "pairwise_accuracy": pa["accuracy"]}
        res["logo"][nm] = out_sets
    res["logo"]["groups"] = len(set(G.tolist()))
    res["logo"]["selection_log"] = chosen_log
    res["logo"]["note"] = ("LOGO per coppia di brani: ogni gruppo (coppia A,B) è escluso a turno; la scelta del critico o i pesi ridge "
                           "si stimano sugli altri gruppi. Con 8–10 gruppi e un solo ascoltatore il risultato è indicativo.")
    C.write_json(out / "validation_v2.json", res)
    log("validate2: fatto")


# ======================================================================= v3
# v3 (21/10): critici del mestiere DSP + giunta (c) in un ensemble con pesi
# non negativi scelto e stimato SOLO su dati sintetici (CV per brano); critici
# anti-correlati nella dose-risposta esclusi prima dell'addestramento.
V3_DOSE_ANTI_RHO = 0.25      # rho medio >= +0.25 in una famiglia = anti-correlato -> escluso
V3_DOSE_INFO_RHO = -0.25
V3_DOSE_MIN_RISE = 0.05      # v3b: aumento minimo della bontà media (livello max − livello 0) per dire "anti"     # serve rho medio <= -0.25 in almeno una famiglia (o parola peggiore in >= 75%)
V3_CANDIDATES = {            # nome nel dose.json -> feature dello stacker
    "seam_v2": "seam", "surprisal": "surprisal", "audiobox": "audiobox", "songeval": "songeval",
    **{f"craft_{k}": f"craft_{k}" for k in C.CRAFT_FEATURES},
}
N_BOOT_V3 = 200


def craft_null_dists(out: Path, craft: dict) -> dict:
    """Distribuzioni nulle delle misure del mestiere: giunte naturali del piano sintetico."""
    _, recs = load_plan(out)
    nat = [r["id"] for r in recs if r["kind"] in ("natural", "natural_vocoder")]
    return {k: [craft[i][k] for i in nat if i in craft and "error" not in craft[i] and craft[i].get(k) is not None
                and np.isfinite(craft[i][k])] for k in C.CRAFT_FEATURES}


def _dose_verdicts(out: Path) -> dict:
    """Esito del test dose-risposta per critico (criterio principale senza umani)."""
    d = _json(out / "dose.json")
    summ = d.get("summary", {})
    res = {}
    for name in V3_CANDIDATES:
        rhos = {fam: summ.get(fam, {}).get(name, {}).get("mean_rho") for fam in ("phase", "gain", "key")}
        w = summ.get("word_cut_binary", {}).get(name, {})
        info = [f for f, r in rhos.items() if r is not None and r <= V3_DOSE_INFO_RHO]
        if w.get("n_pairs", 0) >= 4 and w.get("frac_word_cut_worse", 0) >= 0.75:
            info.append("word")
        # v3b: "anti-correlato" richiede anche che la bontà salga davvero (curva media: livello massimo −
        # livello 0 >= +0,05). Una curva piatta (critico insensibile a quel difetto) con rho positivo da
        # rumore/pareggi non è anti-correlazione. Correzione decisa dopo aver visto la validazione v3a.
        rise = {}
        for fam in ("phase", "gain", "key"):
            mc = summ.get(fam, {}).get(name, {}).get("mean_curve") or {}
            if mc:
                ks = sorted(mc, key=float)
                rise[fam] = mc[ks[-1]] - mc[ks[0]]
        anti = [f for f, r in rhos.items() if r is not None and r >= V3_DOSE_ANTI_RHO and rise.get(f, 1.0) >= V3_DOSE_MIN_RISE]
        if w.get("n_pairs", 0) >= 4 and w.get("frac_word_cut_worse", 1) <= 0.25 and name.startswith("craft_vad"):
            anti.append("word")
        res[name] = {"rho": rhos, "rise": rise, "word": w, "informative_in": info, "anti_in": anti,
                     "keep": bool(info) and not anti}
    return res


def _v3_rows(out: Path, ids: list, byid: dict, seam_p: dict, craft: dict, nulls_craft: dict, synth_nulls: dict) -> list:
    rows = []
    srow = {r["id"]: r for r in synth_nulls["synth_rows"]}
    for i in ids:
        cf = craft.get(i)
        if not cf or "error" in cf:
            continue
        g = {f"craft_{k}": v for k, v in C.craft_goodness(cf, nulls_craft).items()}
        g["seam"] = seam_p.get(i)
        sr = srow.get(i, {})
        nl = synth_nulls["nulls"]
        g["surprisal"] = 1.0 - C.empirical_percentile(sr["surprisal_peak_z"], nl["surprisal_peak_z"]) if sr.get("surprisal_peak_z") is not None else None
        g["audiobox"] = C.empirical_percentile(sr["audiobox_delta"], nl["audiobox_delta"]) if sr.get("audiobox_delta") is not None else None
        g["songeval"] = C.empirical_percentile(sr["songeval_delta"], nl["songeval_delta"]) if sr.get("songeval_delta") is not None else None
        r = byid[i]
        rows.append({"id": i, "label": r["label"], "kind": r["kind"], "fold": r["fold"], "a": r["a"],
                     "v12": (r["a"] in V12_TRACKS) or (r["b"] in V12_TRACKS) or (r.get("bass_from") in V12_TRACKS), **g})
    return rows


def _xy(rows, feats):
    ok = [r for r in rows if all(r.get(f) is not None and np.isfinite(r[f]) for f in feats)]
    return ok, np.array([[r[f] for f in feats] for r in ok]), np.array([r["label"] for r in ok]), \
        np.array([r["fold"] for r in ok]), np.array([r["a"] for r in ok])


def _cv_nonneg(x, y, folds, c=1.0):
    oof = np.full(len(y), np.nan)
    for f in sorted(set(folds.tolist())):
        tr, te = folds != f, folds == f
        if len(set(y[tr].tolist())) < 2 or not te.any():
            continue
        oof[te] = C.fit_nonneg_logistic(x[tr], y[tr], c=c).proba(x[te])
    return oof


def _boot_nonneg(x, y, groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    ug = np.array(sorted(set(groups.tolist())))
    models = [C.fit_nonneg_logistic(x, y)]
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        models.append(C.fit_nonneg_logistic(x[idx], y[idx]))
    return models


def stage_train3(out: Path):
    tracks, recs = load_plan(out)
    byid = {r["id"]: r for r in recs}
    seam_p = _json(out / "seam_oof_v2.json")
    craft = _json(out / "cache/craft.json")
    nulls_craft = craft_null_dists(out, craft)
    nb = build_nulls(out)
    verdicts = _dose_verdicts(out)
    kept = [V3_CANDIDATES[k] for k, v in verdicts.items() if v["keep"]]
    log("train3: dose-risposta, critici tenuti: " + ", ".join(kept))
    rows = _v3_rows(out, [r["id"] for r in recs if r["kind"] not in AMBIGUOUS_KINDS], byid, seam_p, craft, nulls_craft, nb)
    kinds_all = np.array([r["kind"] for r in rows])
    # configurazioni candidate (solo feature tenute); confronto sugli stessi esempi
    craft_k = [f for f in kept if f.startswith("craft_")]
    cfgs = {"craft": craft_k, "craft+seam": craft_k + (["seam"] if "seam" in kept else []),
            "seam": ["seam"] if "seam" in kept else []}
    cfgs = {k: v for k, v in cfgs.items() if v}
    base_ok, *_ = _xy(rows, sorted(set(sum(cfgs.values(), []))))
    base_ids = {r["id"] for r in base_ok}
    sub = [r for r in rows if r["id"] in base_ids]
    comp = {}
    for name, feats in cfgs.items():
        _, x, y, f, _g = _xy(sub, feats)
        oof = _cv_nonneg(x, y, f)
        eq = np.mean([(x[:, j] - x[:, j].mean()) / (x[:, j].std() + 1e-9) for j in range(x.shape[1])], axis=0)
        comp[name] = {"features": feats, "n": int(len(y)), "auc_oof": _auc(y, oof), "auc_equal_weight": _auc(y, eq)}
    # sottoinsieme lungo: la sorpresa (e gli estetici se tenuti) aggiungono qualcosa?
    long_extra = [f for f in ("surprisal", "audiobox", "songeval") if f in kept]
    best_main = max(comp, key=lambda k: comp[k]["auc_oof"])
    long_cmp = {}
    if long_extra:
        feats0 = cfgs[best_main]
        lr, x0, y0, f0, _g = _xy(rows, feats0 + long_extra)
        x0b = x0[:, :len(feats0)]
        long_cmp = {"n": int(len(y0)), "auc_without": _auc(y0, _cv_nonneg(x0b, y0, f0)), "auc_with": _auc(y0, _cv_nonneg(x0, y0, f0)),
                    "extra": long_extra}
    final_feats = cfgs[best_main] + (long_extra if long_cmp and long_cmp["auc_with"] - long_cmp["auc_without"] >= 0.02 else [])
    fr, x, y, folds, groups = _xy(rows, final_feats)
    oof = _cv_nonneg(x, y, folds)
    kinds = np.array([r["kind"] for r in fr])
    per_kind = {}
    for k in sorted(set(kinds.tolist()) - set(C.POSITIVE_KINDS)):
        m = (kinds == k) | np.isin(kinds, [p for p in C.POSITIVE_KINDS if p.startswith("x_") == k.startswith("x_")])
        per_kind[k] = {"auc": _auc(y[m], oof[m]), "n_neg": int((kinds == k).sum())}
    single = {}
    for j, ft in enumerate(final_feats):
        single[ft] = {"auc": _auc(y, x[:, j]), "per_kind": {k: _auc(y[(kinds == k) | np.isin(kinds, C.POSITIVE_KINDS)],
                                                                   x[(kinds == k) | np.isin(kinds, C.POSITIVE_KINDS), j])
                                                           for k in sorted(set(kinds.tolist()) - set(C.POSITIVE_KINDS))}}
    bins = np.clip((oof * 5).astype(int), 0, 4)
    reliab = [{"bin": b, "n": int((bins == b).sum()), "mean_pred": float(oof[bins == b].mean()) if (bins == b).any() else None,
               "frac_pos": float(y[bins == b].mean()) if (bins == b).any() else None} for b in range(5)]
    v12 = np.array([r["v12"] for r in fr])
    ens = _boot_nonneg(x, y, groups, N_BOOT_V3, seed=31)
    ens_nov12 = _boot_nonneg(x[~v12], y[~v12], groups[~v12], N_BOOT_V3, seed=32)
    fold_models = {str(f): C.fit_nonneg_logistic(x[folds != f], y[folds != f]).to_json() for f in sorted(set(folds.tolist()))}
    # modello "rapido" (solo DSP, nessuna rete): per l'ottimizzazione in-process
    fast_feats = [f for f in final_feats if f.startswith("craft_")]
    _, xf, yf, ff, gf = _xy(rows, fast_feats)
    oof_fast = _cv_nonneg(xf, yf, ff)
    fast = {"features": fast_feats, "auc_oof": _auc(yf, oof_fast), "n": int(len(yf)),
            "models": [m.to_json() for m in _boot_nonneg(xf, yf, gf, N_BOOT_V3, seed=33)],
            "fold_models": {str(f): C.fit_nonneg_logistic(xf[ff != f], yf[ff != f]).to_json() for f in sorted(set(ff.tolist()))}}
    C.write_json(out / "models/v3_models.json", {
        "selection": "solo dati sintetici: critici filtrati dal test dose-risposta, configurazione scelta per AUC OOF (CV per brano); nessun giudizio umano",
        "dose_verdicts": verdicts, "kept": kept, "config_comparison": comp, "long_subset_comparison": long_cmp,
        "features": final_feats, "n": int(len(y)), "auc_oof": _auc(y, oof), "brier_oof": float(np.mean((oof - y) ** 2)),
        "per_kind": per_kind, "single_auc": single, "reliability": reliab,
        "coef_standardized": dict(zip(final_feats, ens[0].coef.tolist())),
        "models": [m.to_json() for m in ens], "models_no_v12": [m.to_json() for m in ens_nov12], "fold_models": fold_models,
        "craft_nulls": nulls_craft, "fast": fast, "flag_rules": C.CRAFT_FLAG_RULES,
        "seam_embedding": _json(out / "models/v2_models.json").get("chosen")})
    C.write_json(out / "seam_oof_v3_stack.json", {r["id"]: float(p) for r, p in zip(fr, oof)})
    log(f"train3: feature {final_feats}; n={len(y)} AUC OOF {_auc(y, oof):.3f}; rapido {fast['auc_oof']:.3f}; confronti {json.dumps({k: round(v['auc_oof'], 3) for k, v in comp.items()})}")


def _v3_load(models_dir: Path) -> dict:
    d = json.loads((models_dir / "models/v3_models.json").read_text())
    d["_ens"] = [C.LinearSeamModel.from_json(m) for m in d["models"]]
    d["_ens_nov12"] = [C.LinearSeamModel.from_json(m) for m in d["models_no_v12"]]
    d["_fold"] = {int(k): C.LinearSeamModel.from_json(m) for k, m in d["fold_models"].items()}
    d["_fast"] = [C.LinearSeamModel.from_json(m) for m in d["fast"]["models"]]
    d["_fast_fold"] = {int(k): C.LinearSeamModel.from_json(m) for k, m in d["fast"]["fold_models"].items()}
    return d


def _stack3_predict(v3: dict, g: dict, fold: Optional[int] = None, fast: bool = False):
    feats = v3["fast"]["features"] if fast else v3["features"]
    key = {"seam": "seam_v2"}
    vals = []
    for f in feats:
        v = g.get(f, g.get(key.get(f, f)))
        if v is None or not np.isfinite(v):
            return None
        vals.append(v)
    m = (v3["_fast_fold"] if fast else v3["_fold"]).get(fold) if fold is not None else (v3["_fast"] if fast else v3["_ens"])[0]
    return float(m.proba(np.array([vals]))[0])


def v3_item_score(g: dict, v3: dict, *, use_no_v12: bool, seam_boot: Optional[list] = None, fast: bool = False) -> dict:
    """Punteggio v3 di un item da bontà per critico ``g`` (stessa scala del training)."""
    feats = v3["fast"]["features"] if fast else v3["features"]
    ens = v3["_fast"] if fast else (v3["_ens_nov12"] if use_no_v12 else v3["_ens"])
    missing = [f for f in feats if g.get(f) is None or not np.isfinite(g.get(f))]
    if missing:
        return {"score": float("nan"), "uncertainty": float("nan"), "missing": missing}
    ps = []
    for b, m in enumerate(ens):
        vals = [g[f] if not (f == "seam" and seam_boot and b > 0) else seam_boot[1 + (b - 1) % (len(seam_boot) - 1)] for f in feats]
        ps.append(float(m.proba(np.array([vals]))[0]))
    contrib = {f: float(ens[0].coef[j] * (g[f] - ens[0].mean[j]) / ens[0].scale[j]) for j, f in enumerate(feats)}
    return {"score": ps[0], "uncertainty": float(np.std(ps[1:])) if len(ps) > 1 else float("nan"),
            "contributions_logit": contrib, "missing": []}


def stage_score3(out: Path, models_dir: Optional[Path] = None):
    models_dir = models_dir or out
    v3 = _v3_load(models_dir)
    sc2 = _json(out / "scores_v2.json").get("items", {})
    craft = _json(out / "cache/craft.json")
    v2 = _v2_load(models_dir)
    mq = _json(out / "cache/muq_items.json")
    emb_items = {s: _json(out / f"cache/emb_{s}_items.json") for s in ("mert", "clap")}
    res = {}
    for it in load_items(out):
        uses_v12 = any(s in (it.a_audio or "") + (it.b_audio or "") for s in ("pool_v8", "pool_v10", "pool_v12", "reminder_to_too_many_nights"))
        p2 = (sc2.get(it.item_id) or {}).get("parts", {})
        g: dict[str, Any] = {}
        cf = craft.get(it.item_id) or {}
        if cf and "error" not in cf:
            g.update({f"craft_{k}": v for k, v in C.craft_goodness(cf, v3["craft_nulls"]).items()})
        seam_boot = None
        if p2.get("seam"):
            g["seam"] = p2["seam"]["p"]
            # bootstrap della giunta sul punto più debole (stesso ensemble della v2)
            worst = p2["seam"]["weakest_point"]
            src_pts = {s: (emb_items.get(s, {}).get(it.item_id) or {}) for s in EMB_SOURCES}
            src_pts["muq"] = (mq.get(it.item_id) or {}).get("seam", {})
            per_src = {s: (src_pts[s][worst]["compact"], src_pts[s][worst]["full"]) for s in src_pts if worst in src_pts[s]}
            xs = _seam_x(v2, per_src)
            if xs is not None:
                ens = v2["_ens_nov12"] if uses_v12 else v2["_ens_all"]
                seam_boot = [float(m.proba(xs[None])[0]) for m in ens]
        sp = p2.get("surprisal") or {}
        if sp.get("percentile_vs_natural") is not None:
            g["surprisal"] = 1.0 - sp["percentile_vs_natural"]
        for k in ("audiobox", "songeval"):
            if (p2.get(k) or {}).get("percentile_vs_natural") is not None:
                g[k] = p2[k]["percentile_vs_natural"]
        full = v3_item_score(g, v3, use_no_v12=uses_v12, seam_boot=seam_boot)
        fast = v3_item_score(g, v3, use_no_v12=uses_v12, fast=True)
        is_cut = (it.end_s - it.start_s) < 0.5
        flags = C.craft_flags(cf, is_cut=is_cut) if cf and "error" not in cf else []
        # Huron (Sweet Anticipation p. 116; pp. 349-350): sorpresa del "quando" (fase dei battiti) separata
        # da quella del "cosa" (MusicGen); la seconda è leggibile come sorpresa di contenuto solo se la prima è bassa.
        bp = cf.get("beat_phase_ms") if cf else None
        g_split = {"when_beat_phase_ms": bp, "what_surprisal_goodness": g.get("surprisal"),
                   "reading": None if bp is None or g.get("surprisal") is None else
                   ("sorpresa di contenuto in schema metrico intatto" if bp <= C.CRAFT_FLAG_RULES["beat_phase_ms"][0] else "sorpresa metrica (battiti fuori griglia)")}
        res[it.item_id] = {"item": {k: getattr(it, k) for k in ("dataset", "kind", "group", "label", "wav_sha256")},
                           "score": full["score"], "uncertainty": full["uncertainty"], "score_fast": fast["score"],
                           "uncertainty_fast": fast["uncertainty"], "contributions_logit": full.get("contributions_logit"),
                           "goodness": g, "craft": cf, "flags": flags, "missing": full.get("missing"), "huron_when_what": g_split}
    C.write_json(out / "scores_v3.json", {"schema_version": C.DJ_CRITIC_VERSION + "+v3", "features": v3["features"],
                                          "fast_features": v3["fast"]["features"], "items": res})
    log(f"score3: {len(res)} item")


def _bootstrap_acc(pred, truth, n_boot=2000, seed=0):
    return C.bootstrap_pairwise_ci(pred, truth, n_boot=n_boot, seed=seed)


def stage_validate3(out: Path):
    """Validazione v3 sui giudizi umani (nessun peso stimato su di essi)."""
    sc3 = _json(out / "scores_v3.json")["items"]
    sc2 = _json(out / "scores_v2.json").get("items", {})
    sc1 = _json(out / "scores.json").get("items", {})
    items = {it.item_id: it for it in load_items(out)}
    keys = {
        "v3_ensemble": lambda k: sc3[k]["score"],
        "v3_fast_dsp": lambda k: sc3[k]["score_fast"],
        **{f"g_{f}": (lambda f: (lambda k: sc3[k]["goodness"].get(f)))(f)
           for f in ("seam", "surprisal", "audiobox", "songeval", *[f"craft_{c}" for c in C.CRAFT_FEATURES])},
        "n_flags_neg": lambda k: -len(sc3[k]["flags"]),
        "v2_v3robust": lambda k: sc2[k]["score"] if k in sc2 else None,
        "v1_combined": lambda k: sc1[k]["combined"]["score"] if k in sc1 else None,
    }

    def val(k, key):
        try:
            v = keys[key](k)
        except (KeyError, TypeError):
            return float("nan")
        return float("nan") if v is None else float(v)

    s1 = [(k, v["item"]["label"]["score"]) for k, v in sc3.items() if v["item"]["dataset"] == "dj_gallery_v1" and v["item"]["label"]]
    sets = {"S1_gallery_v1_scores": s1,
            "S2_v10_verbal_ordinal": [(k, v["item"]["label"]["ordinal"]) for k, v in sc3.items() if v["item"]["dataset"] == "v10_reconstructed"],
            "S3_gallery_ge7_vs_v12_main": [(k, 2.0) for k, s in s1 if s >= 7] + [(k, 1.0) for k, v in sc3.items() if v["item"]["dataset"] == "v12_release" and v["item"]["label"]]}
    res: dict[str, Any] = {"sets": {}, "pooled": {}, "v8_pairs": {}, "notes": [
        "Pesi e scelta dei critici stimati solo su giunte sintetiche (models/v3_models.json); qui nessun adattamento.",
        "Per gli item con brani V12 si usano l'ensemble della giunta e lo stacker addestrati senza quei brani.",
        "IC: bootstrap sugli item (2000 ricampionamenti); le coppie non sono indipendenti.",
        "S2 usa una codifica ordinale nostra [H] dei commenti verbali V10."]}
    for name, pairs in sets.items():
        ids = [k for k, _ in pairs]
        truth = np.array([t for _, t in pairs], dtype=float)
        per = {}
        for key in keys:
            pred = np.array([val(k, key) for k in ids])
            pa = C.pairwise_accuracy(pred, truth)
            k_ok = sum(p[2] for p in pa["pairs"])
            per[key] = {"n_items": int(np.isfinite(pred).sum()), "n_pairs": pa["n_pairs"], "pairwise_accuracy": pa["accuracy"],
                        "wilson95_pairs": C.wilson_interval(k_ok, pa["n_pairs"]) if pa["n_pairs"] else None,
                        "bootstrap95_items": _bootstrap_acc(pred, truth) if pa["n_pairs"] else None,
                        "spearman": C.spearman_with_permutation(pred, truth, n_perm=5000) if name != "S3_gallery_ge7_vs_v12_main" else None}
        res["sets"][name] = {"items": ids, "truth": truth.tolist(), "critics": per}
    # aggregato: coppie dentro ogni insieme, sommate (stima unica con bootstrap stratificato per insieme)
    rng = np.random.default_rng(7)
    for key in keys:
        ok = tot = 0.0
        per_set = []
        for name, pairs in sets.items():
            pred = np.array([val(k, key) for k, _ in pairs])
            truth = np.array([t for _, t in pairs], dtype=float)
            pa = C.pairwise_accuracy(pred, truth)
            ok += sum(p[2] for p in pa["pairs"])
            tot += pa["n_pairs"]
            per_set.append((pred, truth))
        boots = []
        for _ in range(2000):
            o = t = 0.0
            for pred, truth in per_set:
                idx = rng.integers(0, len(pred), len(pred))
                pa = C.pairwise_accuracy(pred[idx], truth[idx])
                o += sum(p[2] for p in pa["pairs"])
                t += pa["n_pairs"]
            if t:
                boots.append(o / t)
        res["pooled"][key] = {"n_pairs": int(tot), "pairwise_accuracy": ok / tot if tot else None,
                              "bootstrap95_items": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None}
    # V8: unica preferenza netta
    v8 = {}
    for k, v in sc3.items():
        if v["item"]["dataset"] == "v8_reconstructed":
            lab = v["item"]["label"]
            v8.setdefault(lab["pair_id"], {})[lab["candidate_label"]] = (k, lab["preference"])
    for pid, d in sorted(v8.items()):
        if set(d) != {"A", "B"}:
            continue
        pref = d["A"][1]
        res["v8_pairs"][pid] = {"preference": pref, **{key: {"A": val(d["A"][0], key), "B": val(d["B"][0], key)} for key in ("v3_ensemble", "v3_fast_dsp")}}
    # continuazione naturale V12 contro main (controllo: la continuazione dovrebbe vincere)
    nat = {}
    for k, v in sc3.items():
        if v["item"]["dataset"] in ("v12_release", "v12_natural_reference"):
            nat.setdefault(v["item"]["group"], {})[v["item"]["dataset"]] = k
    res["v12_natural_vs_main"] = {g: {key: {"natural": val(d["v12_natural_reference"], key), "main": val(d["v12_release"], key)}
                                      for key in ("v3_ensemble", "v3_fast_dsp")} for g, d in nat.items() if len(d) == 2}
    # LOGO per coppia di brani: scelta del critico singolo sui gruppi restanti (diagnostica: usa i giudizi)
    pool = []
    for name in ("S1_gallery_v1_scores", "S2_v10_verbal_ordinal"):
        tv = np.array([t for _, t in sets[name]], dtype=float)
        rk = (tv - tv.min()) / (tv.max() - tv.min() + 1e-9)
        for (k, _), r in zip(sets[name], rk):
            it = items[k]
            pool.append({"id": k, "set": name, "y": float(r), "group": "|".join(sorted([it.a_track_id or "", it.b_track_id or ""]))})
    for k, s in s1:
        pass
    for k, v in sc3.items():
        if v["item"]["dataset"] == "v12_release" and v["item"]["label"]:
            it = items[k]
            pool.append({"id": k, "set": "S3_v12", "y": 0.0, "group": "|".join(sorted([it.a_track_id or "", it.b_track_id or ""]))})
    cand = [k for k in keys if k not in ("v1_combined", "v2_v3robust")]
    X = np.array([[val(p["id"], c) for c in cand] for p in pool])
    Y = np.array([p["y"] for p in pool])
    G = np.array([p["group"] for p in pool])
    S = np.array([p["set"] for p in pool])
    pred_sel = np.full(len(Y), np.nan)
    chosen = []
    for g in sorted(set(G.tolist())):
        tr, te = G != g, G == g
        best, best_acc = 0, -1.0
        for j in range(len(cand)):
            accs = []
            for sname in ("S1_gallery_v1_scores", "S2_v10_verbal_ordinal"):
                m = tr & (S == sname)
                pa = C.pairwise_accuracy(X[m, j], Y[m])
                if pa["n_pairs"]:
                    accs.append(pa["accuracy"])
            a = float(np.mean(accs)) if accs else -1.0
            if a > best_acc:
                best, best_acc = j, a
        pred_sel[te] = X[te, best]
        chosen.append(cand[best])
    lg = {}
    for sname in ("S1_gallery_v1_scores", "S2_v10_verbal_ordinal"):
        m = S == sname
        pa = C.pairwise_accuracy(pred_sel[m], Y[m])
        lg[sname] = {"n_pairs": pa["n_pairs"], "pairwise_accuracy": pa["accuracy"],
                     "bootstrap95_items": _bootstrap_acc(pred_sel[m], Y[m]) if pa["n_pairs"] else None}
    from collections import Counter
    res["logo_select_single"] = {"groups": len(set(G.tolist())), "per_set": lg, "chosen": dict(Counter(chosen)),
                                 "note": "diagnostica: la scelta usa i giudizi dei gruppi restanti; non è il punteggio v3"}
    C.write_json(out / "validation_v3.json", res)
    log("validate3: fatto")


# ===================================================================== api
def _musicgen_items(out: Path, mg, items):
    cache = out / "cache/musicgen"
    cache.mkdir(parents=True, exist_ok=True)
    for it in items:
        p = cache / (it.item_id.replace("/", "__") + ".npz")
        if p.exists():
            continue
        dur = min(it.clip_dur_s, C.SURPRISAL_MAX_S)
        span = it.end_s - it.start_s
        side = max(4.0, (dur - span) / 2)
        s0 = max(0.0, min(it.start_s - side, it.clip_dur_s - dur))
        np.savez(p, nll=mg.frame_nll(it.read_clip(C.SR_MUSICGEN, s0, dur)).astype(np.float32), offset_s=s0)


def _emb_items(out: Path, runner, model: str, items):
    ip = out / f"cache/emb_{model}_items.json"
    res = json.loads(ip.read_text()) if ip.exists() else {}
    for it in items:
        if it.item_id in res:
            continue
        entry = {}
        for name, t in seam_points(it).items():
            s0 = min(max(0.0, t - C.EXAMPLE_S / 2), max(0.0, it.clip_dur_s - C.EXAMPLE_S))
            try:
                compact, full = runner.features(it.read_clip(runner.sr, s0, C.EXAMPLE_S), t - s0)
            except ValueError:
                continue
            entry[name] = {"t_s": t, "compact": compact.tolist(), "full": full.astype(np.float32).tolist()}
        res[it.item_id] = entry
        C.write_json(ip, res)


def _api_needs(models_dir: Path) -> dict:
    v2 = json.loads((models_dir / "models/v2_models.json").read_text())
    v3p = models_dir / "models/v3_models.json"
    feats = json.loads(v3p.read_text())["features"] if v3p.exists() else ["seam", "surprisal"]
    emb = v2["chosen"].split(":")[0].split("+") if "seam" in feats else []
    return {"features": feats, "emb": emb, "muq": "muq" in emb or "songeval" in feats, "musicgen": "surprisal" in feats,
            "audiobox": "audiobox" in feats}


def stage_api_np1(out: Path, threads: int, models_dir: Path):
    need = _api_needs(models_dir)
    items = load_items(out)
    t = {}
    t0 = time.time()
    if need["muq"]:
        runner = MuQRunner(threads)
        _muq_items(out, runner)
        del runner
    t["muq_s"] = time.time() - t0
    for s in need["emb"]:
        t0 = time.time()
        if s in ("mert", "clap"):
            _emb_items(out, MERTRunner(threads) if s == "mert" else CLAPRunner(threads), s, items)
        t[f"{s}_s"] = time.time() - t0
    t0 = time.time()
    if need["musicgen"]:
        _musicgen_items(out, C.MusicGenSurprisal(threads=threads), items)
    t["musicgen_s"] = time.time() - t0
    C.write_json(out / "api_timings_np1.json", t)


def stage_api_main(out: Path, threads: int, models_dir: Path):
    need = _api_needs(models_dir)
    items = load_items(out)
    t = _json(out / "api_timings_np1.json")
    t0 = time.time()
    if need["audiobox"]:
        c = audiobox_predictor(threads)
        path = out / "cache/audiobox_items.json"
        res = _json(path)
        for it in items:
            if it.item_id not in res:
                ws = C.aesthetic_windows(it)
                res[it.item_id] = {w["name"]: s for w, s in zip(ws, _ab_score(c, [C.read_window(it, w, C.SR_AUDIOBOX) for w in ws]))}
        C.write_json(path, res)
    t["audiobox_s"] = time.time() - t0
    t0 = time.time()
    stage_craft(out)
    t["craft_s"] = time.time() - t0
    t0 = time.time()
    stage_score2(out, models_dir=models_dir)
    stage_score3(out, models_dir=models_dir)
    t["score_s"] = time.time() - t0
    C.write_json(out / "api_timings.json", t)


# ================================================================== report
def _f(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"[{_f(v[0], nd)}, {_f(v[1], nd)}]"
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def stage_report(out: Path):
    sc = json.loads((out / "scores.json").read_text())
    val = json.loads((out / "validation.json").read_text()) if (out / "validation.json").exists() else {}
    cv = json.loads((out / "seam_cv.json").read_text()) if (out / "seam_cv.json").exists() else {}
    plan = json.loads((out / "synth_plan.json").read_text())
    rec = json.loads((out / "reconstruction.json").read_text())
    tim = json.loads((out / "timings.json").read_text()) if (out / "timings.json").exists() else {}
    L = []
    L.append("# Critico automatico di transizioni DJ — run dj_critic_v1\n")
    L.append(f"Codice: `src/musicmix/dj_critic.py`, `tools/run_dj_critic.py`, test `tests/test_dj_critic.py`. "
             f"Nota di ricerca: `conoscenza_mix/_meta/AUTO_CRITICO_2026-10-21.md`. Combinazione e insiemi di validazione "
             f"dichiarati in `PREREGISTRATION.json` prima di calcolare i punteggi.\n")
    L.append("## 1. Critici e modelli\n")
    L.append("| Critico | Cosa misura | Modello ufficiale (revisione) | Dominio di addestramento |\n|---|---|---|---|")
    L.append(f"| (a) estetico | delta giunta − lati (10 s) e giunta − sorgenti, Audiobox CE/CU/PQ e SongEval 5 assi, percentile vs giunte naturali | "
             f"`{C.PROVENANCE['audiobox']['repo']}` @{C.PROVENANCE['audiobox']['revision'][:10]}; SongEval @{C.PROVENANCE['songeval']['revision'][:10]} su MuQ @{C.PROVENANCE['muq']['revision'][:10]} | "
             "clip 10–30 s (Audiobox), canzoni intere in gran parte generate (SongEval) |")
    L.append(f"| (b) sorpresa | picco di NLL (somma 4 codebook EnCodec) alla giunta, z robusto vs tratti A-solo/B-solo, percentile vs giunte naturali | "
             f"`{C.PROVENANCE['musicgen']['repo']}` @{C.PROVENANCE['musicgen']['revision'][:10]}, non condizionato | 20k h di musica licenziata (crop 30 s) |")
    L.append(f"| (c) giunta | P(giunta plausibile) da logistica su embedding MuQ (layer 6 e 12) prima/dopo/attraverso | MuQ-large-msd-iter + logistica nostra | {cv.get('n','?')} giunte sintetiche da {cv.get('n_tracks','?')} brani del pool |")
    L.append("| (d) tecnico | verifiche dj_qa (ritmo, doppio basso, loudness, true peak, voci, parola) | Beat This! small0 (dentro dj_qa) | regole [H]/[S] |\n")
    if tim:
        L.append("Tempi per fase (CPU condivisa con altri agenti, load medio 8–13 su 4 core): " +
                 ", ".join(f"{k} {v/60:.0f} min" for k, v in tim.items()) + ".\n")
    L.append(f"Ricostruzioni storiche: {sum(r['status']=='bit-identico' for r in rec['results'])}/{len(rec['results'])} clip V8/V10 "
             "rigenerate con `anchor_handoff_v1` e **bit-identiche** allo sha256 registrato (`reconstruction.json`).\n")
    # benchmark sintetico
    L.append("## 2. Benchmark sintetico (sensibilità ai difetti noti, non gusto)\n")
    L.append(f"Esempi: {sum(plan['counts'].values())} pianificati ({json.dumps(plan['counts'])}); sottoinsieme lungo (24 s) per i critici a/b: {json.dumps(plan['long_counts'])}.\n")
    if cv:
        L.append(f"(c) CV per brano (5 fold, brani disgiunti), n={cv['n']} ({cv['n_pos']} plausibili):\n")
        L.append("| feature | AUC | AUC dentro-brano | AUC tra brani | Brier |\n|---|---|---|---|---|")
        for k, v in cv["cv"].items():
            L.append(f"| {k}{' (scelto)' if k == cv['chosen'] else ''} | {_f(v['auc'])} | {_f(v['auc_within'])} | {_f(v['auc_cross'])} | {_f(v['brier'])} |")
        ch = cv["cv"][cv["chosen"]]
        L.append("\nAUC per tipo di difetto (contro le giunte plausibili dello stesso dominio): " +
                 ", ".join(f"{k} {_f(v['auc_vs_positives_same_domain'])} (n={v['n_neg']})" for k, v in ch["per_negative_kind"].items()) + ".\n")
        L.append("P media sulle classi plausibili: " + ", ".join(f"{k} {_f(v)}" for k, v in ch["mean_p_positive_kinds"].items()) + ".\n")
    bench = sc.get("synthetic_benchmark", {})
    if bench:
        L.append("Critici a/b (zero-shot, nessun addestramento) e (c) out-of-fold sul sottoinsieme lungo; AUC naturale vs difetto:\n")
        kinds = sorted({k for b in bench.values() for k in b["auc_vs_natural_per_kind"]})
        L.append("| critico | n | AUC tutti | " + " | ".join(kinds) + " |\n|---|---|---|" + "---|" * len(kinds))
        for name, b in bench.items():
            L.append(f"| {name} | {b['n']} | {_f(b['auc_all'])} | " + " | ".join(_f(b['auc_vs_natural_per_kind'].get(k)) for k in kinds) + " |")
        L.append("")
    # validazione umana
    L.append("## 3. Validazione sui giudizi umani (n minuscolo)\n")
    for note in val.get("notes", []):
        L.append(f"- {note}")
    L.append("")
    for name, sv in val.get("sets", {}).items():
        L.append(f"### {name} — {sv['n_items']} item\n")
        L.append("| critico | n item | coppie | accuratezza a coppie | IC95 Wilson (coppie) | IC95 bootstrap (item) | Spearman ρ | p perm. |\n|---|---|---|---|---|---|---|---|")
        for key, e in sv["critics"].items():
            sp = e.get("spearman") or {}
            L.append(f"| {key} | {e['n_items']} | {e['n_pairs']} | {_f(e['pairwise_accuracy'])} | {_f(e['wilson95_pairs'])} | {_f(e['bootstrap95_items'])} | {_f(sp.get('rho'))} | {_f(sp.get('p_perm_two_sided'), 3)} |")
        L.append("")
    if val.get("v8_pairs"):
        L.append("### V8: confronti A/B ricostruiti\n")
        for pid, row in val["v8_pairs"].items():
            agree = {k: v["agrees"] for k, v in row.items() if isinstance(v, dict)}
            L.append(f"- {pid}: preferenza `{row['preference']}`; accordo critici: " + ", ".join(f"{k}={v}" for k, v in agree.items() if v is not None))
        L.append("")
    if val.get("v12_natural_vs_main"):
        L.append("### Controllo: continuazione naturale V12 contro la transizione main (stessa coppia)\n")
        for g, d in val["v12_natural_vs_main"].items():
            L.append(f"- {g}: " + ", ".join(f"{k} nat {_f(v['natural'])} / main {_f(v['main'])}" for k, v in d.items() if k in ("combined", "aesthetic", "surprisal", "seam", "technical")))
        L.append("")
    # classifiche
    L.append("## 4. Prova d'uso: classifica delle transizioni (galleria v2, set v1b)\n")
    for ds in ("dj_gallery_v2", "dj_gallery_v2_rejected", "dj_set_v1b"):
        rows = [(k, v) for k, v in sc["items"].items() if v["item"]["dataset"] == ds]
        if not rows:
            continue
        rows.sort(key=lambda kv: -(kv[1]["combined"].get("score") or -1))
        L.append(f"### {ds} ({len(rows)} transizioni)\n")
        L.append("| # | transizione | tipo | combinato ± sd | disaccordo | estetico | sorpresa | giunta (punto) | tecnico (QA) |\n|---|---|---|---|---|---|---|---|---|")
        for n, (k, v) in enumerate(rows, 1):
            cmb = v["combined"]
            seam = v.get("seam") or {}
            tq = (v["technical"].get("features") or {}).get("verdict")
            L.append(f"| {n} | `{k.split('/',1)[1]}` | {v['item']['kind']} | {_f(cmb.get('score'))} ± {_f(cmb.get('sd'))} | {_f(cmb.get('disagreement_sd'))} | "
                     f"{_f(v['aesthetic'].get('score'))} | {_f((v.get('surprisal') or {}).get('score'))} | {_f(seam.get('score'))} ({seam.get('weakest_point','–')}) | {_f(v['technical'].get('score'))} ({tq}) |")
        L.append("")
    # ---------------------------------------------------------------- v2
    v2m = _json(out / "models/v2_models.json")
    if v2m:
        L.append("## 5. Iterazione v2: embedding, ensemble calibrato, bootstrap\n")
        L.append("Confronto degli embedding per (c) sugli stessi esempi sintetici (CV per brano, AUC giunta plausibile vs difetto):\n")
        L.append("| embedding:feature | AUC (train e test sui comuni) | AUC test comuni, train su tutti i propri | AUC sul proprio insieme | n comuni | n propri |\n|---|---|---|---|---|---|")
        for k, v in sorted(v2m["embedding_comparison"].items(), key=lambda kv: -(kv[1].get("auc_common_trainall") or 0)):
            L.append(f"| {k}{' (scelto)' if k == v2m['chosen'] else ''}{' (copertura < 80%)' if v.get('selectable') is False else ''} | {_f(v['auc_common'])} | {_f(v.get('auc_common_trainall'))} | {_f(v.get('auc_own'))} | {v['n_common']} | {v.get('n_own')} |")
        L.append(f"\nScelto `{v2m['chosen']}`; sul suo insieme completo (n={v2m['n_seam_train']}) AUC CV {_f(v2m['seam_cv_auc'])}; per difetto: "
                 + ", ".join(f"{k} {_f(v)}" for k, v in v2m["seam_cv_per_kind"].items()) + ".\n")
        st = v2m.get("stack", {})
        if st.get("auc_oof_stacked") is not None:
            L.append(f"Stacking (logistica sui critici, addestrata su {st['n']} giunte sintetiche lunghe, CV per brano): AUC out-of-fold **{_f(st['auc_oof_stacked'])}**, "
                     f"media a pesi uguali {_f(st.get('mean_equal_weight_auc'))}, critici singoli: "
                     + ", ".join(f"{k} {_f(v)}" for k, v in st["auc_single"].items()) + f". Brier OOF {_f(st.get('brier_oof'), 3)}.\n")
            L.append("Coefficienti standardizzati: " + ", ".join(f"{k} {_f(v)}" for k, v in st["coef_standardized"].items()) + ".\n")
            L.append("Affidabilità (OOF, 5 bin): " + "; ".join(f"pred {_f(b['mean_pred'])} → freq {_f(b['frac_pos'])} (n={b['n']})" for b in st["reliability"] if b["n"]) + ".\n")
    dose = _json(out / "dose.json")
    if dose:
        L.append("## 6. Sensibilità ai difetti (dose-risposta su giunte reali buone)\n")
        L.append("Per ogni base (continuazione naturale su inizio frase) si applica un difetto di intensità crescente. Atteso: bontà decrescente (ρ di Spearman < 0). "
                 "Colonne: ρ medio sulle basi, quota di basi con ρ < 0, quota di passi non crescenti, curva media (livello: bontà).\n")
        for fam in ("phase", "gain", "key"):
            L.append(f"### {fam} (livelli {dose['levels'][fam]})\n")
            L.append("| critico | basi | ρ medio | basi con ρ<0 | passi non crescenti | curva media |\n|---|---|---|---|---|---|")
            for cname, v in dose["summary"].get(fam, {}).items():
                curve = " ".join(f"{k}:{_f(x)}" for k, x in v["mean_curve"].items())
                L.append(f"| {cname} | {v['n_bases']} | {_f(v['mean_rho'])} | {_f(v['frac_bases_rho_neg'])} | {_f(v['frac_non_increasing_steps'])} | {curve} |")
            L.append("")
        w = dose["summary"].get("word_cut_binary", {})
        if w:
            L.append("Parola tagliata (salto pulito vs stesso salto con parola tagliata, stesse basi): " +
                     ", ".join(f"{k}: peggiore in {_f(v['frac_word_cut_worse'])} di {v['n_pairs']}" for k, v in w.items()) + ".\n")
    v2v = _json(out / "validation_v2.json")
    if v2v:
        L.append("## 7. Validazione v2 sui giudizi umani\n")
        for name, sv in v2v.get("sets", {}).items():
            L.append(f"### {name} (n={len(sv['items'])})\n")
            L.append("| critico | coppie | accuratezza | IC95 bootstrap (item) | Spearman ρ | p perm. |\n|---|---|---|---|---|---|")
            for key, e in sv["critics"].items():
                sp = e.get("spearman") or {}
                L.append(f"| {key} | {e['n_pairs']} | {_f(e['pairwise_accuracy'])} | {_f(e['bootstrap95_items'])} | {_f(sp.get('rho'))} | {_f(sp.get('p_perm_two_sided'), 3)} |")
            L.append("")
        lg = v2v.get("logo", {})
        if lg:
            L.append(f"LOGO per coppia di brani ({lg.get('groups')} gruppi): " + "; ".join(
                f"{nm}: " + ", ".join(f"{sn} {_f(v['pairwise_accuracy'])} ({v['n_pairs']} coppie)" for sn, v in d.items())
                for nm, d in lg.items() if isinstance(d, dict) and nm.startswith("logo_")) + ".\n")
            from collections import Counter
            L.append("Critico scelto nei fold LOGO: " + ", ".join(f"{k} ×{v}" for k, v in Counter(x["chosen"] for x in lg.get("selection_log", [])).items()) + ".\n")
    sc2 = _json(out / "scores_v2.json")
    if sc2:
        L.append("## 8. Classifica v2 (score = P(giunta plausibile) dello stacker, ± sd bootstrap)\n")
        for ds in ("dj_gallery_v1", "dj_gallery_v2", "dj_gallery_v2_rejected", "dj_set_v1b", "v12_release", "v12_natural_reference"):
            rows = [(k, v) for k, v in sc2["items"].items() if v["item"]["dataset"] == ds]
            if not rows:
                continue
            rows.sort(key=lambda kv: -(kv[1]["score"] if kv[1]["score"] is not None else -1))
            L.append(f"### {ds}\n")
            L.append("| # | transizione | tipo | v2 ± sd | giunta P | Audiobox pct | SongEval pct | sorpresa z | tecnico | voto umano |\n|---|---|---|---|---|---|---|---|---|---|")
            for n, (k, v) in enumerate(rows, 1):
                pa = v["parts"]
                lab = v["item"].get("label") or {}
                L.append(f"| {n} | `{k.split('/',1)[1][:60]}` | {v['item']['kind']} | {_f(v['score'])} ± {_f(v['uncertainty'])} | {_f((pa.get('seam') or {}).get('p'))} | "
                         f"{_f((pa.get('audiobox') or {}).get('percentile_vs_natural'))} | {_f((pa.get('songeval') or {}).get('percentile_vs_natural'))} | "
                         f"{_f((pa.get('surprisal') or {}).get('peak_z'))} | {_f((pa.get('technical') or {}).get('score'))} | {lab.get('score', lab.get('ordinal', ''))} |")
            L.append("")
    # ---------------------------------------------------------------- v3
    v3m = _json(out / "models/v3_models.json")
    if v3m:
        L.append("## 10. Iterazione v3: mestiere DSP + giunta, ensemble scelto solo su dati sintetici\n")
        L.append("Nuovo critico (e) `craft_features` (nessun addestramento): |salto di loudness| BS.1770 3 s, |sfasamento| della griglia dei battiti di B "
                 "rispetto a quella di A estrapolate alla giunta (Beat This! small0 sulla clip, rette nei lati A-solo/B-solo; ripiego DSP), distanza dei profili tonali Krumhansl-Kessler prima/dopo, "
                 "probabilità di voce Silero negli ultimi 160 ms di A. Bontà = 1 − percentile tra le giunte naturali sintetiche. "
                 "La loro AUC sintetica è alta **per costruzione** (misurano i difetti simulati): non è evidenza di gusto.\n")
        L.append("Regola di inclusione (decisa prima dell'addestramento dello stacker, solo dose-risposta): un critico entra se ha ρ medio ≤ "
                 f"{V3_DOSE_INFO_RHO} in almeno una famiglia (o peggiora con la parola tagliata in ≥ 75% delle coppie) e nessuna famiglia con ρ medio ≥ +{V3_DOSE_ANTI_RHO} "
                 f"**e** bontà media che sale di almeno {V3_DOSE_MIN_RISE} dal livello 0 al massimo (v3b). La seconda condizione è stata aggiunta **dopo** aver visto la "
                 "validazione umana della versione v3a (regola solo su ρ, che escludeva la distanza tonale per curve piatte con ρ positivo da pareggi): "
                 "la validazione umana di v3b non è quindi indipendente; i numeri di v3a sono in `validation_v3a_prespecified.json`.\n")
        L.append("| critico | ρ fase | ρ volume | ρ tonalità | parola peggiore | esito |\n|---|---|---|---|---|---|")
        for k, v in v3m["dose_verdicts"].items():
            w = v.get("word") or {}
            esito = "tenuto" if v["keep"] else ("escluso: anti-correlato in " + ", ".join(v["anti_in"]) if v["anti_in"] else "escluso: non informativo")
            L.append(f"| {k} | {_f(v['rho'].get('phase'))} | {_f(v['rho'].get('gain'))} | {_f(v['rho'].get('key'))} | "
                     f"{_f(w.get('frac_word_cut_worse'))} ({w.get('n_pairs', 0)}) | {esito} |")
        L.append("")
        L.append("Configurazioni (stessi esempi, CV per brano, logistica con pesi ≥ 0): " +
                 "; ".join(f"{k} AUC OOF {_f(v['auc_oof'])} (pesi uguali {_f(v['auc_equal_weight'])}, n={v['n']})" for k, v in v3m["config_comparison"].items()) + ".\n")
        lc = v3m.get("long_subset_comparison") or {}
        if lc:
            L.append(f"Sottoinsieme lungo (n={lc['n']}): aggiunta di {', '.join(lc['extra'])}: AUC {_f(lc['auc_without'])} → {_f(lc['auc_with'])} "
                     "(entra solo se migliora di ≥ 0,02).\n")
        L.append(f"**Ensemble finale**: feature {', '.join(v3m['features'])}; n={v3m['n']}; AUC OOF **{_f(v3m['auc_oof'])}**, Brier {_f(v3m['brier_oof'], 3)}; "
                 "coefficienti standardizzati " + ", ".join(f"{k} {_f(x)}" for k, x in v3m["coef_standardized"].items()) +
                 f". Modello rapido solo DSP: AUC OOF {_f(v3m['fast']['auc_oof'])}.\n")
        L.append("AUC OOF per difetto: " + ", ".join(f"{k} {_f(v['auc'])} (n={v['n_neg']})" for k, v in v3m["per_kind"].items()) + ".\n")
        L.append("AUC dei singoli critici (bontà grezza, tutti i difetti): " + ", ".join(f"{k} {_f(v['auc'])}" for k, v in v3m["single_auc"].items()) + ".\n")
        L.append("Affidabilità (OOF, 5 bin): " + "; ".join(f"pred {_f(b['mean_pred'])} → freq {_f(b['frac_pos'])} (n={b['n']})" for b in v3m["reliability"] if b["n"]) + ".\n")
    apit = _json(out / "api_timings_v3.json")
    if apit:
        L.append("Tempi dell'API `dj_critic.score_transition(s)` (CPU condivisa): " + "; ".join(
            f"modo {m}: lotto di {d['n']} clip {d['batch_s']:.0f} s, chiamata singola {d['single_s']:.0f} s" for m, d in apit.items()) + ".\n")
    v3v = _json(out / "validation_v3.json")
    if v3v:
        L.append("## 11. Validazione v3 sui giudizi umani\n")
        for note in v3v.get("notes", []):
            L.append(f"- {note}")
        L.append("")
        L.append("| critico | S1 galleria v1 (6) | S2 V10 verbale (8) | S3 galleria ≥7 vs V12 (≤7) | aggregato | IC95 bootstrap aggregato |\n|---|---|---|---|---|---|")
        for key, pool in v3v["pooled"].items():
            cells = []
            for sn in ("S1_gallery_v1_scores", "S2_v10_verbal_ordinal", "S3_gallery_ge7_vs_v12_main"):
                e = v3v["sets"][sn]["critics"][key]
                cells.append(f"{_f(e['pairwise_accuracy'])} [{_f((e.get('bootstrap95_items') or [None, None])[0])}, {_f((e.get('bootstrap95_items') or [None, None])[1])}] ({e['n_pairs']})")
            L.append(f"| {key} | " + " | ".join(cells) + f" | {_f(pool['pairwise_accuracy'])} ({pool['n_pairs']}) | {_f(pool.get('bootstrap95_items'))} |")
        L.append("")
        v3a = _json(out / "validation_v3a_prespecified.json")
        if v3a:
            L.append("Versione v3a (regola pre-specificata, prima di vedere questi giudizi): " + ", ".join(
                f"{k} {_f(v3a['pooled'][k]['pairwise_accuracy'])} {_f(v3a['pooled'][k].get('bootstrap95_items'))}" for k in ("v3_ensemble", "v3_fast_dsp", "n_flags_neg") if k in v3a.get("pooled", {})) + ".\n")
        if v3v.get("v8_pairs"):
            L.append("V8 (A/B): " + "; ".join(f"{pid} pref {d['preference']}: v3 A {_f(d['v3_ensemble']['A'])} / B {_f(d['v3_ensemble']['B'])}" for pid, d in v3v["v8_pairs"].items()) + ".\n")
        if v3v.get("v12_natural_vs_main"):
            L.append("Controllo V12 (continuazione naturale vs transizione main, stessa coppia): " + "; ".join(
                f"{g[:12]}: nat {_f(d['v3_ensemble']['natural'])} / main {_f(d['v3_ensemble']['main'])}" for g, d in v3v["v12_natural_vs_main"].items()) + ".\n")
        lg = v3v.get("logo_select_single", {})
        if lg:
            L.append(f"LOGO per coppia di brani ({lg['groups']} gruppi; scelta del critico singolo sui gruppi restanti, diagnostica): " +
                     ", ".join(f"{sn} {_f(v['pairwise_accuracy'])} {_f(v.get('bootstrap95_items'))} ({v['n_pairs']} coppie)" for sn, v in lg["per_set"].items()) +
                     "; scelti " + ", ".join(f"{k} ×{v}" for k, v in lg["chosen"].items()) + ".\n")
    sc3 = _json(out / "scores_v3.json")
    if sc3:
        L.append("## 12. Classifica v3 (score = P(giunta plausibile) dell'ensemble ± sd bootstrap per brano; flag = veto tecnico)\n")
        for ds in ("dj_gallery_v1", "dj_gallery_v2", "dj_gallery_v2_rejected", "dj_set_v1b", "v12_release", "v12_natural_reference", "v10_reconstructed"):
            rows = [(k, v) for k, v in sc3["items"].items() if v["item"]["dataset"] == ds]
            if not rows:
                continue
            rows.sort(key=lambda kv: -(kv[1]["score"] if kv[1]["score"] is not None and np.isfinite(kv[1]["score"]) else -1))
            L.append(f"### {ds}\n")
            L.append("| # | transizione | v3 ± sd | rapido DSP | loud dB | fase ms | dist. tonale | voce al taglio | flag | voto umano |\n|---|---|---|---|---|---|---|---|---|---|")
            for n, (k, v) in enumerate(rows, 1):
                cf = v.get("craft") or {}
                lab = v["item"].get("label") or {}
                L.append(f"| {n} | `{k.split('/', 1)[1][:58]}` | {_f(v['score'])} ± {_f(v['uncertainty'])} | {_f(v['score_fast'])} | {_f(cf.get('loud_jump_db'), 1)} | "
                         f"{_f(cf.get('beat_phase_ms'), 0)} | {_f(cf.get('chroma_dist'))} | {_f(cf.get('vad_at_cut'))} | {', '.join(f['flag'] for f in v['flags']) or '–'} | "
                         f"{lab.get('score', lab.get('ordinal', ''))} |")
            L.append("")
    L.append("## 9. Come leggere i numeri\n")
    L.append("- 0,5 = come una giunta naturale tipica (percentile) per estetico e sorpresa; per (c) è una probabilità dal classificatore sintetico.")
    L.append("- Il combinato è una media dichiarata a priori, **non** stimata dai giudizi; la colonna disaccordo è la deviazione standard tra critici.")
    L.append("- Nessun numero qui certifica qualità musicale; la validazione umana ha n da 2 a 8 item per insieme.\n")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("report scritto")


# ==================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", help="fasi: " + ", ".join(STAGES) + " oppure all")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/dj_critic_v1")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="solo per prove: limita gli esempi sintetici")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "runs/dj_critic_v1", help="fasi api: run con modelli v2 e distribuzioni nulle")
    ap.add_argument("--model", default="mert", choices=["mert", "clap"], help="fase emb: quale embedding")
    ap.add_argument("--part", default="all", choices=["all", "items", "synth", "dose", "items_dose"], help="muq/musicgen: sottoinsieme da calcolare")
    args = ap.parse_args(argv)
    if args.stages[0] == "api":
        # chiamata dall'API Python (dj_critic.score_transitions): item già scritti in --out
        if "venv-np1" in sys.executable:
            stage_api_np1(args.out, args.threads, args.models_dir)
        else:
            subprocess.run([PY_NP1, __file__, "api", "--out", str(args.out), "--threads", str(args.threads),
                            "--models-dir", str(args.models_dir)], check=True)
            stage_api_main(args.out, args.threads, args.models_dir)
        return 0
    stages = STAGES if args.stages == ["all"] else args.stages
    args.out.mkdir(parents=True, exist_ok=True)
    np1 = "venv-np1" in sys.executable
    for st in stages:
        if st not in STAGES:
            ap.error(f"fase sconosciuta: {st}")
        need_np1 = st in NP1_STAGES
        if need_np1 != np1:
            py = PY_NP1 if need_np1 else PY_MAIN
            cmd = [py, __file__, st, "--out", str(args.out), "--threads", str(args.threads), "--part", args.part, "--model", args.model]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            log("→", " ".join(cmd))
            subprocess.run(cmd, check=True)
            continue
        log(f"fase {st}")
        fn = globals()[f"stage_{st}"]
        kwargs = {}
        if st in ("muq", "musicgen", "audiobox"):
            kwargs["threads"] = args.threads
        if st in ("muq", "musicgen") and args.limit:
            kwargs["limit"] = args.limit
        if st in ("muq", "musicgen"):
            kwargs["part"] = args.part
        if st == "emb":
            kwargs = {"threads": args.threads, "model": args.model, "limit": args.limit}
        t0 = time.time()
        fn(args.out, **kwargs)
        tp = args.out / "timings.json"
        tim = json.loads(tp.read_text()) if tp.exists() else {}
        tim[st] = tim.get(st, 0.0) + (time.time() - t0)
        C.write_json(tp, tim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
