#!/usr/bin/env python3
"""Build and serve the read-only V10 pretrained audio review gallery.

The gallery is assembled from the two existing six-track pilot directories.
The four all-beats controls are copied once because their bytes are identical
between pilots; the two CUE-DETR and two LinkSeg renders remain separate.  The
HTTP layer deliberately reuses :func:`musicmix.listening_server.make_handler`
so its existing path allowlist and byte-range implementation are retained.
No judgement form or new POST route is added to the gallery.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import shutil
import sys
import wave
from http.server import ThreadingHTTPServer
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
POOL_RUN = ROOT / "runs" / "pretrained_v10_pool6"
LINKSEG_RUN = ROOT / "runs" / "pretrained_v10_linkseg6"
REVIEW_ROOT = ROOT / "runs" / "pretrained_v10_review"
AUDIO_ROOT = REVIEW_ROOT / "audio"
MANIFEST_PATH = REVIEW_ROOT / "manifest.json"
INDEX_PATH = REVIEW_ROOT / "index.html"
QA_PATH = REVIEW_ROOT / "qa_http.json"
TEST_ID = "review"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wav_info(path: Path) -> tuple[int, int, float]:
    with wave.open(str(path), "rb") as stream:
        sample_rate = int(stream.getframerate())
        frames = int(stream.getnframes())
    return sample_rate, frames, frames / sample_rate if sample_rate else 0.0


def _source_artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _track_titles(manifest: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for track in manifest.get("tracks", []):
        title = str(track.get("title", track.get("slug", "")))
        artist = str(track.get("artist", "")).strip()
        label = f"{artist} — {title}" if artist else title
        track_id = str(track.get("track_id", ""))
        if track_id:
            result[track_id] = label
    return result


def _proposals(run_root: Path, method_prefix: str) -> list[dict[str, Any]]:
    items = []
    for path in sorted(run_root.glob(f"proposal_{method_prefix}_*.json")):
        document = _load_json(path)
        candidate = dict(document.get("candidate", {}))
        candidate["_proposal_path"] = path
        candidate["_proposal"] = document
        items.append(candidate)
    return items


def _find_audio(run_root: Path, stem: str) -> Path:
    matches = sorted((run_root / "audio").glob(stem + "_*.wav"))
    if len(matches) != 1:
        raise ValueError(f"Expected one audio for {stem!r}, found {len(matches)}")
    return matches[0]


def _render_item(
    *,
    item_number: int,
    system: str,
    category: str,
    method_label: str,
    run_root: Path,
    run_manifest_path: Path,
    proposal_path: Path,
    proposal: dict[str, Any],
    titles: dict[str, str],
    audio_stem: str,
    technical_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    source_audio = _find_audio(run_root, audio_stem).resolve(strict=True)
    source_sha = _sha256(source_audio)
    sample_rate, duration_samples, duration_seconds = _wav_info(source_audio)
    candidate = proposal.get("candidate", proposal)
    out_sample = int(candidate["cue_out_sample"])
    in_sample = int(candidate["cue_in_sample"])
    source_a = str(candidate.get("track_a", ""))
    source_b = str(candidate.get("track_b", ""))
    pair = {
        "outgoing": titles.get(source_a, source_a),
        "incoming": titles.get(source_b, source_b),
        "outgoing_track_id": source_a,
        "incoming_track_id": source_b,
        "cue_out_sample": out_sample,
        "cue_in_sample": in_sample,
        "cue_out_seconds": out_sample / sample_rate,
        "cue_in_seconds": in_sample / sample_rate,
    }
    review_name = f"trial_{item_number:03d}_A.wav"
    review_audio = AUDIO_ROOT / review_name
    shutil.copyfile(source_audio, review_audio)
    review_sha = _sha256(review_audio)
    if review_sha != source_sha:
        raise RuntimeError(f"Copy hash mismatch for {source_audio}")
    technical = (technical_artifact or {}).get("technical_quality", {})
    technical_metrics = technical.get("metrics", {})
    technical_flags = list(technical.get("flags", []))
    technical_qa = {
        "status": technical.get("status", "unavailable"),
        "flags": technical_flags,
        "flag_count": len(technical_flags),
        "rms_change_db": technical_metrics.get("rms", {}).get("change_db"),
        "seam_silence_pre_seconds": technical_metrics.get("seam_silence", {}).get("pre_edge_seconds"),
        "seam_silence_post_seconds": technical_metrics.get("seam_silence", {}).get("post_edge_seconds"),
        "asr_stress_50ms": "uncertain; does not certify word integrity",
    }
    source_artifacts = [
        _source_artifact(run_manifest_path),
        _source_artifact(proposal_path),
        _source_artifact(source_audio),
    ]
    if technical_artifact:
        artifact_path = Path(str(technical_artifact.get("path", "")))
        if artifact_path.is_file():
            source_artifacts.append(_source_artifact(artifact_path))
    return {
        "audio_id": f"audio_{item_number:03d}",
        "trial": item_number,
        "system": system,
        "category": category,
        "method_label": method_label,
        "audio_filename": review_name,
        "audio_path": str(review_audio.resolve()),
        "source_audio_path": str(source_audio),
        "source_audio_sha256": source_sha,
        "review_audio_sha256": review_sha,
        "sample_rate": sample_rate,
        "duration_samples": duration_samples,
        "duration_seconds": duration_seconds,
        "candidate_id": str(candidate.get("candidate_id", "")),
        "score": proposal.get("score"),
        "pair": pair,
        "technical_qa": technical_qa,
        "source_artifacts": source_artifacts,
        "seek_step_seconds": 5,
    }


def _build_items() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool_manifest_path = POOL_RUN / "manifest.json"
    link_manifest_path = LINKSEG_RUN / "manifest.json"
    pool_manifest = _load_json(pool_manifest_path)
    link_manifest = _load_json(link_manifest_path)
    pool_titles = _track_titles(pool_manifest)
    link_titles = _track_titles(link_manifest)
    pool_artifacts = {str(item.get("candidate_id")): item for item in pool_manifest.get("artifacts", [])}
    link_artifacts = {str(item.get("candidate_id")): item for item in link_manifest.get("artifacts", [])}
    # The manifests contain the same canonical track identity map.  Prefer the
    # pool manifest for controls, while retaining each model pilot's manifest.
    if pool_titles != link_titles:
        raise ValueError("Pilot manifests disagree on canonical track titles")

    specs = [
        ("CUE-DETR", "model", "CUE-DETR direct", POOL_RUN,
         pool_manifest_path, "cue_detr_direct_01", "proposal_cue_detr_direct_01.json"),
        ("CUE-DETR", "model", "CUE-DETR direct", POOL_RUN,
         pool_manifest_path, "cue_detr_direct_02", "proposal_cue_detr_direct_02.json"),
        ("LinkSeg", "model", "LinkSeg direct", LINKSEG_RUN,
         link_manifest_path, "linkseg_direct_01", "proposal_linkseg_direct_01.json"),
        ("LinkSeg", "model", "LinkSeg direct", LINKSEG_RUN,
         link_manifest_path, "linkseg_direct_02", "proposal_linkseg_direct_02.json"),
    ]
    for index in range(1, 5):
        specs.append((
            "All-beats control", "control", "All-beats control", POOL_RUN,
            pool_manifest_path, f"all_beats_control_{index:02d}",
            f"proposal_all_beats_control_{index:02d}.json",
        ))

    items: list[dict[str, Any]] = []
    seen_audio: dict[str, str] = {}
    for number, (system, category, label, run_root, manifest_path, stem, proposal_name) in enumerate(specs, 1):
        proposal_path = run_root / proposal_name
        proposal = _load_json(proposal_path)
        candidate_id = str(proposal.get("candidate", {}).get("candidate_id", ""))
        artifact_map = link_artifacts if run_root == LINKSEG_RUN else pool_artifacts
        item = _render_item(
            item_number=number, system=system, category=category,
            method_label=label, run_root=run_root,
            run_manifest_path=manifest_path, proposal_path=proposal_path,
            proposal=proposal, titles=pool_titles, audio_stem=stem,
            technical_artifact=artifact_map.get(candidate_id),
        )
        previous = seen_audio.get(item["source_audio_sha256"])
        if previous is not None:
            raise ValueError(
                f"Audio deduplication failed: {stem} repeats {previous}; "
                "the gallery expects eight distinct SHA-256 values"
            )
        seen_audio[item["source_audio_sha256"]] = item["audio_id"]
        items.append(item)

    if len(items) != 8 or len(seen_audio) != 8:
        raise ValueError(f"Expected 8 distinct audio items, got {len(items)} / {len(seen_audio)}")
    # Confirm the four controls are byte-identical in the second pilot.  They
    # are represented once in the gallery, but this source lineage is kept in
    # the manifest so the deduplication is reviewable.
    for index in range(1, 5):
        pool_audio = _find_audio(POOL_RUN, f"all_beats_control_{index:02d}")
        link_audio = _find_audio(LINKSEG_RUN, f"all_beats_control_{index:02d}")
        pool_sha = _sha256(pool_audio)
        link_sha = _sha256(link_audio)
        if pool_sha != link_sha:
            raise ValueError(f"Control {index} differs between pilot runs")
        item = items[3 + index]
        item["deduplicated_source_audio"] = {
            "path": str(link_audio.resolve()),
            "sha256": link_sha,
            "bytes": link_audio.stat().st_size,
            "pilot": "pretrained_v10_linkseg6",
        }
    return items, [
        _source_artifact(pool_manifest_path),
        _source_artifact(link_manifest_path),
    ]


def _html(items: list[dict[str, Any]]) -> str:
    cards = []
    for item in items:
        pair = item["pair"]
        score = "" if item.get("score") is None else f"{float(item['score']):.6f}"
        qa = item.get("technical_qa", {})
        flags = qa.get("flags", [])
        if flags:
            flag_lines = " ".join(
                f"{escape(str(flag.get('severity', 'observation')))}: {escape(str(flag.get('description', flag.get('code', 'flag'))))}"
                for flag in flags
            )
        else:
            flag_lines = "none reported"
        rms_change = qa.get("rms_change_db")
        rms_text = "" if rms_change is None else f"RMS change {float(rms_change):+.3f} dB (descriptive observation) · "
        seam_pre = qa.get("seam_silence_pre_seconds")
        seam_post = qa.get("seam_silence_post_seconds")
        seam_text = ""
        if seam_pre is not None and seam_post is not None and max(float(seam_pre), float(seam_post)) >= 0.05:
            seam_text = f"seam silence {float(seam_pre):.3f}s/{float(seam_post):.3f}s (heuristic observation) · "
        cards.append(
            f'''<article class="card" data-system="{escape(item['system'])}" data-category="{escape(item['category'])}">
<div class="card-head"><span class="tag {escape(item['category'])}">{escape(item['system'])}</span><span class="id">{escape(item['audio_id'])}</span></div>
<h2>{escape(item['method_label'])}: {escape(pair['outgoing'])} → {escape(pair['incoming'])}</h2>
<p class="meta">Cue out {pair['cue_out_seconds']:.3f}s · cue in {pair['cue_in_seconds']:.3f}s · score {score or 'n/a'} · {item['duration_seconds']:.3f}s</p>
<audio controls preload="metadata" src="audio/{escape(item['audio_filename'])}"></audio>
<p><button type="button" onclick="seek(this,-5)">−5 s</button> <button type="button" onclick="seek(this,5)">+5 s</button></p>
<p class="qa"><strong>Technical observations:</strong> {rms_text}{seam_text}flags — {flag_lines}<br><span>ASR 50 ms stress: uncertain; word integrity is not certified.</span></p>
<details><summary>Provenance</summary><code>source SHA256: {escape(item['source_audio_sha256'])}<br>review SHA256: {escape(item['review_audio_sha256'])}<br>candidate: {escape(item['candidate_id'])}</code></details>
</article>'''
        )
    generated = datetime.now(timezone.utc).isoformat()
    return f'''<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MusicMix V10 — revisione pretrained</title>
<style>
:root {{ color-scheme: dark; }} body {{ font: 16px system-ui,sans-serif; max-width: 1100px; margin: 28px auto; padding: 0 18px; background:#11151a; color:#e7edf5; }}
h1 {{ margin-bottom:4px; }} .note {{ color:#abb8c8; }} .filters {{ position:sticky; top:0; z-index:2; display:flex; gap:12px; flex-wrap:wrap; padding:14px 0; background:#11151a; }}
select,button {{ font:inherit; padding:7px 11px; border:1px solid #536173; border-radius:6px; background:#202a35; color:#e7edf5; }}
.count {{ color:#9fb5ca; margin:8px 0 16px; }} .card {{ padding:18px 0 22px; border-bottom:1px solid #364453; }} .card[hidden] {{ display:none; }}
.card-head {{ display:flex; justify-content:space-between; color:#9fb5ca; }} .tag {{ border-radius:12px; padding:3px 9px; font-size:13px; background:#315c85; }} .tag.control {{ background:#5b4c2c; }}
h2 {{ font-size:19px; margin:10px 0 5px; }} .meta {{ color:#aebdcc; }} audio {{ width:100%; margin:6px 0; }} details {{ color:#9fb5ca; }} code {{ word-break:break-all; font-size:12px; }}
.qa {{ color:#c1cfdb; font-size:14px; }} .qa strong {{ color:#e7edf5; }} .qa span {{ color:#d3b878; }}
</style></head><body>
<h1>MusicMix V10 — revisione audio pretrained</h1>
<p class="note">Galleria in sola lettura. Sono mostrati una sola volta otto render con byte distinti: due CUE-DETR, due LinkSeg e quattro controlli all-beats. Non sono richiesti né raccolti voti.</p>
<div class="filters"><label>Sistema <select id="system" onchange="filterCards()"><option value="all">Tutti</option><option>CUE-DETR</option><option>LinkSeg</option><option>All-beats control</option></select></label>
<label>Tipo <select id="category" onchange="filterCards()"><option value="all">Tutti</option><option value="model">Modello</option><option value="control">Controllo</option></select></label></div>
<div id="count" class="count"></div><main>{''.join(cards)}</main>
<script>
function seek(button,delta) {{ const audio=button.closest('.card').querySelector('audio'); audio.currentTime=Math.max(0,Math.min(Number.isFinite(audio.duration)?audio.duration:Infinity,audio.currentTime+delta)); }}
function filterCards() {{ const system=document.getElementById('system').value; const category=document.getElementById('category').value; let visible=0; document.querySelectorAll('.card').forEach(card=>{{const ok=(system==='all'||card.dataset.system===system)&&(category==='all'||card.dataset.category===category); card.hidden=!ok; if(ok) visible++;}}); document.getElementById('count').textContent=visible+' of {len(items)} audio items visible'; }}
filterCards();
</script><p class="note">Generata {escape(generated)} · manifest.json contiene gli SHA-256 della sorgente e della copia review.</p></body></html>
'''


def build_gallery() -> dict[str, Any]:
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    items, source_artifacts = _build_items()
    for item in items:
        review_path = Path(item["audio_path"])
        if _sha256(review_path) != item["review_audio_sha256"]:
            raise ValueError(f"Review audio changed while building: {review_path}")
    INDEX_PATH.write_text(_html(items), encoding="utf-8")
    manifest = {
        "schema_version": "musicmix.pretrained-review-gallery.1",
        "status": "ready",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "test_id": TEST_ID,
        "review_root": str(REVIEW_ROOT.resolve()),
        "audio_count": len(items),
        "distinct_audio_sha256_count": len({item["review_audio_sha256"] for item in items}),
        "filters": {"system": ["CUE-DETR", "LinkSeg", "All-beats control"], "category": ["model", "control"]},
        "seek_step_seconds": 5,
        "source_artifacts": source_artifacts,
        "index": _source_artifact(INDEX_PATH),
        "audio": items,
        "limitations": [
            "Audio previews are diagnostic renders; no listening judgement is required.",
            "The gallery does not claim cross-song resolution or perceptual superiority.",
        ],
    }
    # Index hash is intentionally added after writing the index and before the
    # manifest is written, making the index and all copied WAVs auditable.
    _atomic_json(MANIFEST_PATH, manifest)
    return manifest


def _server(port: int) -> None:
    from musicmix.listening_server import make_handler

    tests = {
        TEST_ID: {
            "path": REVIEW_ROOT.resolve(),
            "metadata": {
                "test_id": TEST_ID,
                "run_id": "pretrained_v10_review",
                "title": "MusicMix V10 pretrained audio review",
                "feedback_filename": "unused.json",
            },
        }
    }
    base_handler = make_handler(tests, REVIEW_ROOT / "feedback")
    allowed_audio = {item["audio_filename"] for item in _load_json(MANIFEST_PATH)["audio"]}

    class ReviewHandler(base_handler):
        """Read-only allowlist wrapper retaining inherited ``serve_audio``."""

        def get_file(self, head_only: bool = False):
            parts = self.path.split("?", 1)[0].strip("/").split("/")
            if len(parts) == 3 and parts[:2] == ["test", TEST_ID] and parts[2] == "index.html":
                return self.reply(200, INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
            if (
                len(parts) == 4 and parts[:3] == ["test", TEST_ID, "audio"]
                and parts[3] in allowed_audio
            ):
                path = AUDIO_ROOT / parts[3]
                if path.is_file():
                    return self.serve_audio(path, head_only=head_only)
            return self.reply(404, b"{}")

        def do_POST(self):
            return self.reply(404, b'{"error":"read-only review"}')

    handler = ReviewHandler
    server = ThreadingHTTPServer(("127.0.0.1", int(port)), handler)
    print(f"http://127.0.0.1:{server.server_port}/test/{TEST_ID}/index.html", flush=True)
    server.serve_forever()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--serve", action="store_true", help="serve after building")
    parser.add_argument("--build-only", action="store_true", help="build assets and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.port < 1 or args.port > 65535:
        raise SystemExit("--port must be between 1 and 65535")
    manifest = build_gallery()
    print(json.dumps({"status": manifest["status"], "audio_count": manifest["audio_count"],
                      "distinct_audio_sha256_count": manifest["distinct_audio_sha256_count"],
                      "manifest": str(MANIFEST_PATH.resolve()),
                      "index": str(INDEX_PATH.resolve())}, ensure_ascii=False, indent=2))
    if args.build_only or not args.serve:
        return 0
    _server(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
