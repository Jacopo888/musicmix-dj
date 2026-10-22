#!/usr/bin/env python3
"""Serve the read-only V11 psychology transition review.

The review manifest is intentionally a small, human-readable document.  It
contains no voting endpoint.  Audio is exposed only through manifest item IDs
and is served with single-range HTTP support so an HTML audio element can seek
without giving the browser a filesystem path.

Example::

    python tools/serve_psychology_review.py --serve
    python tools/serve_psychology_review.py --manifest runs/psychology_v11/review_manifest.json --port 8772 --serve
"""

from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "runs" / "psychology_v11" / "review_manifest.json"
SCHEMA = "musicmix.psychology-review.1"


class ReviewManifestError(ValueError):
    """The review manifest is malformed or refers outside the project root."""


def _text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ReviewManifestError(f"{field} must be a non-empty string")
    return value


def _anchor(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReviewManifestError(f"{field} must be an object")
    track = value.get("track", value.get("display_track", value.get("title", "")))
    seconds = value.get("seconds", value.get("time_seconds"))
    if not isinstance(track, str) or not track.strip():
        raise ReviewManifestError(f"{field}.track must be a non-empty string")
    if seconds is not None:
        try:
            seconds = float(seconds)
        except (TypeError, ValueError) as exc:
            raise ReviewManifestError(f"{field}.seconds must be numeric") from exc
        if not seconds >= 0:
            raise ReviewManifestError(f"{field}.seconds must be non-negative")
    result = {"track": track}
    if seconds is not None:
        result["seconds"] = seconds
    return result


def _notes(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.append(_text(item, f"{field}[{index}]").strip())
        return result
    raise ReviewManifestError(f"{field} must be a string or list of strings")


def _safe_audio_path(audio_path: str, project_root: Path) -> Path:
    # A manifest path is deliberately project-relative.  Reject absolute paths
    # and parent components before resolving symlinks, then check the resolved
    # result once more so a symlink cannot escape the project tree.
    raw = Path(audio_path)
    if raw.is_absolute() or raw.drive or ".." in raw.parts:
        raise ReviewManifestError("audio_path must be project-relative")
    root = project_root.resolve(strict=True)
    resolved = (root / raw).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReviewManifestError("audio_path escapes the project root") from exc
    if not resolved.is_file():
        raise ReviewManifestError(f"audio_path does not name a file: {audio_path}")
    return resolved


def _normalise_item(raw: Mapping[str, Any], index: int, project_root: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ReviewManifestError(f"items[{index}] must be an object")
    item_id = _text(raw.get("id"), f"items[{index}].id")
    if "/" in item_id or "\\" in item_id or item_id in {".", ".."}:
        raise ReviewManifestError(f"items[{index}].id must be a URL-safe identifier")
    audio_path = _text(raw.get("audio_path"), f"items[{index}].audio_path")
    resolved = _safe_audio_path(audio_path, project_root)
    a_out = raw.get("A_out", raw.get("a_out"))
    b_in = raw.get("B_in", raw.get("b_in"))
    if a_out is None or b_in is None:
        raise ReviewManifestError(f"items[{index}] requires A_out and B_in anchors")
    context_notes = _notes(raw.get("context"), f"items[{index}].context")
    warning_notes = _notes(raw.get("warnings", raw.get("unknown")), f"items[{index}].warnings")
    # Keep the source-relative path in the document and a private resolved path
    # for the handler.  The latter is never rendered into the HTML.
    return {
        "id": item_id,
        "title": _text(raw.get("title", item_id), f"items[{index}].title"),
        "audio_path": audio_path,
        "_resolved_audio_path": resolved,
        "candidate_id": _text(raw.get("candidate_id", "unknown"), f"items[{index}].candidate_id"),
        "A_out": _anchor(a_out, f"items[{index}].A_out"),
        "B_in": _anchor(b_in, f"items[{index}].B_in"),
        "action": _text(raw.get("action", "unknown"), f"items[{index}].action", allow_empty=True),
        "intent": _text(raw.get("intent", "unknown"), f"items[{index}].intent", allow_empty=True),
        "notes": _notes(raw.get("notes"), f"items[{index}].notes"),
        "context": context_notes,
        "warnings": warning_notes,
    }


def load_manifest(path: str | Path = DEFAULT_MANIFEST, *, project_root: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a psychology review manifest.

    The default manifest is project-relative to the repository root.  For a
    temporary or exported manifest, omitting ``project_root`` makes paths
    relative to that manifest's directory, which keeps tests and offline
    review bundles self-contained.
    """

    manifest_path = Path(path).expanduser().resolve(strict=True)
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewManifestError(f"cannot read manifest: {manifest_path}") from exc
    if not isinstance(document, Mapping):
        raise ReviewManifestError("manifest must be a JSON object")
    if document.get("schema_version") != SCHEMA:
        raise ReviewManifestError(f"expected schema_version {SCHEMA!r}")
    raw_items = document.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ReviewManifestError("manifest items must be a non-empty list")
    if project_root is None:
        root = ROOT if manifest_path.is_relative_to(ROOT) else manifest_path.parent
    else:
        root = Path(project_root).expanduser().resolve(strict=True)
    items = [_normalise_item(raw, index, root) for index, raw in enumerate(raw_items)]
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ReviewManifestError("manifest item IDs must be unique")
    result = dict(document)
    result["items"] = items
    result["_manifest_path"] = manifest_path
    result["_project_root"] = root
    return result


def _display_anchor(anchor: Mapping[str, Any]) -> str:
    track = escape(str(anchor.get("track", "sconosciuta")))
    seconds = anchor.get("seconds")
    return f"{track} · {float(seconds):.3f}s" if seconds is not None else track


_ACTION_LABELS = {
    "stay": "mantieni", "prepare": "prepara", "execute": "esegui",
    "defer": "rimanda", "handoff": "passaggio", "overlap": "sovrapposizione",
}
_INTENT_LABELS = {
    "preserve_groove": "preserva il groove", "continue_motion": "continua il moto",
    "arrive": "arrivo", "deliberate_contrast": "contrasto intenzionale",
    "handoff": "passaggio sequenziale", "overlap": "sovrapposizione equal-power",
}


def _label(value: Any, labels: Mapping[str, str]) -> str:
    text = str(value or "unknown")
    return labels.get(text, text.replace("_", " "))


def _html(document: Mapping[str, Any]) -> str:
    page_title = escape(str(document.get('title', 'MusicMix — revisione delle transizioni')))
    cards: list[str] = []
    for item in document["items"]:
        warnings = list(item.get("context", [])) + list(item.get("warnings", []))
        notes = list(item.get("notes", []))
        note_html = " ".join(escape(note) for note in notes) or "nessuna nota"
        warning_html = " ".join(escape(note) for note in warnings) or "nessuna evidenza ignota o avvertenza"
        cards.append(
            f'''<article class="card" data-item="{escape(item['id'])}">
<h2>{escape(item['title'])}</h2>
<p class="meta">Candidate: <code>{escape(item['candidate_id'])}</code> · Azione: <strong>{escape(_label(item['action'], _ACTION_LABELS))}</strong> · Intento: <strong>{escape(_label(item['intent'], _INTENT_LABELS))}</strong></p>
<p class="anchors"><strong>A out:</strong> { _display_anchor(item['A_out']) } &nbsp; <strong>B in:</strong> { _display_anchor(item['B_in']) }</p>
<audio controls preload="metadata" src="/audio/{escape(item['id'])}"></audio>
<p><button type="button" onclick="seek(this,-5)">−5 s</button> <button type="button" onclick="seek(this,5)">+5 s</button></p>
<p class="notes"><strong>Note:</strong> {note_html}</p>
<p class="warnings"><strong>Da verificare:</strong> {warning_html}</p>
</article>'''
        )
    return f'''<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{page_title}</title>
<style>
:root {{ color-scheme: dark; }} body {{ max-width:1000px; margin:28px auto; padding:0 18px; background:#11151a; color:#e7edf5; font:16px system-ui,sans-serif; }}
h1 {{ margin-bottom:4px; }} .lead,.meta,.warnings {{ color:#b7c3d0; }} .card {{ border-bottom:1px solid #394553; padding:18px 0 24px; }}
h2 {{ margin:0 0 7px; font-size:20px; }} audio {{ width:100%; margin:8px 0; }} button {{ font:inherit; padding:7px 11px; border:1px solid #536173; border-radius:6px; background:#202a35; color:#e7edf5; }}
.warnings strong {{ color:#e8c77d; }} code {{ color:#acd0e9; }}
</style></head><body><h1>{page_title}</h1>
<p class="lead">Ascolto diagnostico in sola lettura. Le note descrivono il contesto; le evidenze ignote restano da verificare e non sono voti di qualità musicale.</p>
<main>{''.join(cards)}</main><script>
function seek(button, delta) {{ const audio=button.closest('.card').querySelector('audio'); const duration=Number.isFinite(audio.duration)?audio.duration:Infinity; audio.currentTime=Math.max(0,Math.min(duration,audio.currentTime+delta)); }}
</script></body></html>'''


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    public = {key: value for key, value in document.items() if not key.startswith("_")}
    public["items"] = [
        {key: value for key, value in item.items() if not str(key).startswith("_")}
        if isinstance(item, Mapping) else item
        for item in public.get("items", [])
    ]
    return (json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _parse_range(value: str | None, size: int) -> tuple[int, int] | None:
    if value is None:
        return 0, size - 1
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or size <= 0 or not any(match.groups()):
        return None
    first, last = match.groups()
    try:
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        else:
            suffix = int(last)
            if suffix <= 0:
                return None
            start, end = max(0, size - suffix), size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or end < start:
        return None
    return start, end


def make_handler(document: Mapping[str, Any]):
    """Return a read-only HTTP handler for a validated manifest document."""

    item_map = {str(item["id"]): item for item in document["items"]}
    index_bytes = _html(document).encode("utf-8")
    manifest_bytes = _json_bytes(document)

    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "MusicMixPsychologyReview/1"

        def log_message(self, *_args: Any) -> None:
            pass

        def _reply(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _audio(self, path: Path) -> None:
            size = path.stat().st_size
            requested = self.headers.get("Range")
            span = _parse_range(requested, size)
            if span is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = span
            partial = requested is not None
            length = end - start + 1
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = length
                    while remaining:
                        block = stream.read(min(65536, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _route(self) -> None:
            path = unquote(urlsplit(self.path).path)
            if path in {"/", "/index.html"}:
                return self._reply(200, index_bytes, "text/html; charset=utf-8")
            if path == "/api/manifest":
                return self._reply(200, manifest_bytes, "application/json; charset=utf-8")
            match = re.fullmatch(r"/audio/([^/]+)", path)
            if match:
                item = item_map.get(match.group(1))
                if item is None:
                    return self._reply(404, b'{"error":"unknown audio item"}\n', "application/json; charset=utf-8")
                return self._audio(Path(item["_resolved_audio_path"]))
            return self._reply(404, b'{"error":"not found"}\n', "application/json; charset=utf-8")

        def do_GET(self) -> None:
            self._route()

        def do_HEAD(self) -> None:
            self._route()

        def do_POST(self) -> None:
            self.send_response(405)
            self.send_header("Allow", "GET, HEAD")
            self.send_header("Content-Length", "0")
            self.end_headers()

    return ReviewHandler


def create_server(manifest: str | Path = DEFAULT_MANIFEST, *, port: int = 8772,
                  project_root: str | Path | None = None) -> ThreadingHTTPServer:
    """Load a manifest and return a localhost-only HTTP server."""

    if not 0 <= int(port) <= 65535:
        raise ValueError("port must be between 0 and 65535")
    document = load_manifest(manifest, project_root=project_root)
    server = ThreadingHTTPServer(("127.0.0.1", int(port)), make_handler(document))
    server.review_manifest = document  # type: ignore[attr-defined]
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--serve", action="store_true", help="serve the read-only review until interrupted")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    document = load_manifest(args.manifest)
    summary = {"schema_version": SCHEMA, "items": len(document["items"]), "manifest": str(document["_manifest_path"])}
    if not args.serve:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    server = create_server(args.manifest, port=args.port)
    print(f"http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
