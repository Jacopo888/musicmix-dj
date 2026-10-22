import http.client
import json
from http.server import ThreadingHTTPServer
import threading
import wave

import pytest

from tools.serve_psychology_review import ReviewManifestError, create_server, load_manifest


def _wav(path, payload=b"0123456789"):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(1)
        stream.setframerate(8000)
        stream.writeframes(payload)


def _manifest(path, audio_path="audio/clip.wav"):
    path.write_text(json.dumps({
        "schema_version": "musicmix.psychology-review.1",
        "items": [{
            "id": "item-01",
            "title": "Passaggio di prova",
            "audio_path": audio_path,
            "candidate_id": "candidate-01",
            "A_out": {"track": "A / Outro", "seconds": 12.5},
            "B_in": {"track": "B / Intro", "seconds": 3.25},
            "action": "execute",
            "intent": "preserve_groove",
            "notes": ["contesto 24 s per lato"],
            "warnings": ["ASR: frase ignota", "tempo non verificato"],
        }],
    }), encoding="utf-8")


@pytest.fixture
def review_server(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "clip.wav"
    _wav(audio)
    manifest_path = tmp_path / "review_manifest.json"
    _manifest(manifest_path)
    server = create_server(manifest_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, audio.read_bytes()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(server, method, path, headers=None, body=None):
    client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    client.request(method, path, body=body, headers=headers or {})
    response = client.getresponse()
    data = response.read()
    client.close()
    return response, data


def test_manifest_is_validated_and_html_is_italian_read_only(review_server):
    server, _ = review_server
    response, body = _request(server, "GET", "/")
    assert response.status == 200
    html = body.decode("utf-8")
    assert "Passaggio di prova" in html
    assert "A out:" in html and "B in:" in html
    assert "preserva il groove" in html
    assert "ASR: frase ignota" in html
    response, _ = _request(server, "POST", "/audio/item-01", body=b"{}")
    assert response.status == 405
    assert response.getheader("Allow") == "GET, HEAD"


def test_audio_supports_get_range_head_and_416(review_server):
    server, data = review_server
    size = len(data)
    response, body = _request(server, "GET", "/audio/item-01")
    assert response.status == 200
    assert body == data
    for requested, expected, content_range in (("bytes=1-3", data[1:4], f"bytes 1-3/{size}"),
                                                ("bytes=2-", data[2:], f"bytes 2-{size - 1}/{size}"),
                                                ("bytes=-2", data[-2:], f"bytes {size - 2}-{size - 1}/{size}")):
        response, body = _request(server, "GET", "/audio/item-01", {"Range": requested})
        assert response.status == 206
        assert response.getheader("Accept-Ranges") == "bytes"
        assert response.getheader("Content-Range") == content_range
        assert body == expected
    response, body = _request(server, "HEAD", "/audio/item-01", {"Range": "bytes=1-3"})
    assert response.status == 206
    assert response.getheader("Content-Range") == f"bytes 1-3/{size}"
    assert int(response.getheader("Content-Length")) == 3
    assert body == b""
    for requested in ("bytes=999999-", "bytes=3-1", "bytes=-0", "bytes=0-1,3-4"):
        response, body = _request(server, "GET", "/audio/item-01", {"Range": requested})
        assert response.status == 416
        assert response.getheader("Content-Range") == f"bytes */{size}"
        assert body == b""


def test_manifest_rejects_traversal_and_unknown_audio_id(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _wav(audio_dir / "clip.wav")
    manifest_path = tmp_path / "review_manifest.json"
    _manifest(manifest_path, "../clip.wav")
    with pytest.raises(ReviewManifestError, match="project-relative"):
        load_manifest(manifest_path)
    _manifest(manifest_path)
    server = create_server(manifest_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response, body = _request(server, "GET", "/audio/../clip.wav")
        assert response.status == 404
        assert body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
