from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_linkseg_registry", ROOT / "tools" / "prepare_linkseg_registry.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_identity_gate_rejects_wrong_track_hash():
    expected = {"track_id": "a" * 64, "sample_rate": 48000, "duration_samples": 100}
    observed = {**expected, "track_id": "b" * 64}
    try:
        MODULE.require_identity(observed, expected, "test")
    except ValueError as exc:
        assert "track_id" in str(exc)
    else:
        raise AssertionError("wrong canonical identity was accepted")


def test_artifact_hash_gate_rejects_wrong_hash():
    try:
        MODULE.require_sha256("a" * 64, "b" * 64, "test artifact")
    except ValueError as exc:
        assert "SHA256" in str(exc)
    else:
        raise AssertionError("wrong artifact hash was accepted")
