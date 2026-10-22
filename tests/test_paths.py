from pathlib import Path

import pytest

from musicmix.paths import localize


@pytest.fixture
def root(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    (repo / "runs" / "pool_v8" / "audio").mkdir(parents=True)
    monkeypatch.setenv("MUSICMIX_ROOT", str(repo))
    return repo


def test_windows_backslash_path_is_rebased(root):
    recorded = r"Z:\Data\nobody\Downloads\Mix in C\music_mixing_progetto\runs\pool_v8\audio\creature.wav"
    assert localize(recorded) == root / "runs" / "pool_v8" / "audio" / "creature.wav"


def test_windows_forward_slash_and_worktree_anchor(root):
    assert localize("Z:/Data/nobody/Downloads/Mix in C/music_mixing_progetto/runs/x.json") == root / "runs" / "x.json"
    assert localize("d:/work/musicmix_cloud/runs/y.json") == root / "runs" / "y.json"


def test_wsl_mount_path_is_rebased(root):
    recorded = "/mnt/z/Data/nobody/Downloads/Mix in C/music_mixing_progetto/runs/pool_v8/audio/a.wav"
    assert localize(recorded) == root / "runs" / "pool_v8" / "audio" / "a.wav"


def test_last_anchor_wins_and_giudizi_is_sibling(root):
    assert localize(r"C:\Mix in C\giudizi json\v12_ab.json") == root.parent / "giudizi json" / "v12_ab.json"
    nested = r"C:\old\music_mixing_progetto\backup\music_mixing_progetto\runs\z.wav"
    assert localize(nested) == root / "runs" / "z.wav"


def test_explicit_root_overrides_environment(root, tmp_path):
    other = tmp_path / "other"
    assert localize(r"C:\x\music_mixing_progetto\runs\a.wav", other) == other / "runs" / "a.wav"


def test_existing_path_is_returned_as_given(root):
    target = root / "runs" / "pool_v8" / "audio" / "real.wav"
    target.write_bytes(b"")
    assert localize(target) == target and localize(str(target)) == target


def test_unanchored_paths_pass_through(root):
    assert localize(r"C:\Data\someone\elsewhere\a.wav") == Path(r"C:\Data\someone\elsewhere\a.wav")
    assert localize("/mnt/c/elsewhere/a.wav") == Path("/mnt/c/elsewhere/a.wav")
    assert localize(root / "missing.wav") == root / "missing.wav"
    assert localize(r"runs\pool_v8\b.wav") == Path("runs/pool_v8/b.wav")
