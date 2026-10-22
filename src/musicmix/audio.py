"""Small, dependency-light PCM WAV ingestion primitives.

The project keeps one canonical audio representation at module boundaries: a
two-channel floating point waveform indexed by the source WAV sample rate.
This module deliberately handles only uncompressed integer PCM.  A later
ingest adapter may resample that representation to the project's canonical
48 kHz timeline; the low-level loader does not silently resample or repair
input files.
"""

from __future__ import annotations

import hashlib
import io
import os
import wave
from dataclasses import dataclass
from typing import Union

import numpy as np

from .paths import localize


PathLike = Union[str, os.PathLike[str]]


class AudioFormatError(ValueError):
    """Raised when a WAV file is not a supported integer PCM stream."""


@dataclass(frozen=True)
class AudioTrack:
    """An audio file decoded to a stereo floating point sample matrix.

    ``samples`` has shape ``(frames, channels)`` and values in approximately
    ``[-1, 1]``.  ``load_audio`` always returns two channels (mono input is
    duplicated), while accepting one-channel arrays here keeps the dataclass
    useful for small in-memory fixtures.
    """

    track_id: str
    path: str
    sample_rate: int
    samples: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("track_id must be a non-empty string")
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        array = np.asarray(self.samples)
        if array.ndim != 2:
            raise ValueError("samples must have shape (frames, channels)")
        if array.shape[1] < 1:
            raise ValueError("samples must contain at least one channel")
        if not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        elif array.dtype != np.float32 and array.dtype != np.float64:
            array = array.astype(np.float32)
        # A read-only view prevents accidental mutation of a content-addressed
        # track while retaining NumPy's efficient array interface.
        array = np.ascontiguousarray(array)
        array.setflags(write=False)
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "samples", array)

    @property
    def duration_samples(self) -> int:
        """Number of frames in the track (the canonical time unit)."""

        return int(self.samples.shape[0])


def _decode_integer_pcm(raw: bytes, sample_width: int, channels: int) -> np.ndarray:
    """Decode little-endian signed integer PCM into a float32 matrix."""

    if sample_width not in (2, 3, 4):
        raise AudioFormatError(
            f"unsupported PCM sample width: {sample_width * 8} bits; "
            "supported widths are 16, 24, and 32 bits"
        )
    frame_bytes = sample_width * channels
    if frame_bytes <= 0 or len(raw) % frame_bytes:
        raise AudioFormatError("truncated PCM data (not an integral number of frames)")

    if sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        scale = 32768.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        scale = 2147483648.0
    else:
        # NumPy has no native 24-bit integer dtype.  Assemble unsigned bytes,
        # then sign extend bit 23 before converting to float.
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        unsigned = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values_i32 = unsigned.copy()
        negative = (values_i32 & 0x800000) != 0
        values_i32[negative] -= 1 << 24
        values = values_i32.astype(np.float32)
        scale = 8388608.0

    decoded = (values / scale).reshape(-1, channels)
    # A mono source is made stereo at this boundary.  This is a channel copy,
    # not a panning operation, and therefore preserves mono content exactly.
    if channels == 1:
        decoded = np.repeat(decoded, 2, axis=1)
    return np.ascontiguousarray(decoded, dtype=np.float32)


def load_audio(path: PathLike) -> AudioTrack:
    """Read an integer PCM WAV file and return a content-addressed track.

    Supported inputs are little-endian, uncompressed PCM WAV files with one or
    two channels and 16/24/32-bit samples.  The original sample rate is
    preserved.  Unsupported WAV variants raise :class:`AudioFormatError`
    instead of being silently coerced.
    """

    path_string = os.fspath(path)
    if isinstance(path_string, bytes):
        path_string = os.fsdecode(path_string)
    if not isinstance(path_string, str):
        raise TypeError("path must be a filesystem path")
    path_string = os.fspath(localize(path_string))

    with open(path_string, "rb") as handle:
        payload = handle.read()
    track_id = hashlib.sha256(payload).hexdigest()

    try:
        wav = wave.open(io.BytesIO(payload), "rb")
    except (EOFError, wave.Error, ValueError) as exc:
        raise AudioFormatError(f"unsupported or malformed WAV file: {path_string}") from exc

    try:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        declared_frames = wav.getnframes()
        comptype = wav.getcomptype()
        if comptype != "NONE":
            raise AudioFormatError(f"compressed WAV is unsupported: {comptype}")
        if channels not in (1, 2):
            raise AudioFormatError(
                f"unsupported channel count: {channels}; only mono and stereo are supported"
            )
        if sample_rate <= 0:
            raise AudioFormatError("WAV sample rate must be positive")
        raw = wav.readframes(declared_frames)
    except (EOFError, wave.Error, ValueError) as exc:
        if isinstance(exc, AudioFormatError):
            raise
        raise AudioFormatError(f"could not decode WAV file: {path_string}") from exc
    finally:
        wav.close()

    expected = int(declared_frames) * int(channels) * int(sample_width)
    if len(raw) != expected:
        raise AudioFormatError(
            f"truncated WAV payload: expected {expected} bytes, received {len(raw)}"
        )
    samples = _decode_integer_pcm(raw, sample_width, channels)
    return AudioTrack(track_id, path_string, int(sample_rate), samples)


__all__ = ["AudioFormatError", "AudioTrack", "load_audio"]
