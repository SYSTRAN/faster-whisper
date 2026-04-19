"""Tests for the pluggable audio_backend parameter on WhisperModel.

The goal of these tests is to lock in the public contract of the
`audio_backend` knob added alongside the PyAV backend:

  1. The default backend ("pyav") still produces the expected float32
     waveform — no behavior change for existing users.
  2. The "ffmpeg" backend decodes the same file to an equivalent waveform
     (within a small tolerance, since both pipelines resample through s16).
  3. An unknown backend is rejected eagerly, both at `decode_audio` and at
     `WhisperModel` construction time, with a helpful ValueError.
  4. When ffmpeg is not on PATH and `FFMPEG_EXE` is unset, the ffmpeg
     backend raises a RuntimeError with an install hint (we simulate this
     by monkeypatching the PATH lookup, so the test does not depend on
     the runner's system state).

These tests intentionally avoid loading the actual Whisper model weights
(which would require a network download) — they exercise `decode_audio`
directly, plus WhisperModel's constructor-time validation.
"""

import os
import shutil

import numpy as np
import pytest

from faster_whisper import WhisperModel, decode_audio
from faster_whisper.audio import SUPPORTED_AUDIO_BACKENDS


def _ffmpeg_available() -> bool:
    return os.environ.get("FFMPEG_EXE") or shutil.which("ffmpeg") is not None


def test_supported_backends_listed():
    assert "pyav" in SUPPORTED_AUDIO_BACKENDS
    assert "ffmpeg" in SUPPORTED_AUDIO_BACKENDS


def test_decode_audio_default_backend_pyav(jfk_path):
    audio = decode_audio(jfk_path, sampling_rate=16000)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    # jfk.flac is ~11s at 16kHz → ~176000 samples, give ourselves a loose
    # bracket so minor resampler differences do not flake this test.
    assert 150_000 <= audio.shape[0] <= 200_000


@pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="ffmpeg binary not available on PATH / FFMPEG_EXE",
)
def test_decode_audio_ffmpeg_backend_matches_pyav(jfk_path):
    pyav_audio = decode_audio(jfk_path, sampling_rate=16000, backend="pyav")
    ff_audio = decode_audio(jfk_path, sampling_rate=16000, backend="ffmpeg")

    assert isinstance(ff_audio, np.ndarray)
    assert ff_audio.dtype == np.float32

    # Both pipelines go through s16 at 16kHz mono — sample counts should
    # match to within a frame, and RMS should be very close.
    assert abs(ff_audio.shape[0] - pyav_audio.shape[0]) < 320  # <20ms drift

    n = min(ff_audio.shape[0], pyav_audio.shape[0])
    rms_pyav = float(np.sqrt(np.mean(pyav_audio[:n] ** 2)))
    rms_ff = float(np.sqrt(np.mean(ff_audio[:n] ** 2)))
    assert abs(rms_pyav - rms_ff) < 1e-2


def test_decode_audio_invalid_backend_raises(jfk_path):
    with pytest.raises(ValueError, match="Unsupported audio_backend"):
        decode_audio(jfk_path, backend="gstreamer")


def test_whisper_model_invalid_audio_backend_raises():
    # Validation must happen before we touch ctranslate2 / download weights.
    with pytest.raises(ValueError, match="Unsupported audio_backend"):
        WhisperModel("tiny.en", audio_backend="gstreamer")


def test_decode_audio_ffmpeg_backend_missing_binary_raises(monkeypatch, jfk_path):
    # Simulate ffmpeg not being installed: clear FFMPEG_EXE and make
    # shutil.which return None. Everything else stays untouched.
    monkeypatch.delenv("FFMPEG_EXE", raising=False)
    monkeypatch.setattr("faster_whisper.audio.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        decode_audio(jfk_path, backend="ffmpeg")
