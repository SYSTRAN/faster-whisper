import numpy as np
import pytest

from faster_whisper.vad import VadOptions, get_speech_timestamps


@pytest.mark.parametrize(
    ("num_samples", "last_probability", "min_duration_ms", "expected"),
    [
        (512, 0.0, 32, [{"start": 0, "end": 512}]),
        (512, 0.9, 32, [{"start": 0, "end": 512}]),
        (511, 0.9, 32, []),
        (0, 0.9, 0, []),
    ],
)
def test_vad_keeps_speech_at_the_minimum_duration(
    monkeypatch, num_samples, last_probability, min_duration_ms, expected
):
    monkeypatch.setattr(
        "faster_whisper.vad.get_vad_model",
        lambda: lambda padded: [0.9, last_probability][: len(padded) // 512],
    )
    options = VadOptions(
        min_speech_duration_ms=min_duration_ms,
        min_silence_duration_ms=0,
        speech_pad_ms=0,
    )

    audio = np.zeros(num_samples, dtype=np.float32)
    assert get_speech_timestamps(audio, options) == expected
