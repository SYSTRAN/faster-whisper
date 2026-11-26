import os

import numpy as np
import pytest

from faster_whisper import WhisperModel, decode_audio


@pytest.fixture(scope="module")
def prefix_model():
    return WhisperModel("tiny", device="cpu", cpu_threads=1)


@pytest.mark.parametrize("clips", ["1", [1], [0, 0, 1]])
def test_prefix_applies_to_the_first_selected_window(prefix_model, data_dir, clips):
    prefix = "your ComfyUI"
    segments, _ = prefix_model.transcribe(
        os.path.join(data_dir, "hotwords.mp3"),
        language="en",
        clip_timestamps=clips,
        prefix=prefix,
        beam_size=1,
        temperature=0,
    )
    segments = list(segments)

    assert segments[0].text.lstrip().startswith(prefix)
    assert segments[0].start >= 1


@pytest.mark.parametrize("start", [0, 1])
def test_prefix_is_not_repeated_on_later_clips(
    prefix_model, data_dir, monkeypatch, start
):
    clip = decode_audio(os.path.join(data_dir, "hotwords.mp3"))
    duration = len(clip) / prefix_model.feature_extractor.sampling_rate
    audio = np.tile(clip, 3)
    clip_timestamps = [
        point for i in range(3) for point in (i * duration + start, (i + 1) * duration)
    ]
    prefix = "your ComfyUI" if start else "Then you go into your ComfyUI"
    original_get_prompt = prefix_model.get_prompt
    prefixes = []

    def record_prompt(*args, **kwargs):
        prefixes.append(kwargs.get("prefix"))
        return original_get_prompt(*args, **kwargs)

    monkeypatch.setattr(prefix_model, "get_prompt", record_prompt)
    segments, _ = prefix_model.transcribe(
        audio,
        language="en",
        clip_timestamps=clip_timestamps,
        prefix=prefix,
        beam_size=1,
        temperature=0,
        condition_on_previous_text=False,
    )
    segments = list(segments)

    assert len(prefixes) >= 3
    assert prefixes[0] == prefix
    assert all(value is None for value in prefixes[1:])
    assert segments[0].text.lstrip().startswith(prefix)
