import os

import numpy as np
import pytest

from faster_whisper import WhisperModel, decode_audio


@pytest.fixture(scope="module")
def language_model():
    return WhisperModel("tiny", device="cpu", cpu_threads=1)


@pytest.fixture
def bilingual_parts(data_dir):
    english = decode_audio(os.path.join(data_dir, "jfk.flac"))[: 3 * 16000]
    german = decode_audio(os.path.join(data_dir, "multilingual.mp3"))[
        30 * 16000 : 57 * 16000
    ]
    return english, german


@pytest.mark.parametrize("start", [0, 5])
@pytest.mark.parametrize("as_string", [False, True])
def test_language_detection_stays_within_selected_clip(
    language_model, bilingual_parts, start, as_string
):
    english, german = bilingual_parts
    audio = np.concatenate((german[: start * 16000], english, german))
    clips = [start, start + 3]
    if as_string:
        clips = ",".join(str(timestamp) for timestamp in clips)
    features = language_model.feature_extractor(audio)
    expected = language_model.detect_language(
        features=features[..., start * 100 : (start + 3) * 100]
    )
    assert expected[0] == "en"

    segments, info = language_model.transcribe(
        audio, clip_timestamps=clips, beam_size=1, temperature=0
    )
    explicit_segments, _ = language_model.transcribe(
        audio, clip_timestamps=clips, language="en", beam_size=1, temperature=0
    )

    assert (
        info.language,
        info.language_probability,
        info.all_language_probs,
    ) == expected
    assert [segment.text for segment in segments] == [
        segment.text for segment in explicit_segments
    ]


@pytest.mark.parametrize("clips", [[0, 3, 10, 15], [0, 0, 0, 3]])
def test_language_detection_uses_first_nonempty_clip(
    language_model, bilingual_parts, clips
):
    english, german = bilingual_parts
    audio = np.concatenate((english, german))
    features = language_model.feature_extractor(audio)
    expected = language_model.detect_language(features=features[..., :300])

    segments, info = language_model.transcribe(audio, clip_timestamps=clips)
    segments.close()

    assert (
        info.language,
        info.language_probability,
        info.all_language_probs,
    ) == expected


@pytest.mark.parametrize("clips", [[0, 11], [0, 100], [1], "1", [0, 0]])
def test_clip_language_detection_preserves_open_and_empty_ranges(
    language_model, jfk_path, clips
):
    audio = decode_audio(jfk_path)
    features = language_model.feature_extractor(audio)
    start_frame = 100 if clips in ([1], "1") else 0
    expected = language_model.detect_language(features=features[..., start_frame:])

    segments, info = language_model.transcribe(audio, clip_timestamps=clips)
    segments.close()

    assert (
        info.language,
        info.language_probability,
        info.all_language_probs,
    ) == expected
