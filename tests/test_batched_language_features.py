import numpy as np
import pytest

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio


@pytest.fixture(scope="module")
def language_model():
    return WhisperModel("tiny", device="cpu", cpu_threads=1)


@pytest.mark.parametrize("repeats,detection_segments", [(5, 1), (8, 2), (2, 1), (0, 1)])
@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_batched_language_features_are_limited_to_detection_windows(
    language_model, jfk_path, monkeypatch, repeats, detection_segments, threshold
):
    clip = decode_audio(jfk_path)
    audio = np.tile(clip, repeats)
    duration = len(clip) / language_model.feature_extractor.sampling_rate
    clips = [{"start": i * duration, "end": (i + 1) * duration} for i in range(repeats)]
    feature = language_model.feature_extractor(clip)[..., :-1]
    dummy = np.full((language_model.model.n_mels, 1), -1.5, dtype=np.float32)
    full_features = np.concatenate([feature] * repeats + [dummy], axis=1)
    options = {
        "language_detection_segments": detection_segments,
        "language_detection_threshold": threshold,
    }
    expected = language_model.detect_language(features=full_features, **options)
    original_detect_language = language_model.detect_language
    observed = {}

    def record_features(*args, **kwargs):
        observed["features"] = kwargs["features"]
        return original_detect_language(*args, **kwargs)

    monkeypatch.setattr(language_model, "detect_language", record_features)
    segments, info = BatchedInferencePipeline(language_model).transcribe(
        audio, clip_timestamps=clips, vad_filter=False, **options
    )
    segments.close()

    frame_limit = detection_segments * language_model.feature_extractor.nb_max_frames
    actual_features = observed["features"]
    # One extra frame is the existing sentinel for empty audio.
    assert actual_features.shape[-1] <= frame_limit + 1
    np.testing.assert_array_equal(
        actual_features[..., :frame_limit], full_features[..., :frame_limit]
    )
    assert (
        info.language,
        info.language_probability,
        info.all_language_probs,
    ) == expected
