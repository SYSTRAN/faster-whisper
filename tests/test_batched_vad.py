import logging

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from faster_whisper import BatchedInferencePipeline
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.vad import VadOptions


@pytest.fixture
def continuous_speech_pipeline(monkeypatch):
    model = SimpleNamespace(
        feature_extractor=FeatureExtractor(),
        model=SimpleNamespace(is_multilingual=False),
        hf_tokenizer=Mock(),
        logger=logging.getLogger("faster_whisper"),
    )
    pipeline = BatchedInferencePipeline(model)
    metadata = []

    def capture_segments(features, tokenizer, chunks_metadata, *args):
        metadata.extend(chunks_metadata)
        return iter(())

    monkeypatch.setattr(pipeline, "_batched_segments_generator", capture_segments)
    monkeypatch.setattr(
        "faster_whisper.vad.get_vad_model",
        lambda: lambda audio: np.ones(len(audio) // 512),
    )
    return pipeline, metadata


@pytest.mark.parametrize("chunk_length", [10, 30])
@pytest.mark.parametrize("max_speech_duration_s", [float("inf"), 120])
@pytest.mark.parametrize("as_dict", [False, True])
def test_batched_vad_limits_continuous_speech(
    continuous_speech_pipeline, chunk_length, max_speech_duration_s, as_dict
):
    pipeline, metadata = continuous_speech_pipeline
    audio = np.ones(65 * 16000, dtype=np.float32)
    parameters = {"max_speech_duration_s": max_speech_duration_s}
    if not as_dict:
        parameters = VadOptions(**parameters)

    _, info = pipeline.transcribe(
        audio,
        language="en",
        chunk_length=chunk_length,
        vad_parameters=parameters,
        suppress_tokens=[],
    )

    assert metadata
    assert all(0 < chunk["duration"] <= chunk_length for chunk in metadata)
    assert sum(chunk["duration"] for chunk in metadata) == pytest.approx(65)
    assert info.duration_after_vad == pytest.approx(65)


@pytest.mark.parametrize("max_speech_duration_s", [2, 120, float("inf")])
def test_batched_vad_preserves_caller_options(
    continuous_speech_pipeline, max_speech_duration_s
):
    pipeline, metadata = continuous_speech_pipeline
    options = VadOptions(
        threshold=0.7,
        min_silence_duration_ms=400,
        speech_pad_ms=100,
        max_speech_duration_s=max_speech_duration_s,
    )
    before = vars(options).copy()

    _, info = pipeline.transcribe(
        np.ones(65 * 16000, dtype=np.float32),
        language="en",
        chunk_length=30,
        vad_parameters=options,
        suppress_tokens=[],
    )

    assert vars(options) == before
    assert info.vad_options.threshold == 0.7
    assert info.vad_options.min_silence_duration_ms == 400
    assert info.vad_options.speech_pad_ms == 100
    assert info.vad_options.max_speech_duration_s == min(max_speech_duration_s, 30)
    assert all(0 < chunk["duration"] <= 30 for chunk in metadata)
    if max_speech_duration_s < 30:
        assert all(
            (segment["end"] - segment["start"]) / 16000 <= max_speech_duration_s
            for chunk in metadata
            for segment in chunk["segments"]
        )
    assert sum(chunk["duration"] for chunk in metadata) == pytest.approx(65)


@pytest.mark.parametrize("speech_pad_ms", [0, 1, 10, 16, 400])
@pytest.mark.parametrize("chunk_length", [10, 30])
def test_batched_vad_preserves_samples_at_hard_splits(
    continuous_speech_pipeline, speech_pad_ms, chunk_length
):
    pipeline, metadata = continuous_speech_pipeline
    audio = np.sin(np.arange(1168304, dtype=np.float32) * 0.013) * 0.2

    pipeline.transcribe(
        audio,
        language="en",
        chunk_length=chunk_length,
        vad_parameters=VadOptions(speech_pad_ms=speech_pad_ms),
        suppress_tokens=[],
    )

    restored = np.concatenate(
        [
            audio[segment["start"] : segment["end"]]
            for chunk in metadata
            for segment in chunk["segments"]
        ]
    )
    np.testing.assert_array_equal(restored, audio)
    assert all(0 < chunk["duration"] <= chunk_length for chunk in metadata)
