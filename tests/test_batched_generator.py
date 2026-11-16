from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from faster_whisper import BatchedInferencePipeline


@pytest.mark.parametrize("finish", ["exhaust", "close", "error"])
def test_batched_generator_cleans_up_progress_and_timestamps(monkeypatch, finish):
    pipeline = BatchedInferencePipeline(model=Mock())
    progress = Mock()
    monkeypatch.setattr("faster_whisper.transcribe.tqdm", lambda **kwargs: progress)
    initial_timestamps = []

    def forward(*args):
        initial_timestamps.append(pipeline.last_speech_timestamp)
        pipeline.last_speech_timestamp = 8.0
        if finish == "error":
            raise RuntimeError("inference failed")
        segment = {
            "seek": 0,
            "text": " test",
            "start": 0.0,
            "end": 8.0,
            "words": [],
            "tokens": [1],
            "avg_logprob": -0.1,
            "no_speech_prob": 0.0,
            "compression_ratio": 1.0,
        }
        return [[segment, segment]]

    monkeypatch.setattr(pipeline, "forward", forward)
    options = SimpleNamespace(word_timestamps=True, temperatures=[0.0])

    def generate():
        return pipeline._batched_segments_generator(
            np.zeros((1, 80, 3000), dtype=np.float32),
            None,
            [{}],
            1,
            options,
            True,
        )

    segments = generate()
    if finish == "error":
        with pytest.raises(RuntimeError, match="inference failed"):
            next(segments)
    elif finish == "close":
        assert next(segments).text == " test"
        assert pipeline.last_speech_timestamp == 8.0
        segments.close()
    else:
        assert len(list(segments)) == 2

    assert pipeline.last_speech_timestamp == 0.0
    progress.close.assert_called_once_with()

    # Reusing the pipeline must not inherit timestamps from the previous audio.
    next_segments = generate()
    if finish == "error":
        with pytest.raises(RuntimeError, match="inference failed"):
            next(next_segments)
    else:
        next(next_segments)
        next_segments.close()
    assert initial_timestamps == [0.0, 0.0]
