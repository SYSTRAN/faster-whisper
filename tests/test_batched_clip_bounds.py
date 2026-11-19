import logging

from types import SimpleNamespace

import numpy as np
import pytest

from faster_whisper import BatchedInferencePipeline


@pytest.mark.parametrize("start,end", [(-0.1, 0.5), (0.2, 1.2), (0.8, 0.5), (0.8, 0.8)])
def test_rejects_invalid_explicit_clips(start, end):
    model = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=10, chunk_length=30),
        logger=logging.getLogger(__name__),
    )
    pipeline = BatchedInferencePipeline(model=model)
    with pytest.raises(ValueError, match="clip_timestamps"):
        pipeline.transcribe(
            np.zeros(10, dtype=np.float32),
            clip_timestamps=[{"start": start, "end": end}],
        )
