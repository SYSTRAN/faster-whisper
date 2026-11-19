import numpy as np
import pytest

from faster_whisper import BatchedInferencePipeline


@pytest.mark.parametrize("batch_size", [0, -1])
def test_nonpositive_batch_size_is_rejected_before_decoding(batch_size):
    pipeline = BatchedInferencePipeline(model=None)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        pipeline.transcribe(np.zeros(1, dtype=np.float32), batch_size=batch_size)
