import numpy as np
import pytest

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.audio import pad_or_trim


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_batch_size_is_rejected_before_accessing_model(value):
    pipeline = BatchedInferencePipeline(model=None)
    with pytest.raises(ValueError, match="batch_size.*positive integer"):
        pipeline.transcribe(np.zeros(160, dtype=np.float32), batch_size=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_language_detection_segments_are_rejected_early(value):
    model = WhisperModel.__new__(WhisperModel)
    with pytest.raises(
        ValueError, match="language_detection_segments.*positive integer"
    ):
        model.detect_language(
            features=np.zeros((80, 3000), dtype=np.float32),
            language_detection_segments=value,
        )


def test_negative_padding_length_is_rejected():
    with pytest.raises(ValueError, match="length.*non-negative"):
        pad_or_trim(np.ones((2, 3)), -1)


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
@pytest.mark.parametrize("length", [0, 2, 10])
def test_pad_or_trim_preserves_values_and_copy_behavior(axis, length):
    array = np.arange(48, dtype=np.float32).reshape(8, 6)[::2]
    if array.shape[axis] > length:
        expected = np.take(array, range(length), axis=axis)
    else:
        padding = [(0, 0)] * array.ndim
        padding[axis] = (0, length - array.shape[axis])
        expected = np.pad(array, padding)
    actual = pad_or_trim(array, length, axis=axis)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == array.dtype
    if array.shape[axis] > length:
        assert not np.shares_memory(actual, array)
