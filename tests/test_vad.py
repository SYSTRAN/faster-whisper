import numpy as np
import pytest

from faster_whisper.vad import SpeechTimestampsMap


def _reference_chunk_index(ts_map, time, is_end):
    sample = int(time * ts_map.sampling_rate)
    ends = ts_map.chunk_end_sample
    if is_end and sample in ends:
        return ends.index(sample)
    return min(sum(end <= sample for end in ends), len(ends) - 1)


@pytest.mark.parametrize("is_end", [False, True])
def test_get_chunk_index(is_end):
    sampling_rate = 16000
    chunks = [
        {"start": 8000, "end": 24000},
        {"start": 32000, "end": 32000},
        {"start": 40000, "end": 72000},
        {"start": 100000, "end": 100160},
    ]
    ts_map = SpeechTimestampsMap(chunks, sampling_rate)

    rng = np.random.default_rng(0)
    times = [end / sampling_rate for end in ts_map.chunk_end_sample]
    times += rng.uniform(0, 5, 200).tolist()

    for time in times:
        assert ts_map.get_chunk_index(time, is_end) == _reference_chunk_index(
            ts_map, time, is_end
        )
