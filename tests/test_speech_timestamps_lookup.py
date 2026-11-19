import bisect

import pytest

from faster_whisper.vad import SpeechTimestampsMap


@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [
            {"start": 0, "end": 100},
            {"start": 200, "end": 300},
            {"start": 400, "end": 500},
        ],
        [
            {"start": 0, "end": 100},
            {"start": 200, "end": 200},
            {"start": 300, "end": 400},
        ],
    ],
)
@pytest.mark.parametrize("is_end", [False, True])
def test_lookup_preserves_boundary_and_duplicate_semantics(chunks, is_end):
    mapping = SpeechTimestampsMap(chunks, sampling_rate=100)
    ends = mapping.chunk_end_sample
    for time in [-1, 0, 0.99, 1, 1.01, 1.99, 2, 3, 4]:
        sample = int(time * 100)
        expected = (
            ends.index(sample)
            if is_end and sample in ends
            else min(bisect.bisect_right(ends, sample), len(ends) - 1)
        )
        assert mapping.get_chunk_index(time, is_end) == expected


def test_boundary_restores_time_before_or_after_removed_silence():
    mapping = SpeechTimestampsMap(
        [{"start": 0, "end": 100}, {"start": 200, "end": 300}],
        sampling_rate=100,
    )
    assert mapping.get_original_time(1, is_end=True) == 1
    assert mapping.get_original_time(1, is_end=False) == 2
