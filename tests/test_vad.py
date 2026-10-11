import pytest

from faster_whisper.vad import SpeechTimestampsMap


@pytest.mark.parametrize(
    "time,is_end,expected",
    [
        (0, False, 0),
        (0.5, False, 0),
        (1, False, 1),
        (1, True, 0),
        (1.5, False, 1),
        (1.5, True, 1),
        (2, False, 2),
        (2, True, 1),
        (3, False, 2),
        (3, True, 2),
        (4, False, 2),
        (4, True, 2),
    ],
)
def test_speech_timestamp_chunk_index(time, is_end, expected):
    timestamp_map = SpeechTimestampsMap(
        [
            {"start": 0, "end": 16000},
            {"start": 32000, "end": 48000},
            {"start": 64000, "end": 80000},
        ],
        sampling_rate=16000,
    )

    assert timestamp_map.get_chunk_index(time, is_end=is_end) == expected


def test_speech_timestamp_boundary_restoration():
    timestamp_map = SpeechTimestampsMap(
        [{"start": 0, "end": 16000}, {"start": 32000, "end": 48000}],
        sampling_rate=16000,
    )

    assert timestamp_map.get_original_time(1, is_end=True) == 1
    assert timestamp_map.get_original_time(1) == 2
