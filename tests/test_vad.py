import pytest

from faster_whisper.transcribe import Segment, Word, restore_speech_timestamps
from faster_whisper.vad import SpeechTimestampsMap


@pytest.fixture(params=[160000, 257280, 259840])
def speech_chunks(request):
    boundary = request.param
    return [
        {"start": 0, "end": boundary},
        {"start": boundary + 8000, "end": boundary + 24000},
    ]


@pytest.mark.parametrize("is_end", [False, True])
@pytest.mark.parametrize("sample_offset", [-1, -0.25, 0, 0.25, 1])
def test_timestamp_map_chunk_boundary(speech_chunks, is_end, sample_offset):
    sampling_rate = 16000
    boundary = speech_chunks[0]["end"] / sampling_rate
    time = boundary + sample_offset / sampling_rate
    expected_index = int(sample_offset > 0 or (sample_offset == 0 and not is_end))

    ts_map = SpeechTimestampsMap(speech_chunks, sampling_rate)

    assert ts_map.get_chunk_index(time, is_end=is_end) == expected_index
    assert ts_map.get_original_time(time, is_end=is_end) == round(
        time + expected_index * 0.5, 2
    )


def test_timestamp_map_explicit_chunk_index(speech_chunks):
    boundary = speech_chunks[0]["end"] / 16000
    ts_map = SpeechTimestampsMap(speech_chunks, 16000)

    assert ts_map.get_original_time(boundary, chunk_index=0) == round(boundary, 2)
    assert ts_map.get_original_time(boundary, chunk_index=1) == round(boundary + 0.5, 2)


@pytest.mark.parametrize("is_end", [False, True])
@pytest.mark.parametrize("time", [0, 0.5, 1, 2])
def test_timestamp_map_single_chunk(time, is_end):
    ts_map = SpeechTimestampsMap([{"start": 8000, "end": 24000}], 16000)

    assert ts_map.get_chunk_index(time, is_end=is_end) == 0
    assert ts_map.get_original_time(time, is_end=is_end) == time + 0.5


def make_segment(start, end, words=None):
    return Segment(
        id=0,
        seek=0,
        start=start,
        end=end,
        text="speech",
        tokens=[],
        avg_logprob=0,
        compression_ratio=1,
        no_speech_prob=0,
        words=words,
        temperature=0,
    )


def test_restore_speech_timestamps_at_chunk_boundary(speech_chunks):
    boundary = speech_chunks[0]["end"] / 16000
    segments = [
        make_segment(boundary - 0.02, boundary),
        make_segment(boundary, boundary + 0.02),
    ]

    restored = list(restore_speech_timestamps(segments, speech_chunks, 16000))

    assert (restored[0].start, restored[0].end) == (
        round(boundary - 0.02, 2),
        round(boundary, 2),
    )
    assert (restored[1].start, restored[1].end) == (
        round(boundary + 0.5, 2),
        round(boundary + 0.52, 2),
    )


def test_restore_word_timestamps_at_chunk_boundary(speech_chunks):
    boundary = speech_chunks[0]["end"] / 16000
    word = Word(boundary - 0.01, boundary + 0.01, "speech", 1)
    segment = make_segment(word.start, word.end, [word])

    restored = list(restore_speech_timestamps([segment], speech_chunks, 16000))[0]

    assert (word.start, word.end) == (
        round(boundary + 0.49, 2),
        round(boundary + 0.51, 2),
    )
    assert restored.start == word.start
    assert restored.end == word.end
