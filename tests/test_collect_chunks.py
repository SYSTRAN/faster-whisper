import numpy as np

from faster_whisper.vad import collect_chunks


def test_collect_chunks_records_the_chunk_that_starts_a_group():
    audio = np.ones(32000, dtype=np.float32)
    first = {"start": 0, "end": 16000}
    second = {"start": 16000, "end": 32000}

    audios, metadata = collect_chunks(
        audio, [first, second], sampling_rate=16000, max_duration=1.5
    )

    assert [chunk.shape[0] for chunk in audios] == [16000, 16000]
    assert metadata[0]["segments"] == [first]
    assert metadata[1]["segments"] == [second]


def test_collect_chunks_does_not_prefix_an_empty_chunk_when_the_first_is_over_the_cap():
    audio = np.ones(32000, dtype=np.float32)
    chunk = {"start": 0, "end": 32000}

    audios, metadata = collect_chunks(
        audio, [chunk], sampling_rate=16000, max_duration=1.0
    )

    assert [piece.shape[0] for piece in audios] == [32000]
    assert metadata[0]["segments"] == [chunk]
    assert metadata[0]["duration"] == 2.0


def test_collect_chunks_keeps_later_chunks_in_the_new_group():
    audio = np.ones(24000, dtype=np.float32)
    chunks = [
        {"start": 0, "end": 8000},
        {"start": 8000, "end": 16000},
        {"start": 16000, "end": 24000},
    ]

    audios, metadata = collect_chunks(
        audio, chunks, sampling_rate=16000, max_duration=1.2
    )

    assert [piece.shape[0] for piece in audios] == [16000, 8000]
    assert metadata[0]["segments"] == chunks[:2]
    assert metadata[1]["segments"] == [chunks[2]]


def test_collect_chunks_keeps_chunks_that_fit_in_one_group():
    audio = np.ones(180, dtype=np.float32)
    chunks = [{"start": 0, "end": 100}, {"start": 100, "end": 180}]

    audios, metadata = collect_chunks(
        audio, chunks, sampling_rate=16000, max_duration=1.0
    )

    assert [piece.shape[0] for piece in audios] == [180]
    assert metadata[0]["segments"] == chunks


def test_collect_chunks_empty_input_returns_one_empty_array():
    audios, metadata = collect_chunks(np.ones(0, dtype=np.float32), [])

    assert len(audios) == 1
    assert audios[0].shape[0] == 0
    assert metadata[0]["segments"] == []
