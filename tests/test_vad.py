import numpy as np

from faster_whisper.vad import collect_chunks


def test_collect_chunks():
    audio = np.arange(100, dtype=np.float32)
    chunks = [
        {"start": 10, "end": 20},
        {"start": 30, "end": 40},
        {"start": 50, "end": 60},
    ]

    audio_chunks, metadata = collect_chunks(audio, chunks, sampling_rate=1)

    assert len(audio_chunks) == 1
    assert audio_chunks[0].dtype == np.float32
    assert np.array_equal(
        audio_chunks[0], np.concatenate((audio[10:20], audio[30:40], audio[50:60]))
    )
    assert metadata == [{"offset": 0.0, "duration": 30.0, "segments": chunks}]


def test_collect_chunks_max_duration():
    audio = np.arange(100, dtype=np.float32)
    chunks = [
        {"start": 10, "end": 20},
        {"start": 30, "end": 40},
        {"start": 50, "end": 60},
    ]

    audio_chunks, metadata = collect_chunks(
        audio, chunks, sampling_rate=1, max_duration=20
    )

    assert len(audio_chunks) == 2
    assert np.array_equal(audio_chunks[0], np.concatenate((audio[10:20], audio[30:40])))
    assert np.array_equal(audio_chunks[1], audio[50:60])
    assert metadata == [
        {"offset": 0.0, "duration": 20.0, "segments": chunks[:2]},
        {"offset": 20.0, "duration": 10.0, "segments": chunks[2:]},
    ]
