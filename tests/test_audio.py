import io
import os

import pytest

from faster_whisper import decode_audio


@pytest.fixture
def mp3_data(data_dir):
    with open(os.path.join(data_dir, "hotwords.mp3"), "rb") as mp3_file:
        return mp3_file.read()


def test_decode_audio_joined_files(mp3_data):
    audio = decode_audio(io.BytesIO(mp3_data))

    # The tag that starts the second file is not a valid audio frame.
    joined = decode_audio(io.BytesIO(mp3_data + mp3_data))

    assert abs(len(joined) - 2 * len(audio)) < 0.2 * 16000


def test_decode_audio_invalid_frames(mp3_data):
    audio = decode_audio(io.BytesIO(mp3_data))

    middle = len(mp3_data) // 2
    damaged_data = mp3_data[:middle] + b"\xff" * 400 + mp3_data[middle + 400 :]
    damaged = decode_audio(io.BytesIO(damaged_data))

    # Only the invalid frames are lost, not the rest of the file.
    assert len(damaged) > 0.95 * len(audio)
