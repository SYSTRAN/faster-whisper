import wave

import numpy as np
import pytest

from faster_whisper.audio import decode_audio


@pytest.mark.parametrize("stereo", [False, True])
def test_pcm_extremes_decode_exactly_as_float32(tmp_path, stereo):
    samples = np.array([-32768, -12345, -1, 0, 1, 12345, 32767], dtype=np.int16)
    pcm = np.column_stack((samples, samples[::-1])) if stereo else samples
    path = tmp_path / "extremes.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2 if stereo else 1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm.tobytes())
    actual = decode_audio(str(path), split_stereo=stereo)
    expected = pcm.astype(np.float32) / 32768.0
    if stereo:
        for index, channel in enumerate(actual):
            assert channel.dtype == np.float32
            np.testing.assert_array_equal(channel, expected[:, index])
    else:
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)
