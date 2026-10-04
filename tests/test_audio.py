import io
import os

import numpy as np

from faster_whisper.audio import decode_audio

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
jfk_path = os.path.join(data_dir, "jfk.flac")
mixed_sample_rate_path = os.path.join(data_dir, "mixed_sample_rate.aac")


def test_decode_audio_sample_rates_and_dtypes():
    audio = decode_audio(jfk_path, sampling_rate=16000)
    assert audio.dtype == np.float32
    assert audio.shape == (176000,)


def test_decode_audio_with_mid_stream_sample_rate_change():
    """A stream that changes sample rate mid-file must decode fully (#1451).

    The fixture is one second of 44.1 kHz audio followed by one second of
    22.05 kHz audio in a single stream. Previously the AudioFifo (and on
    newer PyAV also the AudioResampler) locked onto the first frame's
    parameters and raised ``Frame does not match ... parameters``.
    """
    audio = decode_audio(mixed_sample_rate_path, sampling_rate=16000)
    assert audio.dtype == np.float32
    # Both segments survive: ~1.9 s of audio at 16 kHz (the exact length
    # depends on the AAC encoder's frame alignment).
    assert 28000 < len(audio) < 32000


def test_decode_audio_skips_invalid_packets():
    """A corrupt packet must not abort the rest of the stream."""
    with open(mixed_sample_rate_path, "rb") as f:
        payload = bytearray(f.read())

    # Corrupt a few bytes a quarter into the file: exactly one AAC packet
    # becomes undecodable, the rest of the stream must survive.
    pos = len(payload) // 4
    payload[pos : pos + 8] = b"\xff" * 8
    audio = decode_audio(io.BytesIO(bytes(payload)), sampling_rate=16000)
    assert audio.dtype == np.float32
    # The full file decodes to ~30465 samples; losing one packet costs a few
    # hundred. Without per-packet recovery the remainder of the stream (~three
    # quarters) would be lost instead.
    assert len(audio) > 25000
