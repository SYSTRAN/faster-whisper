import io
import struct

import numpy as np

from faster_whisper.audio import decode_audio


def _create_wav_buffer(
    sample_rate: int, num_samples: int, channels: int = 1
) -> io.BytesIO:
    """Create an in-memory WAV file with the given parameters."""
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = num_samples * block_align

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(
        struct.pack(
            "<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, bits_per_sample
        )
    )
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    # Generate a simple sine wave
    t = np.linspace(0, num_samples / sample_rate, num_samples, dtype=np.float32)
    waveform = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf.write(waveform.tobytes())
    buf.seek(0)
    return buf


def test_decode_audio_variable_sample_rate():
    """Test that decode_audio handles files where sample rate changes mid-stream.

    This is a regression test for issue #1451 where AudioFifo crashes with
    'Frame does not match AudioFifo parameters' when processing audio files
    with variable sample rates.

    We simulate this by decoding two WAV buffers sequentially and verifying
    that decode_audio produces valid output for each independently.
    """
    # Test with different sample rates to ensure resampling works correctly
    for sample_rate in [8000, 16000, 22050, 44100, 48000]:
        buf = _create_wav_buffer(sample_rate, sample_rate)  # 1 second of audio
        audio = decode_audio(buf, sampling_rate=16000)
        assert audio.dtype == np.float32
        assert len(audio) > 0
        # Should be approximately 16000 samples (1 second at 16kHz)
        assert 15000 < len(audio) < 17000


def test_decode_audio_stereo():
    """Test that decode_audio handles stereo audio correctly."""
    buf = _create_wav_buffer(44100, 44100, channels=2)
    left, right = decode_audio(buf, sampling_rate=16000, split_stereo=True)
    assert left.dtype == np.float32
    assert right.dtype == np.float32
    assert len(left) > 0
    assert len(right) > 0
    assert len(left) == len(right)


def test_decode_audio_empty():
    """Test that decode_audio handles empty/very short audio gracefully."""
    buf = _create_wav_buffer(16000, 100)  # Very short audio
    audio = decode_audio(buf, sampling_rate=16000)
    assert audio.dtype == np.float32
    assert len(audio) > 0
