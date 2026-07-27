import os

from faster_whisper.audio import decode_audio


def test_decode_audio_with_mixed_sample_rate():
    # Regression test for https://github.com/SYSTRAN/faster-whisper/issues/1451
    #
    # This file's audio stream changes sample rate partway through
    # (44100 Hz -> 48000 Hz). Both AudioFifo and AudioResampler lock onto
    # the parameters of the first frame they see and raise ValueError when
    # a later frame doesn't match, so decode_audio used to crash on files
    # like this one instead of decoding them.
    path = os.path.join(os.path.dirname(__file__), "data", "mixed_samplerate.mp3")

    audio = decode_audio(path)

    assert audio.ndim == 1
    assert audio.dtype.name == "float32"
    assert audio.shape[0] > 0
