import numpy as np

from faster_whisper.vad import get_vad_model


def test_vad_model_does_not_modify_the_input_audio():
    model = get_vad_model()
    rng = np.random.default_rng(42)
    audio = (rng.standard_normal(1024) * 0.1).astype(np.float32)
    reference = audio.copy()

    model(audio)

    assert np.array_equal(audio, reference)


def test_vad_model_scores_the_last_window_on_its_real_samples():
    # The probability of the final window must come from the audio itself,
    # not from a zeroed copy of its last samples.
    model = get_vad_model()
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(1024) * 0.1).astype(np.float32)

    probs = model(audio.copy())

    zeroed = audio.copy()
    zeroed[-64:] = 0
    zeroed_probs = model(zeroed)

    assert not np.allclose(probs[-1], zeroed_probs[-1])
    assert np.allclose(probs[:-1], zeroed_probs[:-1])
