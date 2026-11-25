import numpy as np

from faster_whisper.vad import get_vad_model


def test_vad_model_does_not_modify_input():
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(1024) * 0.1).astype(np.float32)
    original = audio.copy()

    model = get_vad_model()
    first = model(audio)

    np.testing.assert_array_equal(audio, original)

    second = model(audio)
    np.testing.assert_array_equal(first, second)


def test_vad_model_scores_last_window_on_original_samples():
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(1024) * 0.1).astype(np.float32)
    model = get_vad_model()

    # Reference: build the per-window context from a copy of the audio.
    windows = audio.reshape(-1, 512)
    context = windows[:, -64:].copy()
    context[-1] = 0
    context = np.roll(context, 1, 0)
    batched = np.concatenate([context, windows], 1)

    h = np.zeros((1, 1, 128), dtype="float32")
    c = np.zeros((1, 1, 128), dtype="float32")
    expected, _, _ = model.session.run(None, {"input": batched, "h": h, "c": c})

    np.testing.assert_allclose(model(audio.copy()), expected, rtol=1e-6, atol=1e-7)
