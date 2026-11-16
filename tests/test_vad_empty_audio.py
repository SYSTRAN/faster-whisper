from unittest.mock import Mock

import numpy as np

import faster_whisper.vad as vad


def test_empty_audio_does_not_initialize_vad_model(monkeypatch):
    load = Mock(side_effect=AssertionError("empty audio needs no model"))
    monkeypatch.setattr(vad, "get_vad_model", load)
    assert vad.get_speech_timestamps(np.array([], dtype=np.float32)) == []
    load.assert_not_called()


def test_empty_vad_batch_returns_matching_probability_shape():
    model = vad.get_vad_model()
    nonempty = model(np.zeros(512, dtype=np.float32))
    session = model.session
    try:
        model.session = Mock()
        result = model(np.array([], dtype=np.float32))
        assert result.shape == (0, *nonempty.shape[1:])
        assert result.dtype == nonempty.dtype
        model.session.run.assert_not_called()
    finally:
        model.session = session
