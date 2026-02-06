import numpy as np

import faster_whisper.vad as vad


def test_get_speech_timestamps_prefers_latest_silence_on_tie(monkeypatch):
    # Two equal-length silence candidates appear before max_speech is exceeded.
    speech_probs = np.array(
        [0.9] * 90 + [0.05] * 13 + [0.9] * 20 + [0.05] * 13 + [0.9] * 100,
        dtype=np.float32,
    )
    monkeypatch.setattr(vad, "get_vad_model", lambda: (lambda audio: speech_probs))

    audio = np.zeros(len(speech_probs) * 512, dtype=np.float32)
    speeches = vad.get_speech_timestamps(
        audio,
        vad_options=vad.VadOptions(
            threshold=0.5,
            neg_threshold=0.35,
            min_speech_duration_ms=0,
            max_speech_duration_s=4.6,
            min_silence_duration_ms=2000,
            speech_pad_ms=0,
            min_silence_at_max_speech=98,
            use_max_poss_sil_at_max_speech=True,
        ),
        sampling_rate=16000,
    )

    # Tie on silence length should pick the most recent candidate.
    assert speeches[0]["end"] == (90 + 13 + 20) * 512
