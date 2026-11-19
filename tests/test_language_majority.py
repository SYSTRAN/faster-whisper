from types import SimpleNamespace

import numpy as np

from faster_whisper import WhisperModel


def test_probabilities_match_the_majority_vote_language():
    window_results = iter(
        [
            [[("<|en|>", 0.40), ("<|fr|>", 0.35), ("<|de|>", 0.25)]],
            [[("<|en|>", 0.45), ("<|de|>", 0.30), ("<|fr|>", 0.25)]],
            [[("<|fr|>", 0.48), ("<|en|>", 0.32), ("<|de|>", 0.20)]],
        ]
    )
    model = object.__new__(WhisperModel)
    model.feature_extractor = SimpleNamespace(nb_max_frames=2)
    model.encode = lambda features: features
    model.model = SimpleNamespace(detect_language=lambda _: next(window_results))

    language, confidence, probabilities = model.detect_language(
        features=np.zeros((80, 6), dtype=np.float32),
        language_detection_segments=3,
    )
    assert (language, confidence) == ("en", 0.45)
    assert probabilities == [("en", 0.45), ("de", 0.30), ("fr", 0.25)]
