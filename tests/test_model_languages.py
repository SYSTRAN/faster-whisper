from types import SimpleNamespace

import tokenizers

from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer


def test_supported_languages_are_accepted_by_the_model_tokenizer():
    model = WhisperModel("tiny", device="cpu", cpu_threads=1)

    languages = model.supported_languages
    assert len(languages) == 99
    assert "yue" not in languages
    for language in languages:
        Tokenizer(model.hf_tokenizer, True, task="transcribe", language=language)

    languages.remove("en")
    assert "en" in model.supported_languages


def test_supported_languages_follow_available_language_tokens():
    model = WhisperModel.__new__(WhisperModel)
    model.model = SimpleNamespace(is_multilingual=True)
    model.hf_tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"<|en|>": 0, "<|yue|>": 1})
    )

    assert model.supported_languages == ["en", "yue"]


def test_english_only_model_keeps_english_language():
    model = WhisperModel.__new__(WhisperModel)
    model.model = SimpleNamespace(is_multilingual=False)
    # An English-only model does not need language tokens in its vocabulary.
    model.hf_tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({}))

    assert model.supported_languages == ["en"]
