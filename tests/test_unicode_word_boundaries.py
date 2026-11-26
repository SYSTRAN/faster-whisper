import numpy as np
import pytest

from faster_whisper import WhisperModel, decode_audio
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def unicode_model():
    return WhisperModel("tiny", device="cpu", cpu_threads=1)


@pytest.fixture
def tokenizer(unicode_model):
    return Tokenizer(unicode_model.hf_tokenizer, True, task="transcribe", language="en")


@pytest.mark.parametrize(
    "context", ["bare", "suffix", "prefix", "timestamp", "special"]
)
def test_replacement_boundaries_preserve_text_and_every_token(tokenizer, context):
    # The continuation byte in 94 changes how 1041's final invalid sequence is
    # decoded, even though both [1041] and [1041, 94] decode to two replacements.
    tokens = [1041, 94]
    if context != "bare":
        tokens += tokenizer.encode(" Hello world!")
    if context == "prefix":
        tokens = tokenizer.encode(" Before") + tokens
    elif context == "timestamp":
        tokens.insert(2, tokenizer.timestamp_begin + 50)
    elif context == "special":
        tokens.insert(1, tokenizer.eot)

    words, word_tokens = tokenizer.split_tokens_on_unicode(tokens)

    assert "".join(words) == tokenizer.decode_with_timestamps(tokens)
    assert [token for word in word_tokens for token in word] == tokens


@pytest.mark.parametrize(
    "text", [" And so my fellow Americans.", "é🦄你好", " A literal � followed by words."]
)
def test_valid_unicode_and_literal_replacements_are_preserved(tokenizer, text):
    tokens = tokenizer.encode(text)

    words, word_tokens = tokenizer.split_tokens_on_unicode(tokens)

    assert "".join(words) == text
    assert [token for word in word_tokens for token in word] == tokens


def test_standalone_replacement_keeps_existing_word_boundary(tokenizer):
    assert tokenizer.split_tokens_on_unicode([246, 526]) == (
        ["�", "é"],
        [[246], [526]],
    )


@pytest.mark.parametrize("tokens", [[94], [1041]])
def test_incomplete_utf8_at_end_is_retained(tokenizer, tokens):
    words, word_tokens = tokenizer.split_tokens_on_unicode(tokens)

    assert "".join(words) == tokenizer.decode_with_timestamps(tokens)
    assert [token for word in word_tokens for token in word] == tokens


@pytest.mark.parametrize("text", [" Ordinary English words.", "你好世界。🦄é"])
def test_well_formed_text_does_not_require_extra_decoding(tokenizer, monkeypatch, text):
    tokens = tokenizer.encode(text)
    original_decode = tokenizer.decode_with_timestamps
    calls = []

    def record_decode(value):
        calls.append(value)
        return original_decode(value)

    monkeypatch.setattr(tokenizer, "decode_with_timestamps", record_decode)
    words, _ = tokenizer.split_tokens_on_unicode(tokens)

    assert "".join(words) == text
    assert len(calls) == len(tokens) + 1


def test_native_alignment_retains_words_after_invalid_utf8(
    unicode_model, tokenizer, jfk_path
):
    tokens = [1041, 94] + tokenizer.encode(" And so my fellow Americans.")
    features = unicode_model.feature_extractor(decode_audio(jfk_path))
    encoder_output = unicode_model.encode(pad_or_trim(features))

    alignment = unicode_model.find_alignment(
        tokenizer, [tokens], encoder_output, features.shape[-1] - 1
    )[0]

    assert "".join(word["word"] for word in alignment) == tokenizer.decode(tokens)
    assert [token for word in alignment for token in word["tokens"]] == tokens
    assert all(np.isfinite(word["probability"]) for word in alignment)
    assert all(word["start"] <= word["end"] for word in alignment)
