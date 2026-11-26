import os

import pytest

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def translation_model():
    return WhisperModel("tiny", device="cpu", cpu_threads=1)


@pytest.mark.parametrize("language", ["zh", "ja", "th", "lo", "my", "en", "fr"])
def test_translated_words_use_english_boundaries(translation_model, language):
    tokenizer = Tokenizer(
        translation_model.hf_tokenizer, True, task="translate", language=language
    )
    text = " Translating extraordinarily complicated internationalization."
    tokens = tokenizer.encode(text)
    prompt = tokenizer.sot_sequence

    words, word_tokens = tokenizer.split_to_word_tokens(tokens)

    assert words == [
        " Translating",
        " extraordinarily",
        " complicated",
        " internationalization",
        ".",
    ]
    assert [token for word in word_tokens for token in word] == tokens
    assert tokenizer.sot_sequence == prompt
    assert tokenizer.language_code == language


def test_transcription_keeps_unicode_boundaries(translation_model):
    tokenizer = Tokenizer(
        translation_model.hf_tokenizer, True, task="transcribe", language="zh"
    )
    tokens = tokenizer.encode(" 世界你好。")

    words, word_tokens = tokenizer.split_to_word_tokens(tokens)

    assert words == [" ", "世界", "你好", "。"]
    assert [token for word in word_tokens for token in word] == tokens


@pytest.mark.parametrize("batched", [False, True])
def test_translation_word_timestamps_keep_multitoken_words(
    translation_model, data_dir, batched
):
    pipeline = (
        BatchedInferencePipeline(translation_model) if batched else translation_model
    )
    segments, _ = pipeline.transcribe(
        os.path.join(data_dir, "hotwords.mp3"),
        language="ja",
        task="translate",
        hotwords="ComfyUI",
        word_timestamps=True,
        vad_filter=False,
        beam_size=1,
        temperature=0,
    )
    segments = list(segments)

    assert "ComfyUI" in "".join(segment.text for segment in segments)
    words = [word for segment in segments for word in segment.words]
    assert any(word.word.strip(" .,") == "ComfyUI" for word in words)
    assert all(word.start <= word.end for word in words)
