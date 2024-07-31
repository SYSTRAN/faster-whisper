import os

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_suppressed_tokens


def test_supported_languages():
    model = WhisperModel("tiny.en")
    assert model.supported_languages == ["en"]


def test_transcribe(jfk_path):
    model = WhisperModel("tiny")
    segments, info = model.transcribe(jfk_path, word_timestamps=True)
    assert info.all_language_probs is not None

    assert info.language == "en"
    assert info.language_probability > 0.9
    assert info.duration == 11

    # Get top language info from all results, which should match the
    # already existing metadata
    top_lang, top_lang_score = info.all_language_probs[0]
    assert info.language == top_lang
    assert abs(info.language_probability - top_lang_score) < 1e-16

    segments = list(segments)

    assert len(segments) == 1

    segment = segments[0]

    assert segment.text == (
        " And so my fellow Americans ask not what your country can do for you, "
        "ask what you can do for your country."
    )

    assert segment.text == "".join(word.word for word in segment.words)
    assert segment.start == segment.words[0].start
    assert segment.end == segment.words[-1].end
    batched_model = BatchedInferencePipeline(model=model)
    result, info = batched_model.transcribe(
        jfk_path, word_timestamps=True, vad_filter=False
    )
    assert info.language == "en"
    assert info.language_probability > 0.7
    segments = []
    for segment in result:
        segments.append(
            {"start": segment.start, "end": segment.end, "text": segment.text}
        )

    assert len(segments) == 1
    assert segment.text == (
        " And so my fellow Americans ask not what your country can do for you, "
        "ask what you can do for your country."
    )


def test_batched_transcribe(physcisworks_path):
    model = WhisperModel("tiny")
    batched_model = BatchedInferencePipeline(model=model)
    result, info = batched_model.transcribe(physcisworks_path, batch_size=16)
    assert info.language == "en"
    assert info.language_probability > 0.7
    segments = []
    for segment in result:
        segments.append(
            {"start": segment.start, "end": segment.end, "text": segment.text}
        )
    # number of near 30 sec segments
    assert len(segments) == 7

    result, info = batched_model.transcribe(
        physcisworks_path,
        batch_size=16,
        without_timestamps=False,
        word_timestamps=True,
    )
    segments = []
    for segment in result:
        assert segment.words is not None
        segments.append(
            {"start": segment.start, "end": segment.end, "text": segment.text}
        )
    assert len(segments) > 7


def test_prefix_with_timestamps(jfk_path):
    model = WhisperModel("tiny")
    segments, _ = model.transcribe(jfk_path, prefix="And so my fellow Americans")
    segments = list(segments)

    assert len(segments) == 1

    segment = segments[0]

    assert segment.text == (
        " And so my fellow Americans ask not what your country can do for you, "
        "ask what you can do for your country."
    )

    assert segment.start == 0
    assert 10 < segment.end < 11


def test_vad(jfk_path):
    model = WhisperModel("tiny")
    segments, info = model.transcribe(
        jfk_path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
    )
    segments = list(segments)

    assert len(segments) == 1
    segment = segments[0]

    assert segment.text == (
        " And so my fellow Americans ask not what your country can do for you, "
        "ask what you can do for your country."
    )

    assert 0 < segment.start < 1
    assert 10 < segment.end < 11

    assert info.vad_options.min_silence_duration_ms == 500
    assert info.vad_options.speech_pad_ms == 200


def test_stereo_diarization(data_dir):
    model = WhisperModel("tiny")

    audio_path = os.path.join(data_dir, "stereo_diarization.wav")
    left, right = decode_audio(audio_path, split_stereo=True)

    segments, _ = model.transcribe(left)
    transcription = "".join(segment.text for segment in segments).strip()
    assert transcription == (
        "He began a confused complaint against the wizard, "
        "who had vanished behind the curtain on the left."
    )

    segments, _ = model.transcribe(right)
    transcription = "".join(segment.text for segment in segments).strip()
    assert transcription == "The horizon seems extremely distant."


def test_multisegment_lang_id(physcisworks_path):
    model = WhisperModel("tiny")
    language_info = model.detect_language_multi_segment(physcisworks_path)
    assert language_info["language_code"] == "en"
    assert language_info["language_confidence"] > 0.8


def test_suppressed_tokens_minus_1():
    model = WhisperModel("tiny.en")

    tokenizer = Tokenizer(model.hf_tokenizer, False)
    tokens = get_suppressed_tokens(tokenizer, [-1])
    assert tokens == (
        1,
        2,
        7,
        8,
        9,
        10,
        14,
        25,
        26,
        27,
        28,
        29,
        31,
        58,
        59,
        60,
        61,
        62,
        63,
        90,
        91,
        92,
        93,
        357,
        366,
        438,
        532,
        685,
        705,
        796,
        930,
        1058,
        1220,
        1267,
        1279,
        1303,
        1343,
        1377,
        1391,
        1635,
        1782,
        1875,
        2162,
        2361,
        2488,
        3467,
        4008,
        4211,
        4600,
        4808,
        5299,
        5855,
        6329,
        7203,
        9609,
        9959,
        10563,
        10786,
        11420,
        11709,
        11907,
        13163,
        13697,
        13700,
        14808,
        15306,
        16410,
        16791,
        17992,
        19203,
        19510,
        20724,
        22305,
        22935,
        27007,
        30109,
        30420,
        33409,
        34949,
        40283,
        40493,
        40549,
        47282,
        49146,
        50257,
        50357,
        50358,
        50359,
        50360,
    )


def test_suppressed_tokens_minus_value():
    model = WhisperModel("tiny.en")

    tokenizer = Tokenizer(model.hf_tokenizer, False)
    tokens = get_suppressed_tokens(tokenizer, [13])
    assert tokens == (13, 50257, 50357, 50358, 50359, 50360)
