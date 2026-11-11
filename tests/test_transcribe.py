import inspect
import logging
import os

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import tokenizers

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.vad import VadOptions


@pytest.fixture
def offline_model():
    # Replace only CT2 inference; tokenization, features and segmentation stay real.
    model = WhisperModel.__new__(WhisperModel)
    vocabulary = [
        "speech",
        "[UNK]",
        "<|endoftext|>",
        "<|startoftranscript|>",
        "<|startofprev|>",
        "<|startoflm|>",
        "<|transcribe|>",
        "<|translate|>",
        "<|en|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]
    model.hf_tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"
        )
    )
    model.hf_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    model.feature_extractor = FeatureExtractor()
    initial_state = deepcopy(vars(model.feature_extractor))
    model.logger = logging.getLogger(__name__)
    model.input_stride = 2
    model.num_samples_per_token = 320
    model.frames_per_second = 100
    model.tokens_per_second = 50
    model.time_precision = 0.02
    model.max_length = 448
    model.model = Mock(is_multilingual=True, device="cpu", n_mels=80)

    def encode(features, to_cpu=False):
        # Also catch temporary shared-state changes during a lazy iteration.
        np.testing.assert_equal(vars(model.feature_extractor), initial_state)
        return np.asarray(features).copy()

    model.model.encode.side_effect = encode
    model.model.generate.side_effect = lambda encoder_output, prompts, **kwargs: [
        SimpleNamespace(sequences_ids=[[0]], scores=[-0.1], no_speech_prob=0.0)
        for _ in prompts
    ]
    model.model.detect_language.return_value = [[("<|en|>", 0.9), ("<|de|>", 0.1)]]
    return model


def _offline_encoded_frames(model):
    # Silence extracts to -1.5, while pad_or_trim fills the unused frames with 0.
    return [
        np.count_nonzero(np.any(np.asarray(call.args[0]) != 0, axis=(0, 1)))
        for call in model.model.encode.call_args_list
    ]


@pytest.mark.parametrize("chunk_length", [10, 30])
@pytest.mark.parametrize("parameter_kind", ["none", "dict", "instance"])
def test_offline_batched_vad_options_and_long_chunks(
    offline_model, monkeypatch, chunk_length, parameter_kind
):
    custom_options = dict(
        threshold=0.37,
        min_silence_duration_ms=310,
        speech_pad_ms=120,
        max_speech_duration_s=120,
    )
    parameters = {
        "none": None,
        "dict": custom_options,
        "instance": VadOptions(**custom_options),
    }[parameter_kind]
    original = deepcopy(parameters)
    observed = []

    def speech_timestamps(audio, options):
        observed.append(options)
        # Simulate uninterrupted speech, splitting only at the supplied VAD limit.
        limit = int(options.max_speech_duration_s * 16000)
        return [
            {"start": start, "end": min(start + limit, len(audio))}
            for start in range(0, len(audio), limit)
        ]

    monkeypatch.setattr(
        "faster_whisper.transcribe.get_speech_timestamps", speech_timestamps
    )
    pipeline = BatchedInferencePipeline(offline_model)
    segments, info = pipeline.transcribe(
        np.zeros(65 * 16000, dtype=np.float32),
        language="en",
        vad_parameters=parameters,
        chunk_length=chunk_length,
        suppress_tokens=[],
    )
    expected = asdict(
        VadOptions(min_silence_duration_ms=160)
        if parameters is None
        else VadOptions(**custom_options)
    )
    expected["max_speech_duration_s"] = chunk_length
    assert parameters == original
    assert len(observed) == 1
    assert isinstance(info.vad_options, VadOptions)
    assert info.vad_options is observed[0]
    assert info.vad_options is not parameters
    assert asdict(info.vad_options) == expected

    segments = list(segments)
    assert [(s.start, s.end) for s in segments] == [
        (start, min(start + chunk_length, 65)) for start in range(0, 65, chunk_length)
    ]
    assert all(s.end - s.start <= chunk_length for s in segments)
    assert info.duration_after_vad == info.duration == 65


@pytest.mark.parametrize("schedule", ["sequential", "delayed", "interleaved"])
def test_offline_transcribe_windows_are_call_local(offline_model, schedule):
    model = offline_model
    audio = np.zeros(35 * 16000, dtype=np.float32)
    initial_state = deepcopy(vars(model.feature_extractor))

    def transcribe(chunk_length=None):
        kwargs = {} if chunk_length is None else {"chunk_length": chunk_length}
        segments, info = model.transcribe(
            audio, without_timestamps=True, suppress_tokens=[], **kwargs
        )
        np.testing.assert_equal(vars(model.feature_extractor), initial_state)
        assert info.language == "en"
        # Automatic language detection must receive the same per-call override.
        assert _offline_encoded_frames(model)[-1] == (chunk_length or 30) * 100
        return segments

    if schedule == "sequential":
        short = list(transcribe(10))
        default = list(transcribe())
    elif schedule == "delayed":
        short_generator = transcribe(10)
        default_generator = transcribe()
        default = list(default_generator)
        short = list(short_generator)
    else:
        default_generator = transcribe()
        default = [next(default_generator)]
        short_generator = transcribe(10)
        short = [next(short_generator)]
        default.extend(default_generator)
        short.extend(short_generator)

    assert [(s.start, s.end) for s in short] == [(0, 10), (10, 20), (20, 30), (30, 35)]
    assert [(s.start, s.end) for s in default] == [(0, 30), (30, 35)]
    np.testing.assert_equal(vars(model.feature_extractor), initial_state)


@pytest.mark.parametrize("source", ["audio", "features"])
def test_offline_detect_language_window_override(offline_model, monkeypatch, source):
    model = offline_model
    audio = np.zeros(65 * 16000, dtype=np.float32)
    initial_state = deepcopy(vars(model.feature_extractor))
    inputs = {source: audio if source == "audio" else model.feature_extractor(audio)}
    extracted_samples = []
    extract = FeatureExtractor.__call__

    def record_extraction(self, waveform, *args, **kwargs):
        extracted_samples.append(len(waveform))
        return extract(self, waveform, *args, **kwargs)

    monkeypatch.setattr(FeatureExtractor, "__call__", record_extraction)
    for kwargs, expected_frames in [({"chunk_length": 10}, 1000), ({}, 3000)]:
        model.model.encode.reset_mock()
        language, probability, languages = model.detect_language(
            **inputs,
            language_detection_segments=2,
            language_detection_threshold=1.0,
            **kwargs,
        )
        assert (language, probability) == ("en", 0.9)
        assert languages == [("en", 0.9), ("de", 0.1)]
        assert _offline_encoded_frames(model) == [expected_frames] * 2
        np.testing.assert_equal(vars(model.feature_extractor), initial_state)
    assert extracted_samples == ([20 * 16000, 60 * 16000] if source == "audio" else [])


@pytest.mark.parametrize("failure", ["detect_language", "generate"])
def test_offline_transcribe_exception_preserves_window(offline_model, failure):
    model = offline_model
    audio = np.zeros(35 * 16000, dtype=np.float32)
    initial_state = deepcopy(vars(model.feature_extractor))
    operation = getattr(model.model, failure)
    original_side_effect = operation.side_effect
    operation.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        segments, _ = model.transcribe(audio, chunk_length=10, suppress_tokens=[])
        list(segments)
    np.testing.assert_equal(vars(model.feature_extractor), initial_state)

    operation.side_effect = original_side_effect
    segments, _ = model.transcribe(audio, suppress_tokens=[])
    assert [(s.start, s.end) for s in segments] == [(0, 30), (30, 35)]
    np.testing.assert_equal(vars(model.feature_extractor), initial_state)


def test_offline_detect_language_exception_preserves_window(offline_model):
    model = offline_model
    audio = np.zeros(35 * 16000, dtype=np.float32)
    initial_state = deepcopy(vars(model.feature_extractor))
    model.model.detect_language.side_effect = RuntimeError("detection failed")
    with pytest.raises(RuntimeError, match="detection failed"):
        model.detect_language(audio, chunk_length=10)
    np.testing.assert_equal(vars(model.feature_extractor), initial_state)

    model.model.detect_language.side_effect = None
    model.model.encode.reset_mock()
    assert model.detect_language(audio)[:2] == ("en", 0.9)
    assert _offline_encoded_frames(model) == [3000]


def test_offline_feature_extractor_has_no_per_call_window():
    assert "chunk_length" not in inspect.signature(FeatureExtractor.__call__).parameters
    extractor = FeatureExtractor()
    initial_state = deepcopy(vars(extractor))
    for seconds in (10, 30):
        features = extractor(np.zeros(seconds * 16000, dtype=np.float32))
        assert features.shape == (80, seconds * 100 + 1)
        np.testing.assert_equal(vars(extractor), initial_state)


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
        " And so my fellow Americans, ask not what your country can do for you, "
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
    assert len(segments) == 6

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


def test_empty_audio():
    audio = np.asarray([], dtype="float32")
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model=model)
    assert list(model.transcribe(audio)[0]) == []
    assert list(pipeline.transcribe(audio)[0]) == []
    model.detect_language(audio)


def test_prefix_with_timestamps(jfk_path):
    model = WhisperModel("tiny")
    segments, _ = model.transcribe(jfk_path, prefix="And so my fellow Americans")
    segments = list(segments)

    assert len(segments) == 1

    segment = segments[0]

    assert segment.text == (
        " And so my fellow Americans, ask not what your country can do for you, "
        "ask what you can do for your country."
    )

    assert segment.start == 0
    assert 10 < segment.end <= 11


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


def test_multilingual_transcription(data_dir):
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model)

    audio_path = os.path.join(data_dir, "multilingual.mp3")
    audio = decode_audio(audio_path)

    segments, info = model.transcribe(
        audio,
        multilingual=True,
        without_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = list(segments)

    assert (
        segments[0].text
        == " Permission is hereby granted, free of charge, to any person obtaining a copy of the"
        " software and associated documentation files to deal in the software without restriction,"
        " including without limitation the rights to use, copy, modify, merge, publish, distribute"
        ", sublicence, and or cell copies of the software, and to permit persons to whom the "
        "software is furnished to do so, subject to the following conditions. The above copyright"
        " notice and this permission notice, shall be included in all copies or substantial "
        "portions of the software."
    )

    assert (
        segments[1].text
        == " Jedem, der dieses Software und die dazu gehöregen Dokumentationsdatein erhält, wird "
        "hiermit unengeltlich die Genehmigung erteilt, wird der Software und eingeschränkt zu "
        "verfahren. Dies umfasst insbesondere das Recht, die Software zu verwenden, zu "
        "vervielfältigen, zu modifizieren, zu Samenzofügen, zu veröffentlichen, zu verteilen, "
        "unterzulizenzieren und oder kopieren der Software zu verkaufen und diese Rechte "
        "unterfolgen den Bedingungen anderen zu übertragen."
    )

    segments, info = pipeline.transcribe(audio, multilingual=True)
    segments = list(segments)

    assert (
        segments[0].text
        == " Permission is hereby granted, free of charge, to any person obtaining a copy of the"
        " software and associated documentation files to deal in the software without restriction,"
        " including without limitation the rights to use, copy, modify, merge, publish, distribute"
        ", sublicence, and or cell copies of the software, and to permit persons to whom the "
        "software is furnished to do so, subject to the following conditions. The above copyright"
        " notice and this permission notice, shall be included in all copies or substantial "
        "portions of the software."
    )
    assert (
        "Dokumentationsdatein erhält, wird hiermit unengeltlich die Genehmigung erteilt,"
        " wird der Software und eingeschränkt zu verfahren. Dies umfasst insbesondere das Recht,"
        " die Software zu verwenden, zu vervielfältigen, zu modifizieren"
        in segments[1].text
    )


def test_hotwords(data_dir):
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model)

    audio_path = os.path.join(data_dir, "hotwords.mp3")
    audio = decode_audio(audio_path)

    segments, info = model.transcribe(audio, hotwords="ComfyUI")
    segments = list(segments)

    assert "ComfyUI" in segments[0].text
    assert info.transcription_options.hotwords == "ComfyUI"

    segments, info = pipeline.transcribe(audio, hotwords="ComfyUI")
    segments = list(segments)

    assert "ComfyUI" in segments[0].text
    assert info.transcription_options.hotwords == "ComfyUI"


def test_transcribe_signature():
    model_transcribe_args = set(inspect.getargs(WhisperModel.transcribe.__code__).args)
    pipeline_transcribe_args = set(
        inspect.getargs(BatchedInferencePipeline.transcribe.__code__).args
    )
    pipeline_transcribe_args.remove("batch_size")

    assert model_transcribe_args == pipeline_transcribe_args


def test_monotonic_timestamps(physcisworks_path):
    model = WhisperModel("base")
    pipeline = BatchedInferencePipeline(model=model)

    segments, info = model.transcribe(physcisworks_path, word_timestamps=True)
    segments = list(segments)

    for i in range(len(segments) - 1):
        assert segments[i].start <= segments[i].end
        assert segments[i].end <= segments[i + 1].start
        for word in segments[i].words:
            assert word.start <= word.end
            assert word.end <= segments[i].end
    assert segments[-1].end <= info.duration

    segments, info = pipeline.transcribe(physcisworks_path, word_timestamps=True)
    segments = list(segments)

    for i in range(len(segments) - 1):
        assert segments[i].start <= segments[i].end
        assert segments[i].end <= segments[i + 1].start
        for word in segments[i].words:
            assert word.start <= word.end
            assert word.end <= segments[i].end
    assert segments[-1].end <= info.duration


def test_cliptimestamps_segments(jfk_path):
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model=model)

    audio = decode_audio(jfk_path)
    audio = np.concatenate([audio, audio])
    clip_timestamps = [{"start": 0.0, "end": 11.0}, {"start": 11.0, "end": 22.0}]

    segments, info = pipeline.transcribe(audio, clip_timestamps=clip_timestamps)
    segments = list(segments)

    assert len(segments) == 2
    for segment, clip in zip(segments, clip_timestamps):
        assert segment.start == clip["start"]
        assert segment.end == clip["end"]
        assert segment.text == (
            " And so my fellow Americans ask not what your country can do for you, "
            "ask what you can do for your country."
        )


def test_cliptimestamps_timings(physcisworks_path):
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model=model)

    audio = decode_audio(physcisworks_path)
    clip_timestamps = [{"start": 0.0, "end": 5.0}, {"start": 6.0, "end": 15.0}]
    transcripts = [
        " Now I want to return to the conservation of mechanical energy.",
        (
            " I have here a pendulum. I have an object that weighs 15 kilograms"
            " and I can lift it up one meter, which I have done now."
        ),
    ]
    segments, info = pipeline.transcribe(audio, clip_timestamps=clip_timestamps)
    segments = list(segments)

    assert len(segments) == 2
    for segment, clip, transcript in zip(segments, clip_timestamps, transcripts):
        assert clip["start"] == segment.start
        assert clip["end"] == segment.end
        assert segment.text == transcript
