import inspect
import os

import numpy as np

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio


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


def test_batch_audio_files(data_dir):
    model = WhisperModel("tiny")
    batched_model = BatchedInferencePipeline(model=model)

    hotwords = os.path.join(data_dir, "hotwords.mp3")
    hotwords_audio = decode_audio(hotwords)

    jfk = os.path.join(data_dir, "jfk.flac")
    jfk_audio = decode_audio(jfk)

    audio_files = [
        hotwords,  # batch 0
        hotwords,
        jfk,  # batch 1
        jfk,
        jfk,  # batch 2
        hotwords,
        hotwords,  # batch 3
    ]

    batch_size = 2
    batch_generator = batched_model.batch_audio_files(
        audio_files, batch_size=batch_size
    )

    for i, (audios, valid_lens) in enumerate(batch_generator):
        if i < 3:
            # batches 0-2 are filled batches
            assert audios.shape[0] == 2

            if i == 0:
                assert audios.shape[-1] == hotwords_audio.shape[-1]

            # in batch 2, we should pad to jfk size since len(jfk) > len(hotwords)
            elif i == 1 or i == 2:
                assert audios.shape[-1] == jfk_audio.shape[-1]

            # make sure we can recover original lengths
            if i == 2:
                assert valid_lens[0] == jfk_audio.shape[-1]
                assert valid_lens[1] == hotwords_audio.shape[-1]

        else:
            # only batch 3 is an unfilled batch
            assert audios.shape[0] == 1
            assert audios.shape[-1] == hotwords_audio.shape[-1]


def test_batched_transcribe_many(jfk_path, physcisworks_path):
    model = WhisperModel("tiny")
    batched_model = BatchedInferencePipeline(model=model)

    physcisworks_audio = decode_audio(physcisworks_path)
    jfk_audio = decode_audio(jfk_path)

    audio_files = [
        physcisworks_path,  # batch 0
        jfk_path,
        jfk_path,  # batch 1
        jfk_path,
        physcisworks_path,  # batch 2
        physcisworks_path,
        physcisworks_path,  # batch 3
    ]

    batch_size = 2
    num_batches = (len(audio_files) // batch_size) + 1

    batch_segments, batch_info = batched_model.transcribe(
        audio_files, batch_size=batch_size, language="en"
    )

    # transcribe returns a list of generators with size equal to number of batches
    # iterate through each batch and then through each generator in that batch
    # to get a flat list of segments (processed in parallel)
    # then recreate hierarchy by stacking chunks for each audio file
    regrouped_segments = []
    total_flat_segments = []
    for batch_idx in range(num_batches):
        info = batch_info[batch_idx]

        flat_segments = []
        for segment in batch_segments[batch_idx]:
            flat_segments.append(segment)
            total_flat_segments.append(segment)

        chunk_idx = 0
        for audio_info in info:
            num_chunks = audio_info.num_chunks
            regrouped_segments.append(flat_segments[chunk_idx : chunk_idx + num_chunks])
            chunk_idx += num_chunks

    num_jfk_files = 3
    num_physics_files = 4
    expected_num_chunks_jfk = 1 * num_jfk_files
    expected_num_chunks_physics = 6 * num_physics_files
    expected_total_chunks = expected_num_chunks_jfk + expected_num_chunks_physics

    assert len(total_flat_segments) == expected_total_chunks
    assert len(regrouped_segments) == len(audio_files)

    for i in range(1, 4):
        # because jfk only has one segment
        assert regrouped_segments[i][0].text == (
            " And so my fellow Americans ask not what your country can do for you, "
            "ask what you can do for your country."
        )

    # TODO: assert result for each other flat segments are identical to non-batched result


def test_batched_transcribe_one(physcisworks_path):
    model = WhisperModel("tiny")
    batched_model = BatchedInferencePipeline(model=model)
    result, info = batched_model.transcribe(physcisworks_path, batch_size=16)
    assert info[0][0].language == "en"
    assert info[0][0].language_probability > 0.7
    segments = []

    for segment in result[0]:
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
    for segment in result[0]:
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
    model = WhisperModel("tiny")
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
