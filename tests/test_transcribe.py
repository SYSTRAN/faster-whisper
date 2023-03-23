from faster_whisper import WhisperModel


def test_transcribe(jfk_path):
    model = WhisperModel("tiny")
    segments, info = model.transcribe(jfk_path, word_timestamps=True)

    assert info.language == "en"
    assert info.language_probability > 0.9
    assert info.duration == 11

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
