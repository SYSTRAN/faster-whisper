from faster_whisper import supported_languages


def test_supported_languages():
    languages = supported_languages()
    assert isinstance(languages, list)
    assert "fr" in languages
