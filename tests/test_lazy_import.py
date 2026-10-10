import subprocess
import sys


def _modules_after(import_line):
    """Import in a subprocess and report which heavy modules got loaded."""
    watched = ("ctranslate2", "transformers", "faster_whisper.transcribe")
    code = (
        "import sys\n"
        f"{import_line}\n"
        f"print(','.join(m for m in {watched!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return [m for m in result.stdout.strip().split(",") if m]


def test_vad_import_does_not_load_the_transcription_stack():
    """faster_whisper.vad needs numpy, one utils helper and onnxruntime.

    Python runs faster_whisper/__init__.py before any submodule, so an eager
    import of transcribe there is paid by every vad-only consumer.
    """
    assert _modules_after("from faster_whisper.vad import get_vad_model") == []


def test_package_import_does_not_load_the_transcription_stack():
    assert _modules_after("import faster_whisper") == []


def test_transcription_stack_loads_on_first_use():
    assert _modules_after("from faster_whisper import WhisperModel") == [
        "ctranslate2",
        "faster_whisper.transcribe",
    ]


def test_public_attributes_are_reachable():
    import faster_whisper

    assert faster_whisper.WhisperModel.__name__ == "WhisperModel"
    assert (
        faster_whisper.BatchedInferencePipeline.__name__ == "BatchedInferencePipeline"
    )
    assert "WhisperModel" in dir(faster_whisper)


def test_unknown_attribute_still_raises():
    import faster_whisper

    try:
        faster_whisper.does_not_exist
    except AttributeError as exc:
        assert "does_not_exist" in str(exc)
    else:
        raise AssertionError("expected AttributeError")
