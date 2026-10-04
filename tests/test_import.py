"""Import-level regression tests, run in subprocesses so that module state from
other tests in the session cannot mask what an import actually pulls in."""

import subprocess
import sys


def _run_snippet(snippet: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_vad_import_avoids_transcription_stack():
    """`import faster_whisper.vad` must not pull in the transcription stack.

    VAD only needs numpy plus the lazily imported onnxruntime; loading
    ctranslate2, PyAV or tokenizers for it is pure overhead (#1479).
    """
    result = _run_snippet(
        "import sys\n"
        "import faster_whisper.vad\n"
        "heavy = [m for m in ('ctranslate2', 'av', 'tokenizers') if m in sys.modules]\n"
        "print('loaded:', heavy)\n"
        "assert not heavy, f'unexpected modules imported: {heavy}'\n"
    )
    assert result.returncode == 0, result.stderr


def test_public_api_resolves_lazily():
    """The public API keeps working through the lazy module __getattr__."""
    result = _run_snippet(
        "import faster_whisper\n"
        "from faster_whisper import (\n"
        "    BatchedInferencePipeline,\n"
        "    WhisperModel,\n"
        "    available_models,\n"
        "    decode_audio,\n"
        "    download_model,\n"
        "    format_timestamp,\n"
        "    __version__,\n"
        ")\n"
        "assert callable(WhisperModel) and callable(BatchedInferencePipeline)\n"
        "assert callable(available_models) and callable(download_model)\n"
        "assert callable(decode_audio) and callable(format_timestamp)\n"
        "assert isinstance(__version__, str)\n"
        "import sys\n"
        "assert 'faster_whisper.transcribe' in sys.modules\n"
        "assert faster_whisper.WhisperModel is WhisperModel\n"
    )
    assert result.returncode == 0, result.stderr


def test_lazy_attributes_are_cached():
    """Repeated access returns the same resolved object."""
    result = _run_snippet(
        "import faster_whisper\n"
        "assert faster_whisper.decode_audio is faster_whisper.decode_audio\n"
    )
    assert result.returncode == 0, result.stderr


def test_unknown_attribute_raises():
    result = _run_snippet(
        "import faster_whisper\n"
        "try:\n"
        "    faster_whisper.no_such_attribute\n"
        "except AttributeError as e:\n"
        "    assert 'no_such_attribute' in str(e)\n"
        "else:\n"
        "    raise AssertionError('AttributeError not raised')\n"
    )
    assert result.returncode == 0, result.stderr


def test_dir_lists_public_api():
    result = _run_snippet(
        "import faster_whisper\n"
        "names = dir(faster_whisper)\n"
        "assert 'WhisperModel' in names and 'decode_audio' in names\n"
    )
    assert result.returncode == 0, result.stderr
