from faster_whisper.audio import decode_audio
from faster_whisper.utils import available_models, download_model, format_timestamp
from faster_whisper.version import __version__

__all__ = [
    "available_models",
    "decode_audio",
    "WhisperModel",
    "BatchedInferencePipeline",
    "download_model",
    "format_timestamp",
    "__version__",
]

_LAZY_ATTRIBUTES = frozenset(("BatchedInferencePipeline", "WhisperModel"))


def __getattr__(name):
    """Import the transcription stack on first use (PEP 562).

    Python runs this file before any submodule, so importing faster_whisper.vad
    used to pull in faster_whisper.transcribe and ctranslate2 with it, none of
    which the VAD needs.
    """
    if name in _LAZY_ATTRIBUTES:
        from faster_whisper import transcribe

        return getattr(transcribe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
