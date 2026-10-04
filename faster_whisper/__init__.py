import importlib

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

# Resolved lazily via module __getattr__ (PEP 562) so that importing a narrow
# submodule such as `faster_whisper.vad` does not pull in the transcription
# stack (ctranslate2, tokenizers, PyAV) through this package __init__.
_LAZY_ATTRIBUTE_MODULES = {
    "available_models": "faster_whisper.utils",
    "decode_audio": "faster_whisper.audio",
    "WhisperModel": "faster_whisper.transcribe",
    "BatchedInferencePipeline": "faster_whisper.transcribe",
    "download_model": "faster_whisper.utils",
    "format_timestamp": "faster_whisper.utils",
}


def __getattr__(name: str):
    module_path = _LAZY_ATTRIBUTE_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attribute = getattr(importlib.import_module(module_path), name)
    # Cache on the module so subsequent lookups skip the import machinery.
    globals()[name] = attribute
    return attribute


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
