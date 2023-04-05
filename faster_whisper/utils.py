import logging
import os

from typing import Optional

import huggingface_hub

from tqdm.auto import tqdm

_MODELS = (
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
)


def get_assets_path():
    """Returns the path to the assets directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_logger():
    """Returns the module logger."""
    return logging.getLogger("faster_whisper")


def download_model(size: str, output_dir: Optional[str] = None):
    """Downloads a CTranslate2 Whisper model from the Hugging Face Hub.

    The model is downloaded from https://huggingface.co/guillaumekln.

    Args:
      size: Size of the model to download (tiny, tiny.en, base, base.en, small, small.en,
        medium, medium.en, large-v1, or large-v2).
      output_dir: Directory where the model should be saved. If not set, the model is saved in
        the standard Hugging Face cache directory.

    Returns:
      The path to the downloaded model.

    Raises:
      ValueError: if the model size is invalid.
    """
    if size not in _MODELS:
        raise ValueError(
            "Invalid model size '%s', expected one of: %s" % (size, ", ".join(_MODELS))
        )

    repo_id = "guillaumekln/faster-whisper-%s" % size
    kwargs = {}

    if output_dir is not None:
        kwargs["local_dir"] = output_dir
        kwargs["local_dir_use_symlinks"] = False

    allow_patterns = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.txt",
    ]

    return huggingface_hub.snapshot_download(
        repo_id,
        allow_patterns=allow_patterns,
        tqdm_class=disabled_tqdm,
        **kwargs,
    )


def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


class disabled_tqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
