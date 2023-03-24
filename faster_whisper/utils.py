from typing import Optional

import huggingface_hub

from tqdm.auto import tqdm


def download_model(
    size: str,
    output_dir: Optional[str] = None,
    show_progress_bars: bool = True,
):
    """Downloads a CTranslate2 Whisper model from the Hugging Face Hub.

    The model is downloaded from https://huggingface.co/guillaumekln.

    Args:
      size: Size of the model to download (tiny, tiny.en, base, base.en, small, small.en,
        medium, medium.en, or large-v2).
      output_dir: Directory where the model should be saved. If not set, the model is saved in
        the standard Hugging Face cache directory.
      show_progress_bars: Show the tqdm progress bars during the download.

    Returns:
      The path to the downloaded model.
    """
    repo_id = "guillaumekln/faster-whisper-%s" % size
    kwargs = {}

    if output_dir is not None:
        kwargs["local_dir"] = output_dir
        kwargs["local_dir_use_symlinks"] = False

    if not show_progress_bars:
        kwargs["tqdm_class"] = disabled_tqdm

    return huggingface_hub.snapshot_download(repo_id, **kwargs)


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
