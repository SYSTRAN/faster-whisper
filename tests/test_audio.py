import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from faster_whisper import WhisperModel, decode_audio


class CustomPathLike:
    """A custom class that implements the os.PathLike protocol."""

    def __init__(self, path: str):
        self._path = path

    def __fspath__(self) -> str:
        return self._path


@patch("faster_whisper.audio._resample_frames")
@patch("faster_whisper.audio._group_frames")
@patch("faster_whisper.audio._ignore_invalid_frames")
@patch("faster_whisper.audio.av.open")
def test_decode_audio_pathlib_mock(
    mock_av_open,
    mock_ignore,
    mock_group,
    mock_resample,
):
    """Verify that decode_audio accepts a pathlib.Path object, converts it to str,
    and decodes audio without crashing using mocks.
    """
    fake_path = Path("fake_dir") / "fake_audio.mp3"

    mock_frame = MagicMock()
    mock_frame.to_ndarray.return_value = np.zeros(1600, dtype=np.int16)

    mock_container = MagicMock()
    mock_av_open.return_value.__enter__.return_value = mock_container
    mock_container.decode.return_value = [mock_frame]
    mock_ignore.return_value = [mock_frame]
    mock_group.return_value = [mock_frame]
    mock_resample.return_value = [mock_frame]

    result = decode_audio(fake_path)

    # Verify av.open was called with a str, not a Path object
    mock_av_open.assert_called_once()
    opened_arg = mock_av_open.call_args[0][0]
    assert isinstance(opened_arg, str)
    assert opened_arg == str(os.fspath(fake_path))

    # Verify output is a float32 numpy array
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 1600


@patch("faster_whisper.audio._resample_frames")
@patch("faster_whisper.audio._group_frames")
@patch("faster_whisper.audio._ignore_invalid_frames")
@patch("faster_whisper.audio.av.open")
def test_decode_audio_custom_pathlike_mock(
    mock_av_open,
    mock_ignore,
    mock_group,
    mock_resample,
):
    """Verify that decode_audio accepts any object implementing the
    os.PathLike protocol.
    """
    fake_path = CustomPathLike("custom_dir/custom_audio.wav")

    mock_frame = MagicMock()
    mock_frame.to_ndarray.return_value = np.zeros(800, dtype=np.int16)

    mock_container = MagicMock()
    mock_av_open.return_value.__enter__.return_value = mock_container
    mock_container.decode.return_value = [mock_frame]
    mock_ignore.return_value = [mock_frame]
    mock_group.return_value = [mock_frame]
    mock_resample.return_value = [mock_frame]

    result = decode_audio(fake_path)

    mock_av_open.assert_called_once()
    opened_arg = mock_av_open.call_args[0][0]
    assert isinstance(opened_arg, str)
    assert opened_arg == "custom_dir/custom_audio.wav"
    assert isinstance(result, np.ndarray)


def test_decode_audio_pathlib_real_file(jfk_path):
    """Verify that decode_audio runs end-to-end with an actual pathlib.Path file."""
    path_obj = Path(jfk_path)
    assert isinstance(path_obj, Path)

    audio = decode_audio(path_obj)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0


def test_whisper_model_pathlib():
    """Verify that WhisperModel accepts pathlib.Path for model_size_or_path
    and download_root.
    """
    with patch("faster_whisper.transcribe.download_model") as mock_dl, patch(
        "faster_whisper.transcribe.ctranslate2.models.Whisper"
    ), patch("faster_whisper.transcribe.tokenizers.Tokenizer"):

        mock_dl.return_value = "mock_model_dir"
        WhisperModel(Path("tiny"), download_root=Path("cache_dir"))

        mock_dl.assert_called_once()
        assert isinstance(mock_dl.call_args[0][0], str)
        assert mock_dl.call_args[0][0] == "tiny"
        assert isinstance(mock_dl.call_args[1]["cache_dir"], str)
        assert mock_dl.call_args[1]["cache_dir"] == "cache_dir"

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            "faster_whisper.transcribe.ctranslate2.models.Whisper"
        ) as mock_ct, patch("faster_whisper.transcribe.tokenizers.Tokenizer"):

            WhisperModel(Path(tmpdir))

            mock_ct.assert_called_once()
            called_path = mock_ct.call_args[0][0]
            assert isinstance(called_path, str)
            assert called_path == tmpdir
