"""Audio decoding with pluggable backends.

By default, audio is decoded via the Python library `PyAV`
(https://github.com/PyAV-Org/PyAV), which bundles FFmpeg so no system
dependency is required.

An alternative `ffmpeg` backend is also available. It shells out to the
`ffmpeg` binary via `subprocess`. This is useful on environments where the
PyAV wheels cannot be loaded (e.g. Windows 11 Smart App Control blocking
unsigned binaries, locked-down corporate environments, or sandboxes that
forbid loading arbitrary DLLs) but where a signed/system `ffmpeg` binary is
available on PATH. Select it via the `audio_backend='ffmpeg'` parameter on
`WhisperModel` (or by calling `decode_audio(..., backend='ffmpeg')`).
"""

import gc
import io
import itertools
import os
import shutil
import subprocess

from typing import BinaryIO, Optional, Union

import numpy as np

SUPPORTED_AUDIO_BACKENDS = ("pyav", "ffmpeg")


def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
    backend: str = "pyav",
):
    """Decodes the audio.

    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate.
      split_stereo: Return separate left and right channels.
      backend: Which decoding backend to use. One of:
        - "pyav" (default): decode in-process using the PyAV library.
        - "ffmpeg": shell out to the `ffmpeg` binary via subprocess.
          Requires `ffmpeg` to be available on PATH (or set the
          `FFMPEG_EXE` environment variable to the absolute path of the
          executable).

    Returns:
      A float32 Numpy array.

      If `split_stereo` is enabled, the function returns a 2-tuple with the
      separated left and right channels.

    Raises:
      ValueError: if `backend` is not one of the supported backends.
      RuntimeError: if `backend='ffmpeg'` and the `ffmpeg` binary cannot be
        located, or decoding fails.
    """
    if backend not in SUPPORTED_AUDIO_BACKENDS:
        raise ValueError(
            f"Unsupported audio_backend={backend!r}. "
            f"Expected one of {SUPPORTED_AUDIO_BACKENDS}."
        )

    if backend == "ffmpeg":
        return _decode_audio_ffmpeg(
            input_file,
            sampling_rate=sampling_rate,
            split_stereo=split_stereo,
        )

    return _decode_audio_pyav(
        input_file,
        sampling_rate=sampling_rate,
        split_stereo=split_stereo,
    )


def _decode_audio_pyav(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    # Import PyAV lazily so environments that cannot load its DLLs/shared
    # libraries (e.g. Windows 11 Smart App Control) can still use the
    # ffmpeg backend without ever triggering an `import av`.
    import av

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )

    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(input_file, mode="r", metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    # It appears that some objects related to the resampler are not freed
    # unless the garbage collector is manually run.
    # https://github.com/SYSTRAN/faster-whisper/issues/390
    # note that this slows down loading the audio a little bit
    # if that is a concern, please use ffmpeg directly as in here:
    # https://github.com/openai/whisper/blob/25639fc/whisper/audio.py#L25-L62
    del resampler
    gc.collect()

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # Convert s16 back to f32.
    audio = audio.astype(np.float32) / 32768.0

    if split_stereo:
        left_channel = audio[0::2]
        right_channel = audio[1::2]
        return left_channel, right_channel

    return audio


def _find_ffmpeg_executable() -> str:
    """Locate the ffmpeg binary. Honors FFMPEG_EXE env var first."""
    env = os.environ.get("FFMPEG_EXE")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "audio_backend='ffmpeg' requires the ffmpeg binary to be available "
        "on PATH (or set the FFMPEG_EXE environment variable to its absolute "
        "path). Install ffmpeg from https://ffmpeg.org/download.html."
    )


def _decode_audio_ffmpeg(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    ffmpeg = _find_ffmpeg_executable()

    channels = 2 if split_stereo else 1
    is_path = isinstance(input_file, (str, bytes, os.PathLike))

    cmd = [
        ffmpeg,
        "-nostdin",
        "-threads", "0",
        "-i", str(input_file) if is_path else "-",
        "-f", "s16le",
        "-ac", str(channels),
        "-acodec", "pcm_s16le",
        "-ar", str(sampling_rate),
        "-loglevel", "error",
        "-",
    ]

    stdin_data: Optional[bytes] = None
    if not is_path:
        if hasattr(input_file, "seek"):
            try:
                input_file.seek(0)
            except Exception:
                pass
        stdin_data = input_file.read()

    try:
        completed = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg decode failed (exit {exc.returncode}): "
            f"{exc.stderr.decode('utf-8', errors='replace')}"
        ) from exc

    audio = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    if split_stereo:
        left = audio[0::2]
        right = audio[1::2]
        return left, right

    return audio


def _ignore_invalid_frames(frames):
    import av  # noqa: WPS433 (mirror import location of _decode_audio_pyav)

    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples=None):
    import av  # noqa: WPS433 (PyAV only needed inside the pyav backend)

    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


def pad_or_trim(array, length: int = 3000, *, axis: int = -1):
    """
    Pad or trim the Mel features array to 3000, as expected by the encoder.
    """
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array
