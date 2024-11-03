"""We use system FFmpeg when available for faster audio decoding. As a fallback, we use
the PyAV library: https://github.com/PyAV-Org/PyAV
If FFmpeg is installed on the system, we use it directly for optimal performance.
If FFmpeg is not available, PyAV provides the advantage of bundled FFmpeg libraries,
though its low-level API requires direct frame manipulation.
"""
import gc
import io
import itertools
import subprocess
import tempfile
from typing import BinaryIO, Union

import av
import numpy as np
import torch


def is_ffmpeg_available() -> bool:
    try:
        subprocess.check_output(["ffmpeg", "-version"])
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def decode_audio_ffmpeg(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = f"{tmpdir}/temp.wav"
        channels = 2 if split_stereo else 1

        cmd = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-ar",
            str(sampling_rate),
            "-ac",
            str(channels),
            "-acodec",
            "pcm_s16le",
            output_file,
            "-y",
        ]

        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            audio = np.fromfile(output_file, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0

            if split_stereo:
                left_channel = audio[0::2]
                right_channel = audio[1::2]
                return torch.from_numpy(left_channel), torch.from_numpy(right_channel)
            return torch.from_numpy(audio)

        except subprocess.SubprocessError:
            raise RuntimeError("FFmpeg failed to decode the audio file")


def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """Decodes the audio.

    Args:
        input_file: Path to the input file or a file-like object.
        sampling_rate: Resample the audio to this sample rate.
        split_stereo: Return separate left and right channels.

    Returns:
        A float32 Numpy array.
        If `split_stereo` is enabled, the function returns a 2-tuple with the
        separated left and right channels.
    """
    if is_ffmpeg_available():
        try:
            return decode_audio_ffmpeg(input_file, sampling_rate, split_stereo)
        except:
            pass

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )

    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(input_file, mode="r", metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = ignore_invalid_frames(frames)
        frames = group_frames(frames, 500000)
        frames = resample_frames(frames, resampler)
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
        return torch.from_numpy(left_channel), torch.from_numpy(right_channel)
    return torch.from_numpy(audio)


def ignore_invalid_frames(frames):
    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def group_frames(frames, num_samples=None):
    fifo = av.audio.fifo.AudioFifo()
    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)
        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()
    if fifo.samples > 0:
        yield fifo.read()


def resample_frames(frames, resampler):
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


def pad_or_trim(array, length: int = 3000, *, axis: int = -1):
    """Pad or trim the Mel features array to 3000, as expected by the encoder."""
    axis = axis % array.ndim
    if array.shape[axis] > length:
        idx = [Ellipsis] * axis + [slice(length)] + [Ellipsis] * (array.ndim - axis - 1)
        return array[idx]

    if array.shape[axis] < length:
        pad_widths = ([0] * array.ndim * 2)
        pad_widths[2 * axis] = length - array.shape[axis]
        array = torch.nn.functional.pad(array, tuple(pad_widths[::-1]))
    return array
