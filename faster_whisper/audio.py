"""We use the PyAV library to decode the audio: https://github.com/PyAV-Org/PyAV

The advantage of PyAV is that it bundles the FFmpeg libraries so there is no additional
system dependencies. FFmpeg does not need to be installed on the system.

However, the API is quite low-level so we need to manipulate audio frames directly.
"""

import gc
import io
import itertools
from typing import BinaryIO

import av
import numpy as np

# `metadata_errors` was removed in PyAV 19; pass it only for older versions
# to preserve the previous behavior.
_AV_OPEN_KWARGS = (
    {"metadata_errors": "ignore"} if int(av.__version__.split(".")[0]) < 19 else {}
)

# Decode errors that denote corrupt input data and are safe to skip for the
# affected packet. `UndefinedError` is how some corrupt packets surface when
# FFmpeg has no mapped error code for them. Anything else (bug, permissions,
# ...) still propagates.
_SKIPPABLE_DECODE_ERRORS = (av.error.InvalidDataError, av.error.UndefinedError)


def decode_audio(
    input_file: str | BinaryIO,
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

    def make_resampler():
        return av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono" if not split_stereo else "stereo",
            rate=sampling_rate,
        )

    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(input_file, mode="r", **_AV_OPEN_KWARGS) as container:
        frames = _ignore_invalid_frames(_iter_audio_packets(container))
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, make_resampler)

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
    gc.collect()

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # Convert s16 back to f32.
    audio = audio.astype(np.float32) / 32768.0

    if split_stereo:
        left_channel = audio[0::2]
        right_channel = audio[1::2]
        return left_channel, right_channel

    return audio


def _iter_audio_packets(container):
    """Decode packets one at a time so that a decode error only skips the
    offending packet. Iterating `container.decode()` directly cannot recover:
    once the underlying generator raises, it is dead and the rest of the
    stream is silently lost — which happens e.g. when the stream changes
    sample rate mid-file.
    """
    packets = container.demux(audio=0)
    while True:
        try:
            packet = next(packets)
        except StopIteration:
            break
        except _SKIPPABLE_DECODE_ERRORS:
            continue
        try:
            yield from packet.decode()
        except _SKIPPABLE_DECODE_ERRORS:
            continue


def _ignore_invalid_frames(frames):
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples=None):
    fifo = av.audio.fifo.AudioFifo()
    fifo_params = None

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        params = (frame.format.name, frame.layout.name, frame.sample_rate)
        if fifo.samples > 0 and params != fifo_params:
            # The stream changed parameters mid-file (e.g. concatenated
            # streams with different sample rates). Flush the audio grouped so
            # far and start a new FIFO, since AudioFifo.write() rejects frames
            # that do not match its parameters. The resampler downstream
            # normalizes every group to the target format anyway.
            yield fifo.read()
            fifo = av.audio.fifo.AudioFifo()
        fifo_params = params
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, make_resampler):
    # The resampler locks onto the parameters of the first frame it sees, so a
    # mid-stream parameter change (e.g. a different sample rate) requires
    # flushing it and starting a new one. `None` flushes the resampler.
    resampler = None
    resampler_params = None

    for frame in itertools.chain(frames, [None]):
        if frame is None:
            if resampler is not None:
                yield from resampler.resample(None)
            break

        params = (frame.format.name, frame.layout.name, frame.sample_rate)
        if resampler is None:
            resampler = make_resampler()
            resampler_params = params
        elif params != resampler_params:
            yield from resampler.resample(None)
            resampler = make_resampler()
            resampler_params = params

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
