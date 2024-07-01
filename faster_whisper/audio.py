from typing import BinaryIO, Union

import torch
import torchaudio


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
      A float32 Torch Tensor.

      If `split_stereo` is enabled, the function returns a 2-tuple with the
      separated left and right channels.
    """

    waveform, audio_sf = torchaudio.load(input_file)  # waveform: channels X T

    if audio_sf != sampling_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=audio_sf, new_freq=sampling_rate
        )
    if split_stereo:
        return waveform[0], waveform[1]

    return waveform.mean(0)


def pad_or_trim(array, length: int, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    axis = axis % array.ndim
    if array.shape[axis] > length:
        idx = [Ellipsis] * axis + [slice(length)] + [Ellipsis] * (array.ndim - axis - 1)
        return array[idx]

    if array.shape[axis] < length:
        pad_widths = (
            [
                0,
            ]
            * array.ndim
            * 2
        )
        pad_widths[2 * axis] = length - array.shape[axis]
        array = torch.nn.functional.pad(array, tuple(pad_widths[::-1]))

    return array
