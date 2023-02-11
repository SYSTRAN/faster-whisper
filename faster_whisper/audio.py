import av
import numpy as np


def decode_audio(input_file, sampling_rate=16000):
    """Decodes the audio.

    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate.

    Returns:
      A float32 Numpy array.
    """
    fifo = av.audio.fifo.AudioFifo()
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sampling_rate,
    )

    with av.open(input_file) as container:
        # Decode and resample each audio frame.
        for frame in container.decode(audio=0):
            frame.pts = None
            for new_frame in resampler.resample(frame):
                fifo.write(new_frame)

        # Flush the resampler.
        for new_frame in resampler.resample(None):
            fifo.write(new_frame)

    frame = fifo.read()

    # Convert s16 back to f32.
    return frame.to_ndarray().flatten().astype(np.float32) / 32768.0
