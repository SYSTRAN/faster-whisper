import numpy as np

from faster_whisper.vad_predictor.predictor import VadPredictorModel


def _cast_audio(audio: np.ndarray) -> np.ndarray:
  """ Convert the audio in [audio] into  16-bit integer PCM"""
  audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
  return audio_int16


class TenVadPredictor(VadPredictorModel):
  window_size_samples : int = 256

  def __init__(self):
    from ten_vad import TenVad
    self.hop_size = self.window_size_samples
    self.model = TenVad(self.hop_size, 0.5)

  def __call__(self, audio: np.ndarray):
    num_samples = self.window_size_samples
    assert (
        audio.ndim == 2
    ), "Input should be a 2D array with size (batch_size, num_samples)"
    assert (
        audio.shape[1] % num_samples == 0
    ), "Input size should be a multiple of num_samples"

    batch_size = audio.shape[0]
    num_frames = audio.shape[1] // self.hop_size

    out = []
    for batch_idx in range(0, batch_size):
      for frame in range(num_frames):
        audio_data = _cast_audio(audio[batch_idx][frame * self.hop_size: (frame + 1) * self.hop_size])
        out_probability, _ = self.model.process(audio_data)
        out.append(out_probability)

    return np.array(out).reshape((1, -1, 1))