from faster_whisper.vad import get_speech_timestamps_using_model, VadOptions, \
  merge_segments
from faster_whisper.vad_predictor.predictor import VadPredictorModel
import numpy as np

class TrivialVadPredictor(VadPredictorModel):
  window_size_samples = 512

  def __init__(self, predictions: list[float]):
    self.predictions = predictions * self.window_size_samples

  def __call__(self, audio: np.ndarray,) -> np.ndarray:
    return np.array(self.predictions).reshape((1, -1, 1))

def off(n):
    return (0 for _ in range(n))

def on(n):
  return (1 for _ in range(n))

def sequence(*patterns):
  for pattern in patterns:
    yield from list(pattern)


data = list(sequence( off(5), on(20), off(50), on(100)))

def test_get_speech_timestamps_using_model():
  vad_model = TrivialVadPredictor(data)
  vad_options = VadOptions()


  # Create audio with the same pattern as our VAD predictions
  # This is just for the length - our TrivialVadPredictor ignores the actual audio content
  audio = np.ones(len(data), dtype="float32")
  
  raw = get_speech_timestamps_using_model(vad_model, audio, vad_options)
  # With our pattern (5 off, 20 on, 50 off), we should get one speech segment
  assert len(raw) == 1
  assert raw[0]["start"] == 5 * vad_model.window_size_samples
  assert raw[0]["end"] == 25 * vad_model.window_size_samples  # 5 + 20
  
  merged = merge_segments(raw, vad_options)
  # After merging, we should still have one segment
  assert len(merged) == 1
  assert merged[0]["start"] == raw[0]["start"]
  assert merged[0]["end"] == raw[0]["end"]
