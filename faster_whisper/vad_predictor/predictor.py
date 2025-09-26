from typing import Protocol
import numpy as np

class VadPredictorModel(Protocol):
  window_size_samples: int

  def __call__(
      self, audio: np.ndarray,
  ) -> np.ndarray:
    ...