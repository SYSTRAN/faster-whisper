import torch
import numpy as np
import os


# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py  # noqa: E501
class FeatureExtractor:
    def __init__(
        self,
        device: str = "auto",
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        mel_filter_path: str = "path/to/mel_filters.npz",
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.time_per_frame = hop_length / sampling_rate
        self.sampling_rate = sampling_rate
        self.mel_filters = self.load_mel_filters(mel_filter_path, feature_size)

    def load_mel_filters(self, filepath: str, n_mels: int):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mel filter file not found at: {filepath}")

        mel_data = np.load(filepath)
        key = f"mel_{n_mels}"
        if key not in mel_data:
            available_keys = ', '.join(mel_data.keys())
            raise KeyError(
                f"Key '{key}' not found in mel_filters.npz. Available keys: {available_keys}"
            )
        
        mel_filters_np = mel_data[key]
        mel_filters = torch.from_numpy(mel_filters_np).float().to(self.device)
        return mel_filters

    def __call__(self, waveform, padding=True, chunk_length=None, to_cpu=False):
        """
        Compute the log-Mel spectrogram of the provided audio.
        """

        if chunk_length is not None:
            self.n_samples = chunk_length * self.sampling_rate
            self.nb_max_frames = self.n_samples // self.hop_length

        if waveform.dtype is not torch.float32:
            waveform = waveform.to(torch.float32)

        waveform = (
            waveform.to(self.device)
            if self.device == "cuda" and not waveform.is_cuda
            else waveform
        )

        if padding:
            waveform = torch.nn.functional.pad(waveform, (0, self.n_samples))

        window = torch.hann_window(self.n_fft).to(waveform.device)

        stft = torch.stft(
            waveform, self.n_fft, self.hop_length, window=window, return_complex=True
        )
        magnitudes = stft[..., :-1].abs() ** 2

        mel_spec = self.mel_filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # When the model is running on multiple GPUs, the output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        return log_spec.cpu() if to_cpu else log_spec
