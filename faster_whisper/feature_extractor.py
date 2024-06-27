import torch
import torchaudio.compliance.kaldi as ta_kaldi


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
        self.n_mels = feature_size

    def __call__(self, waveform, padding=True, chunk_length=None, to_cpu=False):
        """
        Compute the log-Mel spectrogram of the provided audio, gives similar results
        whisper's original torch implementation with 1e-5 tolerance.
        """
        if chunk_length is not None:
            self.n_samples = chunk_length * self.sampling_rate
            self.nb_max_frames = self.n_samples // self.hop_length

        if padding:
            waveform = torch.nn.functional.pad(waveform, (0, self.n_samples))

        waveform = waveform.to(self.device) if self.device == "cuda" else waveform

        fbank = ta_kaldi.fbank(
            waveform.unsqueeze(0),
            sample_frequency=self.sampling_rate,
            window_type="hanning",
            num_mel_bins=self.n_mels,
        )
        log_spec = fbank.T

        # normalize

        # Audioset values as default mean and std for audio
        mean_val = -4.2677393
        std_val = 4.5689974
        scaled_features = (log_spec - mean_val) / (std_val * 2)

        # When the model is running on multiple GPUs, the output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        return scaled_features.cpu() if to_cpu else scaled_features
