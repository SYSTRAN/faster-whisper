import torch


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
        self.mel_filters = self.get_mel_filters(
            sampling_rate, n_fft, n_mels=feature_size
        )

    @staticmethod
    def get_mel_filters(sr, n_fft, n_mels=128):
        """
        Implementation of librosa.filters.mel in Pytorch
        """
        # Initialize the weights
        n_mels = int(n_mels)

        # Center freqs of each FFT bin
        fftfreqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = torch.linspace(min_mel, max_mel, n_mels + 2)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = torch.log(torch.tensor(6.4)) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

        mel_f = freqs

        fdiff = torch.diff(mel_f)
        ramps = mel_f.view(-1, 1) - fftfreqs.view(1, -1)

        lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
        upper = ramps[2:] / fdiff[1:].unsqueeze(1)

        # Intersect them with each other and zero, vectorized across all i
        weights = torch.maximum(torch.zeros_like(lower), torch.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm.unsqueeze(1)

        return weights

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

        mel_spec = self.mel_filters.to(waveform.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # When the model is running on multiple GPUs, the output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        return log_spec.cpu() if to_cpu else log_spec
