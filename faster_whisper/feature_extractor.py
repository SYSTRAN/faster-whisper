import torch


# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py  # noqa: E501
class FeatureExtractor:
    def __init__(
        self,
        device: str = "auto",
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.sampling_rate = sampling_rate
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.time_per_frame = hop_length / sampling_rate

        self.window = torch.hann_window(self.n_fft).to(self.device)

        self.mel_filters = self.get_mel_filters(
            sampling_rate, n_fft, n_mels=feature_size
        ).to(self.device)

    @staticmethod
    def get_mel_filters(sr: int, n_fft: int, n_mels: int = 128) -> torch.Tensor:
        """
        Implementation of librosa.filters.mel in PyTorch.
        """
        # Initialize the weights
        n_mels = int(n_mels)

        # Center freqs of each FFT bin
        fftfreqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965  # Corresponds to the maximum mel value used

        mels = torch.linspace(min_mel, max_mel, n_mels + 2)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = torch.log(torch.tensor(6.4)) / 27.0  # step size for log region

        # Apply nonlinear scaling for frequencies above min_log_hz
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

        mel_f = freqs

        # Compute the difference between adjacent mel frequencies
        fdiff = torch.diff(mel_f)

        # Create ramps for lower and upper edges
        ramps = mel_f.view(-1, 1) - fftfreqs.view(1, -1)

        lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
        upper = ramps[2:] / fdiff[1:].unsqueeze(1)

        # Intersect them with each other and zero
        weights = torch.maximum(torch.zeros_like(lower), torch.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]).unsqueeze(1)
        weights *= enorm

        return weights

    def __call__(
        self,
        waveform: torch.Tensor,
        padding: bool = True,
        chunk_length: int = None,
        to_cpu: bool = False,
    ) -> torch.Tensor:
        """
        Compute the log-Mel spectrogram of the provided audio.
        """
        n_samples = (
            chunk_length * self.sampling_rate if chunk_length is not None else self.n_samples
        )

        if waveform.dtype is not torch.float32:
            waveform = waveform.to(torch.float32)

        if self.device == "cuda" and not waveform.is_cuda:
            waveform = waveform.to(self.device)

        if padding:
            waveform = torch.nn.functional.pad(waveform, (0, n_samples))

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )

        magnitudes = stft.abs() ** 2

        mel_spec = self.mel_filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec.cpu() if to_cpu else log_spec
