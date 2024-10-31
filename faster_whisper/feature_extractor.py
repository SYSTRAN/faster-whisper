import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def get_array_module(device: str = "auto"):
    if device in ["auto", "cuda"] and CUPY_AVAILABLE and cp.cuda.is_available():
        return cp, "cuda"
    else:
        return np, "cpu"


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
        self.array_module: np
        self.array_module, self._device = get_array_module(device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.time_per_frame = hop_length / sampling_rate
        self.sampling_rate = sampling_rate
        self.mel_filters = self.get_mel_filters(
            sampling_rate, n_fft, n_mels=feature_size
        ).astype("float32")

    def get_mel_filters(self, sr, n_fft, n_mels=128):
        # Initialize the weights
        n_mels = int(n_mels)

        # Center freqs of each FFT bin
        fftfreqs = self.array_module.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = self.array_module.linspace(min_mel, max_mel, n_mels + 2)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = self.array_module.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * self.array_module.exp(
            logstep * (mels[log_t] - min_log_mel)
        )

        fdiff = self.array_module.diff(freqs)
        ramps = freqs.reshape(-1, 1) - fftfreqs.reshape(1, -1)

        lower = -ramps[:-2] / self.array_module.expand_dims(fdiff[:-1], axis=1)
        upper = ramps[2:] / self.array_module.expand_dims(fdiff[1:], axis=1)

        # Intersect them with each other and zero, vectorized across all i
        weights = self.array_module.maximum(
            self.array_module.zeros_like(lower), self.array_module.minimum(lower, upper)
        )

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (freqs[2 : n_mels + 2] - freqs[:n_mels])
        weights *= self.array_module.expand_dims(enorm, axis=1)

        return weights

    def __call__(self, waveform: np.ndarray, padding=True, chunk_length=None):
        """
        Compute the log-Mel spectrogram of the provided audio.
        """

        if chunk_length is not None:
            self.n_samples = chunk_length * self.sampling_rate
            self.nb_max_frames = self.n_samples // self.hop_length

        if waveform.dtype is not np.float32:
            waveform = waveform.astype(np.float32)

        if padding:
            waveform = np.pad(waveform, (0, self.n_samples))

        window = self.array_module.hanning(self.n_fft + 1)[:-1].astype("float32")

        stft_output = stft(
            self.array_module,
            waveform,
            self.n_fft,
            self.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = self.array_module.abs(stft_output[..., :-1]) ** 2

        mel_spec = self.mel_filters @ magnitudes

        log_spec = self.array_module.log10(
            self.array_module.clip(mel_spec, a_min=1e-10, a_max=None)
        )
        log_spec = self.array_module.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return np.asarray(log_spec.tolist(), dtype=log_spec.dtype)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if device != self.device:
            self.array_module, self._device = get_array_module(device)


def stft(
    array_module: np,
    input_tensor: np.ndarray,
    n_fft: int,
    hop_length: int = None,
    win_length: int = None,
    window: np.ndarray = None,
    center=True,
    mode="reflect",
    normalized=False,
    onesided=None,
    return_complex=None,
):

    # Default initialization for hop_length and win_length
    hop_length = hop_length if hop_length is not None else n_fft // 4
    win_length = win_length if win_length is not None else n_fft
    input_is_complex = np.iscomplexobj(input_tensor)

    # Determine if the output should be complex
    return_complex = (
        return_complex
        if return_complex is not None
        else (input_is_complex or (window is not None and np.iscomplexobj(window)))
    )

    if not return_complex and return_complex is None:
        raise ValueError("stft requires the return_complex parameter for real inputs.")

    # Input checks
    if not np.issubdtype(input_tensor.dtype, np.floating) and not input_is_complex:
        raise ValueError(
            f"stft: expected a tensor of floating point or complex values, got {input_tensor.dtype}"
        )

    if input_tensor.ndim > 2 or input_tensor.ndim < 1:
        raise ValueError(
            f"stft: expected a 1D or 2D tensor, but got {input_tensor.ndim}D tensor"
        )

    # Handle 1D input
    if input_tensor.ndim == 1:
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor_1d = True
    else:
        input_tensor_1d = False

    # Center padding if required
    if center:
        pad_amount = n_fft // 2
        input_tensor = np.pad(
            input_tensor, ((0, 0), (pad_amount, pad_amount)), mode=mode
        )

    batch, length = input_tensor.shape

    # Additional input checks
    if n_fft <= 0 or n_fft > length:
        raise ValueError(f"stft: expected 0 < n_fft <= {length}, but got n_fft={n_fft}")

    if hop_length <= 0:
        raise ValueError(
            f"stft: expected hop_length > 0, but got hop_length={hop_length}"
        )

    if win_length <= 0 or win_length > n_fft:
        raise ValueError(
            f"stft: expected 0 < win_length <= n_fft, but got win_length={win_length}"
        )

    if window is not None:
        if window.ndim != 1 or window.shape[0] != win_length:
            raise ValueError(
                f"stft: expected a 1D window tensor of size equal to win_length={win_length}, "
                f"but got window with size {window.shape}"
            )

    # Handle padding of the window if necessary
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        window_ = array_module.zeros(n_fft, dtype=window.dtype)
        window_[left : left + win_length] = window
    else:
        window_ = window

    # Calculate the number of frames
    n_frames = 1 + (length - n_fft) // hop_length

    # Time to columns
    input_tensor = np.lib.stride_tricks.as_strided(
        input_tensor,
        (batch, n_frames, n_fft),
        (
            input_tensor.strides[0],
            hop_length * input_tensor.strides[1],
            input_tensor.strides[1],
        ),
    )

    if window_ is not None:
        input_tensor = array_module.asarray(input_tensor) * window_

    # FFT and transpose
    complex_fft = input_is_complex
    onesided = onesided if onesided is not None else not complex_fft

    if normalized:
        norm = "ortho"
    else:
        norm = None

    if complex_fft:
        if onesided:
            raise ValueError(
                "Cannot have onesided output if window or input is complex"
            )
        output = array_module.fft.fft(input_tensor, n=n_fft, axis=-1, norm=norm)
    else:
        output = array_module.fft.rfft(input_tensor, n=n_fft, axis=-1, norm=norm)

    output = output.transpose((0, 2, 1))

    if input_tensor_1d:
        output = output.squeeze(0)

    return output if return_complex else array_module.real(output)
