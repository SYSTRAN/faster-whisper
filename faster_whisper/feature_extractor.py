import numpy as np


class FeatureExtractor:
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    ):
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

    @staticmethod
    def get_mel_filters(sr, n_fft, n_mels=128):
        # Initialize the weights
        n_mels = int(n_mels)

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = np.linspace(min_mel, max_mel, n_mels + 2)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        fdiff = np.diff(freqs)
        ramps = freqs.reshape(-1, 1) - fftfreqs.reshape(1, -1)

        lower = -ramps[:-2] / np.expand_dims(fdiff[:-1], axis=1)
        upper = ramps[2:] / np.expand_dims(fdiff[1:], axis=1)

        # Intersect them with each other and zero, vectorized across all i
        weights = np.maximum(np.zeros_like(lower), np.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (freqs[2 : n_mels + 2] - freqs[:n_mels])
        weights *= np.expand_dims(enorm, axis=1)

        return weights

    @staticmethod
    def stft(
        input_array: np.ndarray,
        n_fft: int,
        hop_length: int = None,
        win_length: int = None,
        window: np.ndarray = None,
        center: bool = True,
        mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = None,
        return_complex: bool = None,
    ):
        # Default initialization for hop_length and win_length
        hop_length = hop_length if hop_length is not None else n_fft // 4
        win_length = win_length if win_length is not None else n_fft
        input_is_complex = np.iscomplexobj(input_array)

        # Determine if the output should be complex
        return_complex = (
            return_complex
            if return_complex is not None
            else (input_is_complex or (window is not None and np.iscomplexobj(window)))
        )

        if not return_complex and return_complex is None:
            raise ValueError(
                "stft requires the return_complex parameter for real inputs."
            )

        # Input checks
        if not np.issubdtype(input_array.dtype, np.floating) and not input_is_complex:
            raise ValueError(
                "stft: expected an array of floating point or complex values,"
                f" got {input_array.dtype}"
            )

        if input_array.ndim > 2 or input_array.ndim < 1:
            raise ValueError(
                f"stft: expected a 1D or 2D array, but got {input_array.ndim}D array"
            )

        # Handle 1D input
        if input_array.ndim == 1:
            input_array = np.expand_dims(input_array, axis=0)
            input_array_1d = True
        else:
            input_array_1d = False

        # Center padding if required
        if center:
            pad_amount = n_fft // 2
            input_array = np.pad(
                input_array, ((0, 0), (pad_amount, pad_amount)), mode=mode
            )

        batch, length = input_array.shape

        # Additional input checks
        if n_fft <= 0 or n_fft > length:
            raise ValueError(
                f"stft: expected 0 < n_fft <= {length}, but got n_fft={n_fft}"
            )

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
                    f"stft: expected a 1D window array of size equal to win_length={win_length}, "
                    f"but got window with size {window.shape}"
                )

        # Handle padding of the window if necessary
        if win_length < n_fft:
            left = (n_fft - win_length) // 2
            window_ = np.zeros(n_fft, dtype=window.dtype)
            window_[left : left + win_length] = window
        else:
            window_ = window

        # Calculate the number of frames
        n_frames = 1 + (length - n_fft) // hop_length

        # Time to columns
        input_array = np.lib.stride_tricks.as_strided(
            input_array,
            (batch, n_frames, n_fft),
            (
                input_array.strides[0],
                hop_length * input_array.strides[1],
                input_array.strides[1],
            ),
        )

        if window_ is not None:
            input_array = input_array * window_

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
            output = np.fft.fft(input_array, n=n_fft, axis=-1, norm=norm)
        else:
            output = np.fft.rfft(input_array, n=n_fft, axis=-1, norm=norm)

        output = output.transpose((0, 2, 1))

        if input_array_1d:
            output = output.squeeze(0)

        return output if return_complex else np.real(output)

    def __call__(self, waveform: np.ndarray, padding=160, chunk_length=None):
        """
        Compute the log-Mel spectrogram of the provided audio.
        """

        if chunk_length is not None:
            self.n_samples = chunk_length * self.sampling_rate
            self.nb_max_frames = self.n_samples // self.hop_length

        if waveform.dtype is not np.float32:
            waveform = waveform.astype(np.float32)

        if padding:
            waveform = np.pad(waveform, (0, padding))

        window = np.hanning(self.n_fft + 1)[:-1].astype("float32")

        stft = self.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=window,
            return_complex=True,
        ).astype("complex64")
        magnitudes = np.abs(stft[..., :-1]) ** 2

        mel_spec = self.mel_filters @ magnitudes

        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec
