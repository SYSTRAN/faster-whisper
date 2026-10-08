import numpy as np
import pytest

from faster_whisper.feature_extractor import FeatureExtractor


def reference_stft(audio, n_fft, window, center=False, normalized=False):
    if center:
        padding = [(0, 0)] * audio.ndim
        padding[-1] = (n_fft // 2, n_fft // 2)
        audio = np.pad(audio, padding, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft, axis=-1)
    frames = frames[..., ::2, :] * window
    return np.fft.fft(frames, axis=-1, norm="ortho" if normalized else None).swapaxes(
        -1, -2
    )


@pytest.mark.parametrize("n_fft", [8, 9])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stft_real_full_spectrum(n_fft, batched, center, normalized, dtype):
    audio = np.arange(32, dtype=dtype)
    if batched:
        audio = np.stack([audio, audio[::-1]])
    window = np.hanning(n_fft).astype(dtype)

    result = FeatureExtractor.stft(
        audio,
        n_fft,
        hop_length=2,
        window=window,
        center=center,
        normalized=normalized,
        onesided=False,
        return_complex=True,
    )

    expected = reference_stft(audio, n_fft, window, center, normalized)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n_fft", [8, 9])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("onesided", [None, False])
@pytest.mark.parametrize("short_window", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stft_complex_window(n_fft, batched, onesided, short_window, dtype):
    audio = np.arange(32, dtype=dtype)
    if batched:
        audio = np.stack([audio, audio[::-1]])
    win_length = n_fft - 2 if short_window else n_fft
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    window = np.exp(1j * np.arange(win_length) / 3).astype(complex_dtype)

    result = FeatureExtractor.stft(
        audio,
        n_fft,
        hop_length=2,
        win_length=win_length,
        window=window,
        center=False,
        onesided=onesided,
    )

    padding = n_fft - win_length
    padded_window = np.pad(window, (padding // 2, padding - padding // 2))
    expected = reference_stft(audio, n_fft, padded_window)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("complex_window", [False, True])
def test_stft_one_sided_spectrum(complex_input, complex_window):
    audio = np.arange(32, dtype=np.float64)
    window = np.ones(8)
    if complex_input:
        audio = audio + 1j
    if complex_window:
        window = window + 1j
    kwargs = {"window": window, "center": False, "return_complex": True}

    if complex_input or complex_window:
        with pytest.raises(ValueError, match="onesided"):
            FeatureExtractor.stft(audio, 8, hop_length=2, onesided=True, **kwargs)
    else:
        result = FeatureExtractor.stft(audio, 8, hop_length=2, onesided=True, **kwargs)
        expected = reference_stft(audio, 8, window)[..., :5, :]
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("complex_input", [False, True])
def test_stft_default_spectrum(complex_input):
    audio = np.arange(32, dtype=np.float64)
    if complex_input:
        audio = audio + 1j
    window = np.hanning(8)

    result = FeatureExtractor.stft(
        audio, 8, hop_length=2, window=window, center=False, return_complex=True
    )

    expected = reference_stft(audio, 8, window)
    if not complex_input:
        expected = expected[..., :5, :]
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
