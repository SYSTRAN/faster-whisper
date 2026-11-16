import numpy as np
import pytest

from faster_whisper.feature_extractor import FeatureExtractor


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("complex_window", [False, True])
@pytest.mark.parametrize("short_window", [False, True])
def test_stft_matches_explicit_full_fft(complex_input, complex_window, short_window):
    samples = np.arange(24, dtype=np.float32) / 24
    if complex_input:
        samples = samples + 1j * samples[::-1]
    win_length = 4 if short_window else 8
    window = None
    expected_window = np.ones(win_length)
    if complex_window:
        window = np.exp(1j * np.arange(win_length))
        expected_window = window
    expected_window = np.pad(expected_window, ((8 - win_length) // 2,) * 2)
    expected = np.stack(
        [
            np.fft.fft(samples[start : start + 8] * expected_window)
            for start in range(0, 17, 4)
        ],
        axis=1,
    )
    result = FeatureExtractor.stft(
        samples,
        n_fft=8,
        hop_length=4,
        win_length=win_length,
        window=window,
        center=False,
        onesided=False,
        return_complex=True,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_stft_rejects_onesided_complex_window():
    with pytest.raises(ValueError, match="onesided"):
        FeatureExtractor.stft(
            np.ones(16, dtype=np.float32),
            n_fft=8,
            window=np.ones(8, dtype=np.complex64),
            center=False,
            onesided=True,
            return_complex=True,
        )
