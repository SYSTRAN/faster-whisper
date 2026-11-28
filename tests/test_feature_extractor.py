import numpy as np
import pytest

from faster_whisper import feature_extractor
from faster_whisper.feature_extractor import FeatureExtractor


def _reference_features(extractor, waveform, padding=160):
    waveform = np.pad(waveform, (0, padding))
    window = np.hanning(extractor.n_fft + 1)[:-1].astype("float32")
    stft = extractor.stft(
        waveform,
        extractor.n_fft,
        extractor.hop_length,
        window=window,
        return_complex=True,
    ).astype("complex64")
    mel_spec = extractor.mel_filters @ (np.abs(stft[..., :-1]) ** 2)
    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


@pytest.mark.parametrize("num_samples", [0, 1, 160, 16000, 480000, 480001, 1520003])
@pytest.mark.parametrize("feature_size", [80, 128])
@pytest.mark.parametrize("block_frames", [1, feature_extractor.STFT_BLOCK_FRAMES, 3000])
def test_features_match_single_pass(
    monkeypatch, num_samples, feature_size, block_frames
):
    monkeypatch.setattr(feature_extractor, "STFT_BLOCK_FRAMES", block_frames)
    extractor = FeatureExtractor(feature_size=feature_size)
    rng = np.random.default_rng(0)
    waveform = (rng.standard_normal(num_samples) * 0.1).astype(np.float32)

    features = extractor(waveform)
    expected = _reference_features(extractor, waveform)

    assert features.dtype == np.float32
    assert features.shape == expected.shape
    np.testing.assert_array_equal(features, expected)
