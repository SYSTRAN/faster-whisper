import numpy as np
import pytest

from faster_whisper.feature_extractor import FeatureExtractor


@pytest.mark.parametrize(
    "n_fft,padding,n_samples", [(400, 160, 16000), (401, 0, 700013)]
)
def test_log_mel_matches_full_stft(n_fft, padding, n_samples, monkeypatch):
    extractor = FeatureExtractor(n_fft=n_fft)
    audio = np.random.default_rng(7).standard_normal(n_samples).astype(np.float32)
    audio[: n_samples // 2] *= 1e-5
    padded = np.pad(audio, (0, padding))
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    stft = extractor.stft(
        padded, n_fft, extractor.hop_length, window=window, return_complex=True
    )
    mel = extractor.mel_filters @ (np.abs(stft.astype(np.complex64)[..., :-1]) ** 2)
    log_mel = np.log10(np.maximum(mel, 1e-10))
    expected = (np.maximum(log_mel, log_mel.max() - 8.0) + 4.0) / 4.0

    block_lengths = []
    original_stft = extractor.stft

    def record_block(samples, *args, **kwargs):
        block_lengths.append(samples.shape[-1])
        return original_stft(samples, *args, **kwargs)

    monkeypatch.setattr(extractor, "stft", record_block)
    actual = extractor(audio, padding=padding)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    if n_samples > 16000:
        assert max(block_lengths) < n_samples
