import json

from types import SimpleNamespace

import pytest
import tokenizers

from faster_whisper import WhisperModel


def make_files(with_preprocessor=True):
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")
    )
    files = {
        "config.json": json.dumps(
            {"lang_ids": [], "suppress_ids": [], "suppress_ids_begin": []}
        ).encode(),
        "model.bin": b"controlled model boundary",
        "tokenizer.json": tokenizer.to_str().encode(),
    }
    if with_preprocessor:
        files["preprocessor_config.json"] = json.dumps(
            {"feature_size": 8, "hop_length": 128}
        ).encode()
    return files


@pytest.mark.parametrize("with_preprocessor", [False, True])
@pytest.mark.parametrize("load_fails", [False, True])
def test_model_files_preserved(monkeypatch, tmp_path, with_preprocessor, load_fails):
    files = make_files(with_preprocessor)
    original = files.copy()
    received = []

    def load(*args, **kwargs):
        received.append(kwargs["files"].copy())
        if load_fails:
            raise RuntimeError("controlled loader failure")
        return SimpleNamespace(is_multilingual=False)

    monkeypatch.setattr("ctranslate2.models.Whisper", load)
    if load_fails:
        with pytest.raises(RuntimeError, match="controlled loader failure"):
            WhisperModel(str(tmp_path / "memory-model"), files=files)
    else:
        model = WhisperModel(str(tmp_path / "memory-model"), files=files)
        assert model.hf_tokenizer.encode("hello").ids == [1]
        assert model.feature_extractor.hop_length == (128 if with_preprocessor else 160)

    assert "tokenizer.json" not in received[0]
    assert "preprocessor_config.json" not in received[0]
    assert received[0]["model.bin"] is original["model.bin"]
    assert files == original
    assert all(files[name] is value for name, value in original.items())


def test_model_files_can_be_reused_without_hub_fallback(monkeypatch, tmp_path):
    files = make_files()
    monkeypatch.setattr(
        "ctranslate2.models.Whisper",
        lambda *args, **kwargs: SimpleNamespace(is_multilingual=False),
    )

    def unexpected_download(*args, **kwargs):
        raise AssertionError("in-memory tokenizer must not use the Hub")

    monkeypatch.setattr(tokenizers.Tokenizer, "from_pretrained", unexpected_download)
    first = WhisperModel(str(tmp_path / "memory-model"), files=files)
    second = WhisperModel(str(tmp_path / "memory-model"), files=files)
    assert first.hf_tokenizer.to_str() == second.hf_tokenizer.to_str()
    assert (
        first.feature_extractor.hop_length == second.feature_extractor.hop_length == 128
    )
