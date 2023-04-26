import os

from faster_whisper import download_model


def test_download_model(tmpdir):
    output_dir = str(tmpdir.join("model"))

    model_dir = download_model("tiny", output_dir=output_dir)

    assert os.path.isdir(output_dir)
    assert not os.path.islink(model_dir)

    files = os.listdir(model_dir)
    for filename in ["tokenizer.json", "vocabulary.txt", "config.json", "model.bin"]:
        assert filename in files
