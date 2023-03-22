import os

import ctranslate2
import pytest


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture
def jfk_path(data_dir):
    return os.path.join(data_dir, "jfk.flac")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    model_path = str(tmp_path_factory.mktemp("data") / "model")
    convert_model("tiny", model_path)
    return model_path


def convert_model(size, output_dir):
    name = "openai/whisper-%s" % size

    ctranslate2.converters.TransformersConverter(
        name,
        copy_files=["tokenizer.json"],
        load_as_float16=True,
    ).convert(output_dir, quantization="float16")
