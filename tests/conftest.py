import os

import pytest


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture
def jfk_path(data_dir):
    return os.path.join(data_dir, "jfk.flac")
