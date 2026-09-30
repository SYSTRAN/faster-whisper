import io
import shutil
import struct

import pytest

from faster_whisper.coreo import (
    CoreoStreamError,
    run_coreo_self_tests,
    transform_stereo_float32,
)

pytestmark = pytest.mark.skipif(
    shutil.which("pwsh") is None,
    reason="PowerShell 7 is required for the Coreo stream converter",
)


def test_transform_stereo_float32_stream_order_and_polarity():
    source = io.BytesIO(struct.pack("<6f", 1, 10, 2, 20, 3, 30))
    destination = io.BytesIO()

    frame_count = transform_stereo_float32(source, destination, frames_per_block=2)

    assert frame_count == 3
    assert struct.unpack("<12f", destination.getvalue()) == (
        -3,
        -30,
        -1,
        -10,
        -2,
        -20,
        -2,
        -20,
        -1,
        -10,
        -3,
        -30,
    )


@pytest.mark.parametrize(
    "input_bytes", [b"\x00" * 7, struct.pack("<2f", float("nan"), 0)]
)
def test_invalid_streams_fail_without_writing_output(input_bytes):
    destination = io.BytesIO()

    with pytest.raises(CoreoStreamError):
        transform_stereo_float32(io.BytesIO(input_bytes), destination)

    assert destination.getvalue() == b""


def test_empty_stream_produces_empty_output():
    destination = io.BytesIO()

    assert transform_stereo_float32(io.BytesIO(), destination) == 0
    assert destination.getvalue() == b""


def test_upstream_stream_self_tests():
    result = run_coreo_self_tests()

    assert any("exact whole-stream YIN reverse" in line for line in result)
    assert any("empty-stream handling" in line for line in result)
    assert result[-1] == "All stdin/stdout transform tests passed."
