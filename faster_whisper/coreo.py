"""Binary stdin/stdout bridge to the pinned DUNGU Coreo stream transform."""

import argparse
import os
import subprocess
import sys
import tempfile

from pathlib import Path
from typing import BinaryIO, Optional

DEFAULT_FRAMES_PER_BLOCK = 16384
MAX_FRAMES_PER_BLOCK = 1048576
BYTES_PER_STEREO_FRAME = 2 * 4
BYTES_PER_COREO_FRAME = 4 * 4
COPY_CHUNK_SIZE = 65536
DEFAULT_SCRIPT_PATH = (
    Path(__file__).parent / "assets" / "Convert-AnythingToCoreoFloat.ps1"
)


class CoreoStreamError(RuntimeError):
    """The Coreo converter rejected the stream or violated its contract."""


def _validate_positive_integer(name, value, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer" % name)
    if value <= 0 or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError("%s must be greater than zero" % name)
        raise ValueError("%s must be between 1 and %d" % (name, maximum))


def _resolve_script_path(script_path):
    path = DEFAULT_SCRIPT_PATH if script_path is None else Path(script_path)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError("Coreo PowerShell script not found: %s" % path)
    return path


def _command(powershell_executable, script_path, frames_per_block, mode):
    command = [
        os.fspath(powershell_executable),
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-File",
        os.fspath(script_path),
        mode,
    ]
    if mode == "-StdinStdout":
        command.extend(["-FramesPerBlock", str(frames_per_block)])
    return command


def _write_all(stream, data):
    view = memoryview(data).cast("B")
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise OSError("output stream did not accept the complete byte chunk")
        view = view[written:]


def _error_detail(stderr):
    return stderr.decode("utf-8", errors="replace").strip()


def transform_stereo_float32(
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    *,
    powershell_executable: str = "pwsh",
    script_path: Optional[Path] = None,
    frames_per_block: int = DEFAULT_FRAMES_PER_BLOCK,
    chunk_size: int = COPY_CHUNK_SIZE,
) -> int:
    """Transform raw stereo float32 little-endian frames to Coreo four-channel.

    Input frames are interleaved ``(left, right)``. Output frames are
    interleaved ``(negative reversed left, negative reversed right, negative
    forward left, negative forward right)``. The whole finite input is
    consumed before any transformed bytes are copied to ``output_stream``.

    Returns the number of frames transformed. No container headers are read
    or written.
    """
    _validate_positive_integer(
        "frames_per_block", frames_per_block, MAX_FRAMES_PER_BLOCK
    )
    _validate_positive_integer("chunk_size", chunk_size)
    if not callable(getattr(input_stream, "read", None)):
        raise TypeError("input_stream must provide a read() method")
    if not callable(getattr(output_stream, "write", None)):
        raise TypeError("output_stream must provide a write() method")

    script = _resolve_script_path(script_path)
    command = _command(
        powershell_executable,
        script,
        frames_per_block,
        "-StdinStdout",
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    input_bytes = 0
    output_bytes = 0
    input_error = None
    broken_pipe = None

    try:
        with tempfile.SpooledTemporaryFile(
            max_size=1024 * 1024, mode="w+b"
        ) as transformed_output:
            try:
                while True:
                    chunk = input_stream.read(chunk_size)
                    if chunk is None:
                        raise OSError(
                            "input stream returned no data without reaching EOF"
                        )
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise TypeError("input_stream must return bytes")

                    view = memoryview(chunk).cast("B")
                    while view:
                        written = process.stdin.write(view)
                        if written is None or written <= 0:
                            raise BrokenPipeError(
                                "PowerShell converter closed its input stream"
                            )
                        input_bytes += written
                        view = view[written:]
            except BrokenPipeError as error:
                broken_pipe = error
            except Exception as error:
                input_error = error
            finally:
                try:
                    process.stdin.close()
                except BrokenPipeError as error:
                    if broken_pipe is None:
                        broken_pipe = error

            while True:
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    break
                transformed_output.write(chunk)
                output_bytes += len(chunk)

            stderr = process.stderr.read()
            return_code = process.wait()

            if return_code != 0:
                detail = _error_detail(stderr)
                message = "Coreo PowerShell converter exited with code %d" % return_code
                if detail:
                    message += ": " + detail
                raise CoreoStreamError(message)
            if input_error is not None:
                raise CoreoStreamError(
                    "failed while reading the input stream: %s" % input_error
                ) from input_error
            if broken_pipe is not None:
                raise CoreoStreamError(
                    "PowerShell converter closed its input before the stream was complete"
                ) from broken_pipe
            if input_bytes % BYTES_PER_STEREO_FRAME:
                raise CoreoStreamError(
                    "input length is not a whole number of stereo float32 frames"
                )

            frame_count = input_bytes // BYTES_PER_STEREO_FRAME
            expected_output_bytes = frame_count * BYTES_PER_COREO_FRAME
            if output_bytes != expected_output_bytes:
                raise CoreoStreamError(
                    "Coreo converter produced %d bytes for %d input frames; expected %d"
                    % (output_bytes, frame_count, expected_output_bytes)
                )

            transformed_output.seek(0)
            while True:
                chunk = transformed_output.read(chunk_size)
                if not chunk:
                    break
                _write_all(output_stream, chunk)
            output_stream.flush()

            return frame_count
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()


def run_coreo_self_tests(powershell_executable="pwsh", script_path=None):
    """Run the upstream script's in-memory stream-transform tests."""
    script = _resolve_script_path(script_path)
    command = _command(
        powershell_executable,
        script,
        DEFAULT_FRAMES_PER_BLOCK,
        "-StreamSelfTest",
    )
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = _error_detail(result.stderr)
        message = "Coreo stream self-tests exited with code %d" % result.returncode
        if detail:
            message += ": " + detail
        raise CoreoStreamError(message)

    return result.stdout.decode("utf-8", errors="replace").splitlines()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Transform raw interleaved stereo float32 little-endian stdin to "
            "raw interleaved four-channel Coreo float32 stdout."
        )
    )
    parser.add_argument(
        "--powershell",
        default=os.environ.get("FASTER_WHISPER_COREO_POWERSHELL", "pwsh"),
        help="PowerShell 7 executable (default: pwsh)",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="override the bundled PowerShell converter",
    )
    parser.add_argument(
        "--frames-per-block",
        type=int,
        default=DEFAULT_FRAMES_PER_BLOCK,
        help="transform block size (default: %(default)s)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the upstream transform's in-memory self-tests",
    )
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            for line in run_coreo_self_tests(args.powershell, args.script):
                print(line)
            return 0

        frames = transform_stereo_float32(
            sys.stdin.buffer,
            sys.stdout.buffer,
            powershell_executable=args.powershell,
            script_path=args.script,
            frames_per_block=args.frames_per_block,
        )
    except (CoreoStreamError, OSError, TypeError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1

    print(
        "COREO stream complete: %d stereo float32 frames transformed to raw "
        "4-channel float32 stdout." % frames,
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
