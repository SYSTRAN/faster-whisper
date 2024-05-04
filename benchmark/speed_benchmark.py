import argparse
import timeit

from typing import Callable

from utils import inference

parser = argparse.ArgumentParser(description="Speed benchmark")
parser.add_argument(
    "--repeat",
    type=int,
    default=3,
    help="Times an experiment will be run.",
)
args = parser.parse_args()


def measure_speed(func: Callable[[], None]):
    # as written in https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat,
    # min should be taken rather than the average
    runtimes = timeit.repeat(
        func,
        repeat=args.repeat,
        number=10,
    )
    print(runtimes)
    print("Min execution time: %.3fs" % (min(runtimes) / 10.0))


if __name__ == "__main__":
    measure_speed(inference)
