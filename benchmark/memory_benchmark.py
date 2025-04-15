import argparse
import time

from typing import Callable

import py3nvml.py3nvml as nvml

from memory_profiler import memory_usage
from utils import MyThread, get_logger, inference

logger = get_logger("faster-whisper")
parser = argparse.ArgumentParser(description="Memory benchmark")
parser.add_argument(
    "--gpu_memory", action="store_true", help="Measure GPU memory usage"
)
parser.add_argument("--device-index", type=int, default=0, help="GPU device index")
parser.add_argument(
    "--interval",
    type=float,
    default=0.5,
    help="Interval at which measurements are collected",
)
args = parser.parse_args()
device_idx = args.device_index
interval = args.interval


def measure_memory(func: Callable[[], None]):
    if args.gpu_memory:
        logger.info(
            "Measuring maximum GPU memory usage on GPU device."
            " Make sure to not have additional processes running on the same GPU."
        )
        # init nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(device_idx)
        gpu_name = nvml.nvmlDeviceGetName(handle)
        gpu_memory_limit = nvml.nvmlDeviceGetMemoryInfo(handle).total >> 20
        gpu_power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        info = {"gpu_memory_usage": [], "gpu_power_usage": []}

        def _get_gpu_info():
            while True:
                info["gpu_memory_usage"].append(
                    nvml.nvmlDeviceGetMemoryInfo(handle).used >> 20
                )
                info["gpu_power_usage"].append(
                    nvml.nvmlDeviceGetPowerUsage(handle) / 1000
                )
                time.sleep(interval)

                if stop:
                    break

            return info

        stop = False
        thread = MyThread(_get_gpu_info, params=())
        thread.start()
        func()
        stop = True
        thread.join()
        result = thread.get_result()

        # shutdown nvml
        nvml.nvmlShutdown()
        max_memory_usage = max(result["gpu_memory_usage"])
        max_power_usage = max(result["gpu_power_usage"])
        print("GPU name: %s" % gpu_name)
        print("GPU device index: %s" % device_idx)
        print(
            "Maximum GPU memory usage: %dMiB / %dMiB (%.2f%%)"
            % (
                max_memory_usage,
                gpu_memory_limit,
                (max_memory_usage / gpu_memory_limit) * 100,
            )
        )
        print(
            "Maximum GPU power usage: %dW / %dW (%.2f%%)"
            % (
                max_power_usage,
                gpu_power_limit,
                (max_power_usage / gpu_power_limit) * 100,
            )
        )
    else:
        logger.info("Measuring maximum increase of memory usage.")
        max_usage = memory_usage(func, max_usage=True, interval=interval)
        print("Maximum increase of RAM memory usage: %d MiB" % max_usage)


if __name__ == "__main__":
    measure_memory(inference)
