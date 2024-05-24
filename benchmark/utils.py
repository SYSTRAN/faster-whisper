import logging

from threading import Thread
from typing import Optional

from faster_whisper import WhisperModel

model_path = "large-v3"
model = WhisperModel(model_path, device="cuda")


def inference():
    segments, info = model.transcribe("benchmark.m4a", language="fr")
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class MyThread(Thread):
    def __init__(self, func, params):
        super(MyThread, self).__init__()
        self.func = func
        self.params = params
        self.result = None

    def run(self):
        self.result = self.func(*self.params)

    def get_result(self):
        return self.result
