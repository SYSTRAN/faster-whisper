# from faster_whisper import WhisperModel
from jimmy_whisper.transcribe import WhisperModel
from jimmy_whisper.audio import decode_audio
import time
# import whisper
# import ffmpeg
# from typing import BinaryIO
# import numpy as np
# import torchaudio

# SAMPLE_RATE = 16000
# def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
#     """
#     Open an audio file object and read as mono waveform, resampling as necessary.
#     Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
#     Parameters
#     ----------
#     file: BinaryIO
#         The audio file like object
#     encode: Boolean
#         If true, encode audio stream to WAV before sending to whisper
#     sr: int
#         The sample rate to resample the audio if necessary
#     Returns
#     -------
#     A NumPy array containing the audio waveform, in float32 dtype.
#     """
#     if encode:
#         try:
#             # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
#             # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
#             out, _ = (
#                 ffmpeg.input("pipe:", threads=0)
#                 .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
#                 .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
#             )
#         except ffmpeg.Error as e:
#             raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
#     else:
#         out = file.read()
#
#     return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


model = None

def init_test(file, model_size):
    global model
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    time.sleep(5)

    begin_time = time.time()
    language = "en"
    segments, info = model.transcribe(file, beam_size=5, language=language)
    total_text = ""
    for segment in segments:
        total_text += " " + segment.text
    print(f"test0 cost_time:{time.time() - begin_time}")
def test1(filepath, target_language):
    begin_time = time.time()
    language = target_language
    segments, info = model.transcribe(filepath, beam_size=5, language=language)
    total_text = ""
    for segment in segments:
        total_text += " " + segment.text
    lang = info.language
    lang_prob = info.language_probability
    cost_time = time.time() - begin_time
    print(f"test1 lang:{lang} lang_prob:{lang_prob} cost_time:{cost_time} text:{total_text}")
    return lang, lang_prob, cost_time

def test2(filepath):
    begin_time = time.time()
    segments, info = model.transcribe(filepath, beam_size=5)
    total_text = ""
    for segment in segments:
        total_text += " " + segment.text
    lang = info.language
    lang_prob = info.language_probability
    cost_time = time.time() - begin_time
    print(f"test2 lang:{lang} lang_prob:{lang_prob} cost_time:{time.time() - begin_time} text:{total_text}")
    return lang, lang_prob, cost_time

def test3(filepath, target_language, target_language_probability_threshold):
    begin_time = time.time()
    segments, info = model.transcribe_after_detect_language(filepath, beam_size=5, target_language=target_language, target_language_probability_threshold=target_language_probability_threshold)
    total_text = ""
    if segments is not None:
        for segment in segments:
            total_text += " " + segment.text
    lang = info.language
    lang_prob = info.language_probability
    cost_time = time.time() - begin_time
    print(f"test3 lang:{lang} lang_prob:{lang_prob} cost_time:{time.time() - begin_time} text:{total_text}")
    return lang, lang_prob, cost_time
def test4(filepath):
    begin_time = time.time()
    waveform = decode_audio(filepath)
    lang, lang_prob, all_language_probs = model.detect_language(waveform)
    cost_time = time.time() - begin_time
    print(f"test4 lang:{lang} lang_prob:{lang_prob} cost_time:{time.time() - begin_time} text:")
    return lang, lang_prob, cost_time

def total_test(filepath, model_size, count, target_language):
    test1_time = 0
    test2_time = 0
    test3_time = 0
    test4_time = 0
    test5_time = 0
    for i in range(count):
        lang, lang_prob, cost_time = test1(filepath, target_language)
        test1_time += cost_time
        lang, lang_prob, cost_time = test2(filepath)
        test2_time += cost_time
        lang, lang_prob, cost_time = test3(filepath, target_language, 0.99999)
        test3_time += cost_time
        lang, lang_prob, cost_time = test3(filepath, target_language, 0.1)
        test4_time += cost_time
        lang, lang_prob, cost_time = test4(filepath)
        test5_time += cost_time
    print(f"transcribe with en                            {model_size} {filepath} avg_time:{test1_time/count}")
    print(f"transcribe with detect                        {model_size} {filepath} avg_time:{test2_time/count}")
    print(f"transcribe_after_detect_language without asr  {model_size} {filepath} avg_time:{test3_time/count}")
    print(f"transcribe_after_detect_language with asr     {model_size} {filepath} avg_time:{test4_time/count}")
    print(f"detect_language                               {model_size} {filepath} avg_time:{test5_time/count}")

if __name__ == "__main__":
    init_test("/data/jimmy/test_data/en_1.wav", "large-v3")
    total_test("/data/jimmy/test_data/en_2.wav", "large-v3", 10, "en")
    total_test("/data/jimmy/test_data/en_3.wav", "large-v3", 10, "en")
    total_test("/data/jimmy/test_data/en_1.wav", "large-v3", 10, "en")

    init_test("/data/jimmy/test_data/en_1.wav", "medium")
    total_test("/data/jimmy/test_data/en_2.wav", "medium", 10, "en")
    total_test("/data/jimmy/test_data/en_3.wav", "medium", 10, "en")
    total_test("/data/jimmy/test_data/en_1.wav", "medium", 10, "en")