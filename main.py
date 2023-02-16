from faster_whisper import WhisperModel
import os
os.environ["LD_LIBRARY_PATH"] = "D:/Work/CTranslate2/bin"

os.add_dll_directory("D:/Work/CTranslate2/bin")

model_path = "whisper-medium-ct2/"

# Run on GPU with FP16
model = WhisperModel(model_path, device="cuda")

# or run on GPU with INT8
# model = WhisperModel(model_path, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_path, device="cpu", compute_type="int8")

transcription, info = model.transcribe("Sample.mp4", beam_size=1)
# transcription, info = model.transcribe("Vocaroo.mp3", beam_size=1)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in transcription["segments"]:
    print("[%fs -> %fs] %s" % (segment["start"], segment["end"], segment["text"]))


for segment in transcription["segments"]:
    for word in segment["words"]:
        print("[%fs -> %fs] %s" % (word["start"], word["end"], word["text"]))