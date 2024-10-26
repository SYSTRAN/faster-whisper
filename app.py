from io import BytesIO

import modal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import modal.gpu

from faster_whisper import WhisperModel

web_app = FastAPI()

model = None


def initialize_model():
    global model
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print(f"Model {model_size} weights loaded on GPU")


@web_app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    supported_audio_extensions = (".mp3", ".wav", ".flac")
    if not file.filename.lower().endswith(supported_audio_extensions):
        raise HTTPException(status_code=400, detail="File type not supported. Please upload MP3, WAV, or FLAC files.")

    audio_data = BytesIO(await file.read())

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized")

    segments, info = model.transcribe(audio_data, beam_size=5)

    response = {
        "language": info.language,
        "language_probability": info.language_probability,
        "transcription": [{"start": segment.start, "end": segment.end, "text": segment.text} for segment in segments],
    }

    return JSONResponse(content=response)


app = modal.App("faster-whisper-server")

image = (
    modal.Image.from_registry("nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04", add_python="3.11")
    .pip_install(
        "ctranslate2==4.4.0",
        "huggingface_hub>=0.13",
        "tokenizers>=0.13,<1",
        "onnxruntime>=1.14,<2",
        "pyannote-audio>=3.1.1",
        "torch",
        "av>=11",
        "google",
        "tqdm",
        "fastapi==0.115.3",
        "faster-whisper==1.0.3",
        "uvicorn==0.32.0",
        "python-multipart",
        "nvidia-cublas-cu11",
        "nvidia-cudnn-cu11",
    )
    .copy_local_file("download_weights.py", "/root/download_weights.py")
    .copy_local_file("helper.sh", "/root/helper.sh")
    .run_commands("python3 /root/download_weights.py")
    .run_commands("bash /root/helper.sh")
)


@app.function(image=image, gpu=[modal.gpu.H100()], concurrency_limit=1, container_idle_timeout=600)
@modal.asgi_app()
def fastapi_wrapper():
    initialize_model()
    return web_app
