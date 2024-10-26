from faster_whisper.utils import download_model

model = download_model(size_or_id="large-v3", local_files_only=False)
print(f"Model large-v3 weights loaded on GPU")
