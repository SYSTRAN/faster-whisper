import argparse
import json
import os

from io import BytesIO

from datasets import load_dataset
from evaluate import load
from pytubefix import YouTube
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio


def url_to_audio(row):
    buffer = BytesIO()
    yt = YouTube(row["link"])
    video = (
        yt.streams.filter(only_audio=True, mime_type="audio/mp4")
        .order_by("bitrate")
        .desc()
        .first()
    )
    video.stream_to_buffer(buffer)
    buffer.seek(0)
    row["audio"] = decode_audio(buffer)
    return row


parser = argparse.ArgumentParser(description="WER benchmark")
parser.add_argument(
    "--audio_numb",
    type=int,
    default=None,
    help="Specify the number of validation audio files in the dataset."
    " Set to None to retrieve all audio files.",
)
args = parser.parse_args()

# define the evaluation metric
wer_metric = load("wer")

with open(os.path.join(os.path.dirname(__file__), "normalizer.json"), "r") as f:
    normalizer = EnglishTextNormalizer(json.load(f))

dataset = load_dataset("mobiuslabsgmbh/youtube-commons-asr-eval", streaming=True).map(
    url_to_audio
)
dataset = iter(
    DataLoader(dataset["test"], batch_size=1, prefetch_factor=4, num_workers=2)
)

model = WhisperModel("large-v3", device="cuda")
pipeline = BatchedInferencePipeline(model, device="cuda")


all_transcriptions = []
all_references = []
# iterate over the dataset and run inference
for i, row in tqdm(enumerate(dataset), desc="Evaluating..."):
    result, info = pipeline.transcribe(
        row["audio"][0],
        batch_size=8,
        word_timestamps=False,
        without_timestamps=True,
    )

    all_transcriptions.append("".join(segment.text for segment in result))
    all_references.append(row["text"][0])
    if args.audio_numb and i == (args.audio_numb - 1):
        break

# normalize predictions and references
all_transcriptions = [normalizer(transcription) for transcription in all_transcriptions]
all_references = [normalizer(reference) for reference in all_references]

# compute the WER metric
wer = 100 * wer_metric.compute(
    predictions=all_transcriptions, references=all_references
)
print("WER: %.3f" % wer)
