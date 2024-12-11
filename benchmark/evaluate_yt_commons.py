import argparse
import json
import os

from io import BytesIO

from datasets import load_dataset
from jiwer import wer
from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio


def url_to_audio(row):
    buffer = BytesIO()
    yt = YouTube(row["link"])
    try:
        video = (
            yt.streams.filter(only_audio=True, mime_type="audio/mp4")
            .order_by("bitrate")
            .desc()
            .last()
        )
        video.stream_to_buffer(buffer)
        buffer.seek(0)
        row["audio"] = decode_audio(buffer)
    except VideoUnavailable:
        print(f'Failed to download: {row["link"]}')
        row["audio"] = []
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

with open(os.path.join(os.path.dirname(__file__), "normalizer.json"), "r") as f:
    normalizer = EnglishTextNormalizer(json.load(f))

dataset = load_dataset("mobiuslabsgmbh/youtube-commons-asr-eval", streaming=True).map(
    url_to_audio
)
model = WhisperModel("large-v3", device="cuda")
pipeline = BatchedInferencePipeline(model, device="cuda")


all_transcriptions = []
all_references = []
# iterate over the dataset and run inference
for i, row in tqdm(enumerate(dataset["test"]), desc="Evaluating..."):
    if not row["audio"]:
        continue
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
word_error_rate = 100 * wer(hypothesis=all_transcriptions, reference=all_references)
print("WER: %.3f" % word_error_rate)
