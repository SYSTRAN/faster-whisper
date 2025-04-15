import argparse
import json
import os

from datasets import load_dataset
from jiwer import wer
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from faster_whisper import WhisperModel

parser = argparse.ArgumentParser(description="WER benchmark")
parser.add_argument(
    "--audio_numb",
    type=int,
    default=None,
    help="Specify the number of validation audio files in the dataset."
    " Set to None to retrieve all audio files.",
)
args = parser.parse_args()

model_path = "large-v3"
model = WhisperModel(model_path, device="cuda")

# load the dataset with streaming mode
dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)

with open(os.path.join(os.path.dirname(__file__), "normalizer.json"), "r") as f:
    normalizer = EnglishTextNormalizer(json.load(f))


def inference(batch):
    batch["transcription"] = []
    for sample in batch["audio"]:
        segments, info = model.transcribe(sample["array"], language="en")
        batch["transcription"].append("".join([segment.text for segment in segments]))
    batch["reference"] = batch["text"]
    return batch


dataset = dataset.map(function=inference, batched=True, batch_size=16)

all_transcriptions = []
all_references = []

# iterate over the dataset and run inference
for i, result in tqdm(enumerate(dataset), desc="Evaluating..."):
    all_transcriptions.append(result["transcription"])
    all_references.append(result["reference"])
    if args.audio_numb and i == (args.audio_numb - 1):
        break

# normalize predictions and references
all_transcriptions = [normalizer(transcription) for transcription in all_transcriptions]
all_references = [normalizer(reference) for reference in all_references]

# compute the WER metric
word_error_rate = 100 * wer(hypothesis=all_transcriptions, reference=all_references)
print("WER: %.3f" % word_error_rate)
