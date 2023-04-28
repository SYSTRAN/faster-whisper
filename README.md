[![CI](https://github.com/guillaumekln/faster-whisper/workflows/CI/badge.svg)](https://github.com/guillaumekln/faster-whisper/actions?query=workflow%3ACI) [![PyPI version](https://badge.fury.io/py/faster-whisper.svg)](https://badge.fury.io/py/faster-whisper)

# Faster Whisper transcription with CTranslate2

**faster-whisper** is a reimplementation of OpenAI's Whisper model using [CTranslate2](https://github.com/OpenNMT/CTranslate2/), which is a fast inference engine for Transformer models.

This implementation is up to 4 times faster than [openai/whisper](https://github.com/openai/whisper) for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU.

## Benchmark

For reference, here's the time and memory usage that are required to transcribe [**13 minutes**](https://www.youtube.com/watch?v=0u7tTptBo9I) of audio using different implementations:

* [openai/whisper](https://github.com/openai/whisper)@[6dea21fd](https://github.com/openai/whisper/commit/6dea21fd7f7253bfe450f1e2512a0fe47ee2d258)
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)@[3b010f9](https://github.com/ggerganov/whisper.cpp/commit/3b010f9bed9a6068609e9faf52383aea792b0362)
* [faster-whisper](https://github.com/guillaumekln/faster-whisper)@[cce6b53e](https://github.com/guillaumekln/faster-whisper/commit/cce6b53e4554f71172dad188c45f10fb100f6e3e)

### Large-v2 model on GPU

| Implementation | Precision | Beam size | Time | Max. GPU memory | Max. CPU memory |
| --- | --- | --- | --- | --- | --- |
| openai/whisper | fp16 | 5 | 4m30s | 11325MB | 9439MB |
| faster-whisper | fp16 | 5 | 54s | 4755MB | 3244MB |
| faster-whisper | int8 | 5 | 59s | 3091MB | 3117MB |

*Executed with CUDA 11.7.1 on a NVIDIA Tesla V100S.*

### Small model on CPU

| Implementation | Precision | Beam size | Time | Max. memory |
| --- | --- | --- | --- | --- |
| openai/whisper | fp32 | 5 | 10m31s | 3101MB |
| whisper.cpp | fp32 | 5 | 17m42s | 1581MB |
| whisper.cpp | fp16 | 5 | 12m39s | 873MB |
| faster-whisper | fp32 | 5 | 2m44s | 1675MB |
| faster-whisper | int8 | 5 | 2m04s | 995MB |

*Executed with 8 threads on a Intel(R) Xeon(R) Gold 6226R.*

## Installation

The module can be installed from [PyPI](https://pypi.org/project/faster-whisper/):

```bash
pip install faster-whisper
```

**Other installation methods:**

```bash
# Install the master branch:
pip install --force-reinstall "faster-whisper @ https://github.com/guillaumekln/faster-whisper/archive/refs/heads/master.tar.gz"

# Install a specific commit:
pip install --force-reinstall "faster-whisper @ https://github.com/guillaumekln/faster-whisper/archive/a4f1cc8f11433e454c3934442b5e1a4ed5e865c3.tar.gz"

# Install for development:
git clone https://github.com/guillaumekln/faster-whisper.git
pip install -e faster-whisper/
```

### GPU support

GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html).

## Usage

```python
from faster_whisper import WhisperModel

model_size = "large-v2"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

**Warning:** `segments` is a *generator* so the transcription only starts when you iterate over it. The transcription can be run to completion by gathering the segments in a list or a `for` loop:

```python
segments, _ = model.transcribe("audio.mp3")
segments = list(segments)  # The transcription will actually run here.
```

### Word-level timestamps

```python
segments, _ = model.transcribe("audio.mp3", word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
```

### VAD filter

The library integrates the [Silero VAD](https://github.com/snakers4/silero-vad) model to filter out parts of the audio without speech:

```python
segments, _ = model.transcribe("audio.mp3", vad_filter=True)
```

The default behavior is conservative and only removes silence longer than 2 seconds. See the available VAD parameters and default values in the function [`get_speech_timestamps`](https://github.com/guillaumekln/faster-whisper/blob/master/faster_whisper/vad.py). They can be customized with the dictionary argument `vad_parameters`:

```python
segments, _ = model.transcribe("audio.mp3", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
```

### Logging

The library logging level can be configured like this:

```python
import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
```

### Going further

See more model and transcription options in the [`WhisperModel`](https://github.com/guillaumekln/faster-whisper/blob/master/faster_whisper/transcribe.py) class implementation.

## Community integrations

Here is a non exhaustive list of open-source projects using *faster-whisper*. Feel free to add your project to the list!

* [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2) is a command line client based on `faster-whisper` and compatible with the original client from openai/whisper.

* [whisper-diarize](https://github.com/MahmoudAshraf97/whisper-diarization) is a speaker diarization tool that is based on `faster-whisper` and nvidia nemo.

## Model conversion

When loading a model from its size such as `WhisperModel("large-v2")`, the correspondig CTranslate2 model is automatically downloaded from the [Hugging Face Hub](https://huggingface.co/guillaumekln).

We also provide a script to convert any Whisper models compatible with the Transformers library. They could be the original OpenAI models or user fine-tuned models.

For example the command below converts the [original "large-v2" Whisper model](https://huggingface.co/openai/whisper-large-v2) and saves the weights in FP16:

```bash
pip install transformers[torch]>=4.23

ct2-transformers-converter --model openai/whisper-large-v2 --output_dir whisper-large-v2-ct2 \
    --copy_files tokenizer.json --quantization float16
```

* The option `--model` accepts a model name on the Hub or a path to a model directory.
* If the option `--copy_files tokenizer.json` is not used, the tokenizer configuration is automatically downloaded when the model is loaded later.

Models can also be converted from the code. See the [conversion API](https://opennmt.net/CTranslate2/python/ctranslate2.converters.TransformersConverter.html).

## Comparing performance against other implementations

If you are comparing the performance against other Whisper implementations, you should make sure to run the comparison with similar settings. In particular:

* Verify that the same transcription options are used, especially the same beam size. For example in openai/whisper, `model.transcribe` uses a default beam size of 1 but here we use a default beam size of 5.
* When running on CPU, make sure to set the same number of threads. Many frameworks will read the environment variable `OMP_NUM_THREADS`, which can be set when running your script:

```bash
OMP_NUM_THREADS=4 python3 my_script.py
```
