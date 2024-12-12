[![CI](https://github.com/SYSTRAN/faster-whisper/workflows/CI/badge.svg)](https://github.com/SYSTRAN/faster-whisper/actions?query=workflow%3ACI) [![PyPI version](https://badge.fury.io/py/faster-whisper.svg)](https://badge.fury.io/py/faster-whisper)

# Faster Whisper transcription with CTranslate2

**faster-whisper** is a reimplementation of OpenAI's Whisper model using [CTranslate2](https://github.com/OpenNMT/CTranslate2/), which is a fast inference engine for Transformer models.

This implementation is up to 4 times faster than [openai/whisper](https://github.com/openai/whisper) for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU.

## Benchmark

### Whisper

For reference, here's the time and memory usage that are required to transcribe [**13 minutes**](https://www.youtube.com/watch?v=0u7tTptBo9I) of audio using different implementations:

* [openai/whisper](https://github.com/openai/whisper)@[v20240930](https://github.com/openai/whisper/tree/v20240930)
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)@[v1.7.2](https://github.com/ggerganov/whisper.cpp/tree/v1.7.2)
* [transformers](https://github.com/huggingface/transformers)@[v4.46.3](https://github.com/huggingface/transformers/tree/v4.46.3)
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper)@[v1.1.0](https://github.com/SYSTRAN/faster-whisper/tree/v1.1.0)

### Large-v2 model on GPU

| Implementation | Precision | Beam size | Time | VRAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper | fp16 | 5 | 2m23s | 4708MB |
| whisper.cpp (Flash Attention) | fp16 | 5 | 1m05s | 4127MB |
| transformers (SDPA)[^1] | fp16 | 5 | 1m52s | 4960MB |
| faster-whisper | fp16 | 5 | 1m03s | 4525MB |
| faster-whisper (`batch_size=8`) | fp16 | 5 | 17s | 6090MB |
| faster-whisper | int8 | 5 | 59s | 2926MB |
| faster-whisper (`batch_size=8`) | int8 | 5 | 16s | 4500MB |

### distil-whisper-large-v3 model on GPU

| Implementation | Precision | Beam size | Time | YT Commons WER |
| --- | --- | --- | --- | --- |
| transformers (SDPA) (`batch_size=16`) | fp16 | 5 | 46m12s | 14.801 |
| faster-whisper (`batch_size=16`) | fp16 | 5 | 25m50s | 13.527 |

*GPU Benchmarks are Executed with CUDA 12.4 on a NVIDIA RTX 3070 Ti 8GB.*
[^1]: transformers OOM for any batch size > 1

### Small model on CPU

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper | fp32 | 5 | 6m58s | 2335MB |
| whisper.cpp | fp32 | 5 | 2m05s | 1049MB |
| whisper.cpp (OpenVINO) | fp32 | 5 | 1m45s | 1642MB |
| faster-whisper | fp32 | 5 | 2m37s | 2257MB |
| faster-whisper (`batch_size=8`) | fp32 | 5 | 1m06s | 4230MB |
| faster-whisper | int8 | 5 | 1m42s | 1477MB |
| faster-whisper (`batch_size=8`) | int8 | 5 | 51s | 3608MB |

*Executed with 8 threads on an Intel Core i7-12700K.*


## Requirements

* Python 3.9 or greater

Unlike openai-whisper, FFmpeg does **not** need to be installed on the system. The audio is decoded with the Python library [PyAV](https://github.com/PyAV-Org/PyAV) which bundles the FFmpeg libraries in its package.

### GPU

GPU execution requires the following NVIDIA libraries to be installed:

* [cuBLAS for CUDA 12](https://developer.nvidia.com/cublas)
* [cuDNN 9 for CUDA 12](https://developer.nvidia.com/cudnn)

**Note**: The latest versions of `ctranslate2` only support CUDA 12 and cuDNN 9. For CUDA 11 and cuDNN 8, the current workaround is downgrading to the `3.24.0` version of `ctranslate2`, for CUDA 12 and cuDNN 8, downgrade to the `4.4.0` version of `ctranslate2`, (This can be done with `pip install --force-reinstall ctranslate2==4.4.0` or specifying the version in a `requirements.txt`).

There are multiple ways to install the NVIDIA libraries mentioned above. The recommended way is described in the official NVIDIA documentation, but we also suggest other installation methods below. 

<details>
<summary>Other installation methods (click to expand)</summary>


**Note:** For all these methods below, keep in mind the above note regarding CUDA versions. Depending on your setup, you may need to install the _CUDA 11_ versions of libraries that correspond to the CUDA 12 libraries listed in the instructions below.

#### Use Docker

The libraries (cuBLAS, cuDNN) are installed in this official NVIDIA CUDA Docker images: `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`.

#### Install with `pip` (Linux only)

On Linux these libraries can be installed with `pip`. Note that `LD_LIBRARY_PATH` must be set before launching Python.

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

#### Download the libraries from Purfview's repository (Windows & Linux)

Purfview's [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) provides the required NVIDIA libraries for Windows & Linux in a [single archive](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs). Decompress the archive and place the libraries in a directory included in the `PATH`.

</details>

## Installation

The module can be installed from [PyPI](https://pypi.org/project/faster-whisper/):

```bash
pip install faster-whisper
```

<details>
<summary>Other installation methods (click to expand)</summary>

### Install the master branch

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
```

### Install a specific commit

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/a4f1cc8f11433e454c3934442b5e1a4ed5e865c3.tar.gz"
```

</details>

## Usage

### Faster-whisper

```python
from faster_whisper import WhisperModel

model_size = "large-v3"

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

### Batched Transcription
The following code snippet illustrates how to run batched transcription on an example audio file. `BatchedInferencePipeline.transcribe` is a drop-in replacement for `WhisperModel.transcribe`

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel("turbo", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)
segments, info = batched_model.transcribe("audio.mp3", batch_size=16)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

### Faster Distil-Whisper

The Distil-Whisper checkpoints are compatible with the Faster-Whisper package. In particular, the latest [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)
checkpoint is intrinsically designed to work with the Faster-Whisper transcription algorithm. The following code snippet 
demonstrates how to run inference with distil-large-v3 on a specified audio file:

```python
from faster_whisper import WhisperModel

model_size = "distil-large-v3"

model = WhisperModel(model_size, device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5, language="en", condition_on_previous_text=False)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

For more information about the distil-large-v3 model, refer to the original [model card](https://huggingface.co/distil-whisper/distil-large-v3).

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

The default behavior is conservative and only removes silence longer than 2 seconds. See the available VAD parameters and default values in the [source code](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py). They can be customized with the dictionary argument `vad_parameters`:

```python
segments, _ = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
```
Vad filter is enabled by default for batched transcription.

### Logging

The library logging level can be configured like this:

```python
import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
```

### Going further

See more model and transcription options in the [`WhisperModel`](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py) class implementation.

## Community integrations

Here is a non exhaustive list of open-source projects using faster-whisper. Feel free to add your project to the list!


* [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) is an OpenAI compatible server using `faster-whisper`. It's easily deployable with Docker, works with OpenAI SDKs/CLI, supports streaming, and live transcription.
* [WhisperX](https://github.com/m-bain/whisperX) is an award-winning Python library that offers speaker diarization and accurate word-level timestamps using wav2vec2 alignment
* [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2) is a command line client based on faster-whisper and compatible with the original client from openai/whisper.
* [whisper-diarize](https://github.com/MahmoudAshraf97/whisper-diarization) is a speaker diarization tool that is based on faster-whisper and NVIDIA NeMo.
* [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) Standalone CLI executables of faster-whisper for Windows, Linux & macOS. 
* [asr-sd-pipeline](https://github.com/hedrergudene/asr-sd-pipeline) provides a scalable, modular, end to end multi-speaker speech to text solution implemented using AzureML pipelines.
* [Open-Lyrics](https://github.com/zh-plus/Open-Lyrics) is a Python library that transcribes voice files using faster-whisper, and translates/polishes the resulting text into `.lrc` files in the desired language using OpenAI-GPT.
* [wscribe](https://github.com/geekodour/wscribe) is a flexible transcript generation tool supporting faster-whisper, it can export word level transcript and the exported transcript then can be edited with [wscribe-editor](https://github.com/geekodour/wscribe-editor)
* [aTrain](https://github.com/BANDAS-Center/aTrain) is a graphical user interface implementation of faster-whisper developed at the BANDAS-Center at the University of Graz for transcription and diarization in Windows ([Windows Store App](https://apps.microsoft.com/detail/atrain/9N15Q44SZNS2)) and Linux.
* [Whisper-Streaming](https://github.com/ufal/whisper_streaming) implements real-time mode for offline Whisper-like speech-to-text models with faster-whisper as the most recommended back-end. It implements a streaming policy with self-adaptive latency based on the actual source complexity, and demonstrates the state of the art.
* [WhisperLive](https://github.com/collabora/WhisperLive) is a nearly-live implementation of OpenAI's Whisper which uses faster-whisper as the backend to transcribe audio in real-time.
* [Faster-Whisper-Transcriber](https://github.com/BBC-Esq/ctranslate2-faster-whisper-transcriber) is a simple but reliable voice transcriber that provides a user-friendly interface.
* [Open-dubbing](https://github.com/softcatala/open-dubbing) is open dubbing is an AI dubbing system which uses machine learning models to automatically translate and synchronize audio dialogue into different languages.

## Model conversion

When loading a model from its size such as `WhisperModel("large-v3")`, the corresponding CTranslate2 model is automatically downloaded from the [Hugging Face Hub](https://huggingface.co/Systran).

We also provide a script to convert any Whisper models compatible with the Transformers library. They could be the original OpenAI models or user fine-tuned models.

For example the command below converts the [original "large-v3" Whisper model](https://huggingface.co/openai/whisper-large-v3) and saves the weights in FP16:

```bash
pip install transformers[torch]>=4.23

ct2-transformers-converter --model openai/whisper-large-v3 --output_dir whisper-large-v3-ct2
--copy_files tokenizer.json preprocessor_config.json --quantization float16
```

* The option `--model` accepts a model name on the Hub or a path to a model directory.
* If the option `--copy_files tokenizer.json` is not used, the tokenizer configuration is automatically downloaded when the model is loaded later.

Models can also be converted from the code. See the [conversion API](https://opennmt.net/CTranslate2/python/ctranslate2.converters.TransformersConverter.html).

### Load a converted model

1. Directly load the model from a local directory:
```python
model = faster_whisper.WhisperModel("whisper-large-v3-ct2")
```

2. [Upload your model to the Hugging Face Hub](https://huggingface.co/docs/transformers/model_sharing#upload-with-the-web-interface) and load it from its name:
```python
model = faster_whisper.WhisperModel("username/whisper-large-v3-ct2")
```

## Comparing performance against other implementations

If you are comparing the performance against other Whisper implementations, you should make sure to run the comparison with similar settings. In particular:

* Verify that the same transcription options are used, especially the same beam size. For example in openai/whisper, `model.transcribe` uses a default beam size of 1 but here we use a default beam size of 5.
* Transcription speed is closely affected by the number of words in the transcript, so ensure that other implementations have a similar WER (Word Error Rate) to this one.
* When running on CPU, make sure to set the same number of threads. Many frameworks will read the environment variable `OMP_NUM_THREADS`, which can be set when running your script:

```bash
OMP_NUM_THREADS=4 python3 my_script.py
```
