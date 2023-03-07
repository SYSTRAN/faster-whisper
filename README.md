# Faster Whisper transcription with CTranslate2

This repository demonstrates how to implement the Whisper transcription using [CTranslate2](https://github.com/OpenNMT/CTranslate2/), which is a fast inference engine for Transformer models.

This implementation is up to 4 times faster than [openai/whisper](https://github.com/openai/whisper) for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU.

## Benchmark

For reference, here's the time and memory usage that are required to transcribe **13 minutes** of audio using different implementations:

* [openai/whisper](https://github.com/openai/whisper)@[7858aa9](https://github.com/openai/whisper/commit/7858aa9c08d98f75575035ecd6481f462d66ca27)
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)@[3b010f9](https://github.com/ggerganov/whisper.cpp/commit/3b010f9bed9a6068609e9faf52383aea792b0362)
* [faster-whisper](https://github.com/guillaumekln/faster-whisper)@[cda834c](https://github.com/guillaumekln/faster-whisper/commit/cda834c8ea76c2cab9da19031815c1e937a88c7f)

### Large model on GPU

| Implementation | Precision | Beam size | Time | Max. GPU memory | Max. CPU memory |
| --- | --- | --- | --- | --- | --- |
| openai/whisper | fp16 | 5 | 4m30s | 11413MB | 9553MB |
| faster-whisper | fp16 | 5 | 1m02s | 4659MB | 3244MB |

*Executed with CUDA 11.7.1 on a NVIDIA Tesla V100S.*

### Small model on CPU

| Implementation | Precision | Beam size | Time | Max. memory |
| --- | --- | --- | --- | --- |
| openai/whisper | fp32 | 5 | 10m39s | 2850MB |
| whisper.cpp | fp32 | 5 | 17m42s | 1581MB |
| whisper.cpp | fp16 | 5 | 12m39s | 873MB |
| faster-whisper | fp32 | 5 | 2m53s | 1482MB |
| faster-whisper | int8 | 5 | 2m01s | 1008MB |

*Executed with 8 threads on a Intel(R) Xeon(R) Gold 6226R.*

## Installation

```bash
pip install -e .[conversion]
```

The model conversion requires the modules `transformers` and `torch` which are installed by the `[conversion]` requirement. Once a model is converted, these modules are no longer needed and the installation could be simplified to:

```bash
pip install -e .
```

It is also possible to install the module without cloning the Git repository:

```bash
# Install the master branch:
pip install "faster-whisper @ https://github.com/guillaumekln/faster-whisper/archive/refs/heads/master.tar.gz"

# Install a specific commit:
pip install "faster-whisper @ https://github.com/guillaumekln/faster-whisper/archive/a4f1cc8f11433e454c3934442b5e1a4ed5e865c3.tar.gz"
```

### GPU support

GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html).

## Usage

### Model conversion

A Whisper model should be first converted into the CTranslate2 format. We provide a script to download and convert models from the [Hugging Face model repository](https://huggingface.co/models?sort=downloads&search=whisper).

For example the command below converts the "large-v2" Whisper model and saves the weights in FP16:

```bash
ct2-transformers-converter --model openai/whisper-large-v2 --output_dir whisper-large-v2-ct2 \
    --copy_files tokenizer.json --quantization float16
```

If the option `--copy_files tokenizer.json` is not used, the tokenizer configuration is automatically downloaded when the model is loaded later.

Models can also be converted from the code. See the [conversion API](https://opennmt.net/CTranslate2/python/ctranslate2.converters.TransformersConverter.html).

### Transcription

```python
from faster_whisper import WhisperModel

model_path = "whisper-large-v2-ct2/"

# Run on GPU with FP16
model = WhisperModel(model_path, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_path, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_path, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

See more model and transcription options in the [`WhisperModel`](https://github.com/guillaumekln/faster-whisper/blob/master/faster_whisper/transcribe.py) class implementation.

## Comparing performance against other implementations

If you are comparing the performance against other Whisper implementations, you should make sure to run the comparison with similar settings. In particular:

* Verify that the same transcription options are used, especially the same beam size. For example in openai/whisper, `model.transcribe` uses a default beam size of 1 but here we use a default beam size of 5.
* When running on CPU, make sure to set the same number of threads. Many frameworks will read the environment variable `OMP_NUM_THREADS`, which can be set when running your script:

```bash
OMP_NUM_THREADS=4 python3 my_script.py
```
