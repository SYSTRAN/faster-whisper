# Faster Whisper transcription with CTranslate2

This repository demonstrates how to implement the Whisper transcription using [CTranslate2](https://github.com/OpenNMT/CTranslate2/), which is a fast inference engine for Transformer models.

This implementation is about 4 times faster than [openai/whisper](https://github.com/openai/whisper) for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU.

## Installation

```bash
pip install -e .[conversion]
```

The model conversion requires the modules `transformers` and `torch` which are installed by the `[conversion]` requirement. Once a model is converted, these modules are no longer needed and the installation could be simplified to:

```bash
pip install -e .
```

### GPU support

GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html).

**Note for Windows users:** the published Windows wheels are currently not built with cuDNN, which is required to run the Whisper model on GPU. Consequently GPU execution is currently not possible on Windows out of the box. If you feel adventurous, a possible workaround is to [compile from sources](https://opennmt.net/CTranslate2/installation.html#install-from-sources).

## Usage

### Model conversion

A Whisper model should be first converted into the CTranslate2 format. For example the command below converts the "medium" Whisper model and saves the weights in FP16:

```bash
ct2-transformers-converter --model openai/whisper-medium --output_dir whisper-medium-ct2 --quantization float16
```

If needed, models can also be converted from the code. See the [conversion API](https://opennmt.net/CTranslate2/python/ctranslate2.converters.TransformersConverter.html).

### Transcription

```python
from faster_whisper import WhisperModel

model_path = "whisper-medium-ct2/"

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

## Comparing performance against other implementations

If you are comparing the performance against other Whisper implementations, you should make sure to run the comparison with similar settings. In particular:

* Verify that the same transcription options are used, especially the same beam size. For example in openai/whisper, `model.transcribe` uses a default beam size of 1 but here we use a default beam size of 5.
* When running on CPU, make sure to set the same number of threads. Many frameworks will read the environment variable `OMP_NUM_THREADS`, which can be set when running your script:

```bash
OMP_NUM_THREADS=4 python3 my_script.py
```
