import itertools
import json
import logging
import os
import random
import zlib

from collections import Counter, defaultdict
from inspect import signature
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union

import ctranslate2
import numpy as np
import tokenizers
import torch

from pyannote.audio import Model
from tqdm import tqdm

from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
from faster_whisper.utils import (
    download_model,
    format_timestamp,
    get_assets_path,
    get_end,
    get_logger,
)
from faster_whisper.vad import (
    SpeechTimestampsMap,
    VadOptions,
    VoiceActivitySegmentation,
    collect_chunks,
    get_speech_timestamps,
    merge_chunks,
)


class Word(NamedTuple):
    start: float
    end: float
    word: str
    probability: float


class Segment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]
    temperature: Optional[float] = 1.0


# Added additional parameters for multilingual videos and fixes below
class TranscriptionOptions(NamedTuple):
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
    log_prob_low_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperatures: List[float]
    initial_prompt: Optional[Union[str, Iterable[int]]]
    prefix: Optional[str]
    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    without_timestamps: bool
    max_initial_timestamp: float
    word_timestamps: bool
    prepend_punctuations: str
    append_punctuations: str
    multilingual: bool
    output_language: Optional[str]
    max_new_tokens: Optional[int]
    clip_timestamps: Union[str, List[float]]
    hallucination_silence_threshold: Optional[float]
    hotwords: Optional[str]


class TranscriptionInfo(NamedTuple):
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionOptions
    vad_options: VadOptions


# The code below is originally from HF pipeline and is used in whisper-x
# (https://github.com/m-bain/whisperX) and adapted for faster_whisper


class BatchedInferencePipeline:
    """
    Huggingface Pipeline wrapper for WhisperModel.
    Copyright (c) 2022, Max Bain
    All rights reserved.
    Modified by Mobius Labs GmbH
    """

    def __init__(
        self,
        model,
        use_vad_model: bool = True,
        options: Optional[NamedTuple] = None,
        tokenizer=None,
        chunk_length: int = 30,
        vad_device: Union[int, str, "torch.device"] = "auto",
        vad_onset: float = 0.500,
        vad_offset: float = 0.363,
        language: Optional[str] = None,
    ):
        self.model: WhisperModel = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.use_vad_model = use_vad_model
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset
        self.vad_model_path = os.path.join(get_assets_path(), "pyannote_vad_model.bin")
        if self.use_vad_model:
            self.vad_device = self.get_device(vad_device)
            self.vad_model = self.load_vad_model(
                vad_onset=self.vad_onset, vad_offset=self.vad_offset
            )
        else:
            self.vad_model = None
        self.chunk_length = chunk_length  # VAD merging size
        self.last_speech_timestamp = 0.0

    def get_device(self, device: Union[int, str, "torch.device"]):
        """
        Converts the input device into a torch.device object.

        The input can be an integer, a string, or a `torch.device` object.

        The function handles a special case where the input device is "auto".
        When "auto" is specified, the device will default to the
        device of the model (self.model.device). If the model's device is also "auto",
        it selects "cuda" if a CUDA-capable device is available; otherwise, it selects "cpu".
        """
        if isinstance(device, torch.device):
            return device
        elif isinstance(device, str):
            if device == "auto" and self.model.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device == "auto":
                device = self.model.device
            return torch.device(device)
        elif device < 0:
            return torch.device("cpu")
        else:
            return torch.device(f"cuda:{device}")

    def forward(self, features, segments_metadata, **forward_params):
        encoder_output, outputs = self.model.generate_segment_batched(
            features, self.tokenizer, forward_params
        )

        segmented_outputs = []
        segment_sizes = []
        for segment_metadata, output in zip(segments_metadata, outputs):
            duration = segment_metadata["end_time"] - segment_metadata["start_time"]
            segment_size = int(duration * self.model.frames_per_second)
            segment_sizes.append(segment_size)
            (
                subsegments,
                seek,
                single_timestamp_ending,
            ) = self.model._split_segments_by_timestamps(
                tokenizer=self.tokenizer,
                tokens=output["tokens"],
                time_offset=segment_metadata["start_time"],
                segment_size=segment_size,
                segment_duration=duration,
                seek=0,
            )
            segmented_outputs.append(
                [
                    dict(
                        text=self.tokenizer.decode(subsegment["tokens"]),
                        avg_logprob=output["avg_logprob"],
                        no_speech_prob=output["no_speech_prob"],
                        tokens=subsegment["tokens"],
                        start=subsegment["start"],
                        end=subsegment["end"],
                        compression_ratio=get_compression_ratio(
                            self.tokenizer.decode(subsegment["tokens"])
                        ),
                    )
                    for subsegment in subsegments
                ]
            )
        if forward_params["word_timestamps"]:
            self.last_speech_timestamp = self.model.add_word_timestamps(
                segmented_outputs,
                self.tokenizer,
                encoder_output,
                segment_sizes,
                forward_params["prepend_punctuations"],
                forward_params["append_punctuations"],
                self.last_speech_timestamp,
            )

        return segmented_outputs

    def get_language_and_tokenizer(
        self, audio, task: Optional[str] = None, language: Optional[str] = None
    ):
        all_language_probs = None
        language_probability = 1.0

        if self.tokenizer is None:
            if not language:
                (
                    language,
                    language_probability,
                    all_language_probs,
                ) = self.model.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            if task is not None:
                self.tokenizer.task = self.tokenizer.tokenizer.token_to_id(
                    f"<|{task}|>"
                )

            if language is not None:
                self.tokenizer.language = self.tokenizer.tokenizer.token_to_id(
                    f"<|{language}|>"
                )
                self.tokenizer.language_code = language

        return language, language_probability, task, all_language_probs

    @staticmethod
    def audio_split(audio, segments, sampling_rate):
        """Returns splitted audio chunks as iterator"""
        audio_segments = []
        segments_metadata = []
        for seg in segments:
            f1 = int(seg["start"] * sampling_rate)
            f2 = int(seg["end"] * sampling_rate)
            seg_metadata = {
                "start_time": seg["start"],
                "end_time": seg["end"],
                "stitched_seg": seg["segments"],
            }
            audio_segments.append(audio[f1:f2])
            segments_metadata.append(seg_metadata)
        return audio_segments, segments_metadata

    def load_vad_model(self, vad_onset=0.500, vad_offset=0.363):
        vad_model = Model.from_pretrained(self.vad_model_path)
        hyperparameters = {
            "onset": vad_onset,
            "offset": vad_offset,
            "min_duration_on": 0.1,
            "min_duration_off": 0.1,
        }

        vad_pipeline = VoiceActivitySegmentation(
            segmentation=vad_model, device=torch.device(self.vad_device)
        )
        vad_pipeline.instantiate(hyperparameters)
        return vad_pipeline

    def transcribe(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        vad_segments: Optional[List[dict]] = None,
        batch_size: int = 16,
        language: Optional[str] = None,
        task: str = None,
        log_progress: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        log_prob_low_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = 0.6,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        max_new_tokens: Optional[int] = None,
        hotwords: Optional[str] = None,
        word_timestamps: bool = False,
        without_timestamps: bool = True,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """transcribe audio in chunks in batched fashion and return with language info.

        Arguments:
            audio: audio file as numpy array/path for batched transcription.
            vad_segments: Optionally provide list of dictionaries each containing "start", "end",
                and "segments" keys.
                "start" and "end" keys specify the start and end of the voiced region within
                30 sec boundary. An additional key "segments" contains all the start
                and end of voiced regions within that 30sec boundary as a list of tuples.
                If no vad_segments specified, it uses internal vad model automatically segment them.
            batch_size: the maximum number of parallel requests to model for decoding.
            language: The language spoken in the audio.
            task: either "transcribe" or "translate".
            log_progress: whether to show progress bar or not.
            beam_size: Beam size to use for decoding.
            best_of: Number of candidates when sampling with non-zero temperature.
            patience: Beam search patience factor.
            length_penalty: Exponential length penalty constant.
            repetition_penalty: Penalty applied to the score of previously generated tokens
                (set > 1 to penalize).
            no_repeat_ngram_size: Prevent repetitions of ngrams with this size (set 0 to disable).
            temperature: Temperature for sampling. It can be a tuple of temperatures,
                which will be successively used upon failures according to either
                `compression_ratio_threshold` or `log_prob_threshold`.
            compression_ratio_threshold: If the gzip compression ratio is above this value,
                treat as failed.
            log_prob_threshold: If the average log probability over sampled tokens is
                below this value, treat as failed.
            log_prob_low_threshold: This parameter alone is sufficient to skip an output text,
            whereas log_prob_threshold also looks for appropriate no_speech_threshold value.
            This value should be less than log_prob_threshold.
            no_speech_threshold: If the no_speech probability is higher than this value AND
                the average log probability over sampled tokens is below `log_prob_threshold`,
                consider the segment as silent.
            initial_prompt: Optional text string or iterable of token ids to provide as a
                prompt for the first window.
            prefix: Optional text to provide as a prefix for the first window.
            suppress_blank: Suppress blank outputs at the beginning of the sampling.
            suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
                of symbols as defined in `tokenizer.non_speech_tokens()`.
            prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
                with the next word
            append_punctuations: If word_timestamps is True, merge these punctuation symbols
                with the previous word
            max_new_tokens: Maximum number of new tokens to generate per-chunk. If not set,
                the maximum will be set by the default max_length.
            hotwords:
                Hotwords/hint phrases to the model. Has no effect if prefix is not None.
            word_timestamps: Extract word-level timestamps using the cross-attention pattern
                and dynamic time warping, and include the timestamps for each word in each segment.
                Set as False.
            without_timestamps: Only sample text tokens.

        Static params: (Fixed for batched version)
            max_initial_timestamp: The initial timestamp cannot be later than this, set at 0.0.
            multilingual: If True, perform transcription on multilingual videos. Set as False.
            output_language: Valid only if multilingual is set to True.
                Specifies the string representing the output language. One of
                'en' (English) or 'hybrid' (code-switched transcription). set as None.
            condition_on_previous_text: If True, the previous output of the model is provided
                as a prompt for the next window; disabling may make the text inconsistent across
                windows, but the model becomes less prone to getting stuck in a failure loop,
                such as repetition looping or timestamps going out of sync. Set as False
            prompt_reset_on_temperature: Resets prompt if temperature is above this value.
                Arg has effect only if condition_on_previous_text is True. Set at 0.5
            #TODO: support "hallucination_silence_threshold" when "word_timestamps=True"
            hallucination_silence_threshold: Optional[float]
                When word_timestamps is True, skip silent periods longer than this threshold
                (in seconds) when a possible hallucination is detected. set as None.
            clip_timestamps:
                Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to
                process. The last end timestamp defaults to the end of the file. Set as "0".

        unused:
            language_detection_threshold: If the maximum probability of the language tokens is
                higher than this value, the language is detected.
            language_detection_segments: Number of segments to consider for the language detection.
            vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
                without speech. This step is using the Silero VAD model
                https://github.com/snakers4/silero-vad.
            vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
                parameters and default values in the class `VadOptions`).
            chunk_length: The length of audio segments. If it is not None, it will overwrite the
                default chunk_length of the FeatureExtractor.


        Returns:
          A tuple with:

            - a generator over transcribed batched segments.
            - an instance of TranscriptionInfo.
        """

        sampling_rate = self.model.feature_extractor.sampling_rate

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        elif not isinstance(audio, torch.Tensor):
            audio = decode_audio(audio, sampling_rate=sampling_rate)
        duration = audio.shape[0] / sampling_rate

        # if no segment split is provided, use vad_model and generate segments
        if not vad_segments:
            # run the audio if it is less than 30 sec even without vad_segments
            if self.use_vad_model:
                vad_segments = self.vad_model(
                    {
                        "waveform": audio.unsqueeze(0),
                        "sample_rate": 16000,
                    }
                )
                vad_segments = merge_chunks(
                    vad_segments,
                    self.chunk_length,
                    onset=self.vad_onset,
                    offset=self.vad_offset,
                )
            elif duration < self.chunk_length:
                vad_segments = [
                    {"start": 0.0, "end": duration, "segments": [(0.0, duration)]}
                ]
            else:
                raise RuntimeError(
                    "No vad segments found. Set 'use_vad_model' to True while loading the model"
                )
        if self.model.model.is_multilingual:
            language = language or self.preset_language
        elif language != "en":
            if language is not None:
                self.model.logger.warning(
                    f"English-only model is used, but {language} language is"
                    "chosen, setting language to 'en'."
                )
            language = "en"

        (
            language,
            language_probability,
            task,
            all_language_probs,
        ) = self.get_language_and_tokenizer(audio, task, language)

        duration_after_vad = sum(
            segment["end"] - segment["start"] for segment in vad_segments
        )

        # batched options: see the difference with default options in WhisperModel
        batched_options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            log_prob_low_threshold=log_prob_low_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            temperatures=(
                temperature if isinstance(temperature, (list, tuple)) else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=get_suppressed_tokens(self.tokenizer, suppress_tokens),
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
            word_timestamps=word_timestamps,
            hallucination_silence_threshold=None,
            condition_on_previous_text=False,
            clip_timestamps="0",
            prompt_reset_on_temperature=0.5,
            multilingual=False,
            output_language=None,
            without_timestamps=without_timestamps,
            max_initial_timestamp=0.0,
        )

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=batched_options,
            vad_options=None,
            all_language_probs=all_language_probs,
        )

        audio_segments, segments_metadata = self.audio_split(
            audio, vad_segments, sampling_rate
        )
        to_cpu = (
            self.model.model.device == "cuda" and len(self.model.model.device_index) > 1
        )
        audio_segments = torch.nested.nested_tensor(audio_segments).to_padded_tensor(
            padding=0
        )
        features = (
            torch.stack(
                [
                    self.model.feature_extractor(audio_segment, to_cpu=to_cpu)[
                        ..., : self.model.feature_extractor.nb_max_frames
                    ]
                    for audio_segment in audio_segments
                ]
            )
            if duration_after_vad
            else []
        )

        segments = self._batched_segments_generator(
            features,
            segments_metadata,
            batch_size,
            batched_options,
            log_progress,
        )

        return segments, info

    def _batched_segments_generator(
        self, features, segments_metadata, batch_size, options, log_progress
    ):
        pbar = tqdm(total=len(features), disable=not log_progress, position=0)
        seg_idx = 0
        for i in range(0, len(features), batch_size):
            results = self.forward(
                features[i : i + batch_size],
                segments_metadata[i : i + batch_size],
                **options._asdict(),
            )

            for result in results:
                for segment in result:
                    seg_idx += 1
                    yield Segment(
                        seek=int(result[-1]["end"] * self.model.frames_per_second),
                        id=seg_idx,
                        text=segment["text"],
                        start=round(segment["start"], 3),
                        end=round(segment["end"], 3),
                        words=(
                            None
                            if not options.word_timestamps
                            else [Word(**word) for word in segment["words"]]
                        ),
                        tokens=segment["tokens"],
                        avg_logprob=segment["avg_logprob"],
                        no_speech_prob=segment["no_speech_prob"],
                        compression_ratio=segment["compression_ratio"],
                    )

                pbar.update(1)

        pbar.close()
        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None
        self.last_speech_timestamp = 0.0


class WhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        device: str = "auto",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 16,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        files: dict = None,
        **model_kwargs,
    ):
        """Initializes the Whisper model.

        Args:
          model_size_or_path: Size of the model to use (tiny, tiny.en, base, base.en,
            small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,
            large-v2, large-v3, large, distil-large-v2 or distil-large-v3), a path to a
            converted model directory, or a CTranslate2-converted Whisper model ID from the HF Hub.
            When a size or a model ID is configured, the converted model is downloaded
            from the Hugging Face Hub.
          device: Device to use for computation ("cpu", "cuda", "auto").
          device_index: Device ID to use.
            The model can also be loaded on multiple GPUs by passing a list of IDs
            (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can run in parallel
            when transcribe() is called from multiple Python threads (see also num_workers).
          compute_type: Type to use for computation.
            See https://opennmt.net/CTranslate2/quantization.html.
          cpu_threads: Number of threads to use when running on CPU (4 by default).
            A non zero value overrides the OMP_NUM_THREADS environment variable.
          num_workers: When transcribe() is called from multiple Python threads,
            having multiple workers enables true parallelism when running the model
            (concurrent calls to self.model.generate() will run in parallel).
            This can improve the global throughput at the cost of increased memory usage.
          download_root: Directory where the models should be saved. If not set, the models
            are saved in the standard Hugging Face cache directory.
          local_files_only:  If True, avoid downloading the file and return the path to the
            local cached file if it exists.
          files: Load model files from the memory. This argument is a dictionary mapping file names
            to file contents as file-like or bytes objects. If this is set, model_path acts as an
            identifier for this model.
        """
        self.logger = get_logger()

        tokenizer_bytes, preprocessor_bytes = None, None
        if files:
            model_path = model_size_or_path
            tokenizer_bytes = files.pop("tokenizer.json", None)
            preprocessor_bytes = files.pop("preprocessor_config.json", None)
        elif os.path.isdir(model_size_or_path):
            model_path = model_size_or_path
        else:
            model_path = download_model(
                model_size_or_path,
                local_files_only=local_files_only,
                cache_dir=download_root,
            )
        self.device = device
        # set the random seed to make sure consistency across runs
        ctranslate2.set_random_seed(42)
        self.model = ctranslate2.models.Whisper(
            model_path,
            device=self.device,
            device_index=device_index,
            compute_type=compute_type,
            intra_threads=cpu_threads,
            inter_threads=num_workers,
            files=files,
            **model_kwargs,
        )

        tokenizer_file = os.path.join(model_path, "tokenizer.json")
        if tokenizer_bytes:
            self.hf_tokenizer = tokenizers.Tokenizer.from_buffer(tokenizer_bytes)
        elif os.path.isfile(tokenizer_file):
            self.hf_tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
        else:
            self.hf_tokenizer = tokenizers.Tokenizer.from_pretrained(
                "openai/whisper-tiny" + ("" if self.model.is_multilingual else ".en")
            )
        self.feat_kwargs = self._get_feature_kwargs(model_path, preprocessor_bytes)
        self.feature_extractor = FeatureExtractor(
            **self.feat_kwargs, device=self.device
        )
        self.input_stride = 2
        self.num_samples_per_token = (
            self.feature_extractor.hop_length * self.input_stride
        )
        self.frames_per_second = (
            self.feature_extractor.sampling_rate // self.feature_extractor.hop_length
        )
        self.tokens_per_second = (
            self.feature_extractor.sampling_rate // self.num_samples_per_token
        )
        self.time_precision = 0.02
        self.max_length = 448

    @property
    def supported_languages(self) -> List[str]:
        """The languages supported by the model."""
        return list(_LANGUAGE_CODES) if self.model.is_multilingual else ["en"]

    def _get_feature_kwargs(self, model_path, preprocessor_bytes=None) -> dict:
        config = {}
        try:
            config_path = os.path.join(model_path, "preprocessor_config.json")
            if preprocessor_bytes:
                config = json.loads(preprocessor_bytes)
            elif os.path.isfile(config_path):
                with open(config_path, "r", encoding="utf-8") as file:
                    config = json.load(file)
            else:
                return config
            valid_keys = signature(FeatureExtractor.__init__).parameters.keys()
            return {k: v for k, v in config.items() if k in valid_keys}
        except json.JSONDecodeError as e:
            self.logger.warning("Could not load preprocessor config: %s", e)

        return config

    def transcribe(
        self,
        audio: Union[str, BinaryIO, torch.Tensor, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        log_prob_low_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        multilingual: bool = False,
        output_language: Optional[str] = None,
        vad_filter: bool = False,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None,
        hotwords: Optional[str] = None,
        language_detection_threshold: Optional[float] = None,
        language_detection_segments: int = 1,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribes an input file.

        Arguments:
          audio: Path to the input file (or a file-like object), or the audio waveform.
          language: The language spoken in the audio. It should be a language code such
            as "en" or "fr". If not set, the language will be detected in the first 30 seconds
            of audio.
          task: Task to execute (transcribe or translate).
          beam_size: Beam size to use for decoding.
          best_of: Number of candidates when sampling with non-zero temperature.
          patience: Beam search patience factor.
          length_penalty: Exponential length penalty constant.
          repetition_penalty: Penalty applied to the score of previously generated tokens
            (set > 1 to penalize).
          no_repeat_ngram_size: Prevent repetitions of ngrams with this size (set 0 to disable).
          temperature: Temperature for sampling. It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            `compression_ratio_threshold` or `log_prob_threshold`.
          compression_ratio_threshold: If the gzip compression ratio is above this value,
            treat as failed.
          log_prob_threshold: If the average log probability over sampled tokens is
            below this value, treat as failed.
          log_prob_low_threshold: This parameter alone is sufficient to skip an output text,
          wheras log_prob_threshold also looks for appropriate no_speech_threshold value.
          This value should be less than log_prob_threshold.
          no_speech_threshold: If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
          condition_on_previous_text: If True, the previous output of the model is provided
            as a prompt for the next window; disabling may make the text inconsistent across
            windows, but the model becomes less prone to getting stuck in a failure loop,
            such as repetition looping or timestamps going out of sync.
          prompt_reset_on_temperature: Resets prompt if temperature is above this value.
            Arg has effect only if condition_on_previous_text is True.
          initial_prompt: Optional text string or iterable of token ids to provide as a
            prompt for the first window.
          prefix: Optional text to provide as a prefix for the first window.
          suppress_blank: Suppress blank outputs at the beginning of the sampling.
          suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
            of symbols as defined in `tokenizer.non_speech_tokens()`.
          without_timestamps: Only sample text tokens.
          max_initial_timestamp: The initial timestamp cannot be later than this.
          word_timestamps: Extract word-level timestamps using the cross-attention pattern
            and dynamic time warping, and include the timestamps for each word in each segment.
          prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the next word
          append_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the previous word
          multilingual: If True, perform transcription on multilingual videos
            and return the transcript based
            on the 'output_language' flag.
          output_language: Valid only if multilingual is set to True.
            Specifies the string representing the output language. One of
            'en' (English) or 'hybrid' (code-switched transcription).
          vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
            without speech. This step is using the Silero VAD model
            https://github.com/snakers4/silero-vad.
          vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
            parameters and default values in the class `VadOptions`).
          max_new_tokens: Maximum number of new tokens to generate per-chunk. If not set,
            the maximum will be set by the default max_length.
          chunk_length: The length of audio segments. If it is not None, it will overwrite the
            default chunk_length of the FeatureExtractor.
          clip_timestamps:
            Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to
             process. The last end timestamp defaults to the end of the file.
             vad_filter will be ignored if clip_timestamps is used.
          hallucination_silence_threshold:
            When word_timestamps is True, skip silent periods longer than this threshold
             (in seconds) when a possible hallucination is detected
          hotwords:
            Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.
          language_detection_threshold: If the maximum probability of the language tokens is higher
           than this value, the language is detected.
          language_detection_segments: Number of segments to consider for the language detection.
        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """

        sampling_rate = self.feature_extractor.sampling_rate

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        elif not isinstance(audio, torch.Tensor):
            audio = decode_audio(audio, sampling_rate=sampling_rate)

        duration = audio.shape[0] / sampling_rate
        duration_after_vad = duration

        self.logger.info(
            "Processing audio with duration %s", format_timestamp(duration)
        )

        if vad_filter and clip_timestamps == "0":
            if vad_parameters is None:
                vad_parameters = VadOptions()
            elif isinstance(vad_parameters, dict):
                vad_parameters = VadOptions(**vad_parameters)
            speech_chunks = get_speech_timestamps(audio, vad_parameters)
            audio = collect_chunks(audio, speech_chunks)
            duration_after_vad = audio.shape[0] / sampling_rate

            self.logger.info(
                "VAD filter removed %s of audio",
                format_timestamp(duration - duration_after_vad),
            )

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "VAD filter kept the following audio segments: %s",
                    ", ".join(
                        "[%s -> %s]"
                        % (
                            format_timestamp(chunk["start"] / sampling_rate),
                            format_timestamp(chunk["end"] / sampling_rate),
                        )
                        for chunk in speech_chunks
                    ),
                )

        else:
            speech_chunks = None

        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        features = self.feature_extractor(
            audio, chunk_length=chunk_length, to_cpu=to_cpu
        )

        encoder_output = None
        all_language_probs = None

        # setting output_language for multilingual videos
        if multilingual:
            if output_language is None:
                output_language = "en"
            elif output_language not in ["en", "hybrid"]:
                raise ValueError("Output language needs to be one of 'en'/'hybrid'.")

        # detecting the language if not provided
        if language is None:
            if not self.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                if (
                    language_detection_segments is None
                    or language_detection_segments < 1
                ):
                    language_detection_segments = 1
                start_timestamp = (
                    float(clip_timestamps.split(",")[0])
                    if isinstance(clip_timestamps, str)
                    else clip_timestamps[0]
                )
                content_frames = (
                    features.shape[-1] - self.feature_extractor.nb_max_frames
                )
                seek = (
                    int(start_timestamp * self.frames_per_second)
                    if start_timestamp * self.frames_per_second < content_frames
                    else 0
                )
                end_frames = min(
                    seek
                    + self.feature_extractor.nb_max_frames
                    * language_detection_segments,
                    content_frames,
                )
                detected_language_info = {}
                while seek <= end_frames:
                    segment = features[
                        :, seek : seek + self.feature_extractor.nb_max_frames
                    ]
                    encoder_output = self.encode(segment)
                    # results is a list of tuple[str, float] with language names and
                    # probabilities.
                    results = self.model.detect_language(encoder_output)[0]
                    # Parse language names to strip out markers
                    all_language_probs = [
                        (token[2:-2], prob) for (token, prob) in results
                    ]
                    # Get top language token and probability
                    language, language_probability = all_language_probs[0]
                    if (
                        language_detection_threshold is None
                        or language_probability > language_detection_threshold
                    ):
                        break
                    detected_language_info.setdefault(language, []).append(
                        language_probability
                    )
                    seek += segment.shape[-1]
                else:
                    if detected_language_info:
                        # If no language detected for all segments, the majority vote of the highest
                        # projected languages for all segments is used to determine the language.
                        language = max(
                            detected_language_info,
                            key=lambda lang: len(detected_language_info[lang]),
                        )
                        language_probability = max(detected_language_info[language])
                    else:
                        # It's possible VAD removes all segments due to no voice,
                        # then it doesn't matter which language
                        language = "en"
                        language_probability = 0

                self.logger.info(
                    "Detected language '%s' with probability %.2f",
                    language,
                    language_probability,
                )
        else:
            if not self.model.is_multilingual and language != "en":
                self.logger.warning(
                    "The current model is English-only but the language parameter is set to '%s'; "
                    "using 'en' instead." % language
                )
                language = "en"

            language_probability = 1

        tokenizer = Tokenizer(
            self.hf_tokenizer,
            self.model.is_multilingual,
            task=task,
            language=language,
        )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            log_prob_low_threshold=log_prob_low_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            temperatures=(
                temperature if isinstance(temperature, (list, tuple)) else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=(
                get_suppressed_tokens(tokenizer, suppress_tokens)
                if suppress_tokens
                else suppress_tokens
            ),
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            multilingual=multilingual,
            output_language=output_language,
            max_new_tokens=max_new_tokens,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
        )

        segments = self.generate_segments(features, tokenizer, options, encoder_output)

        if speech_chunks:
            segments = restore_speech_timestamps(segments, speech_chunks, sampling_rate)

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )
        return segments, info

    def _split_segments_by_timestamps(
        self,
        tokenizer: Tokenizer,
        tokens: List[int],
        time_offset: float,
        segment_size: int,
        segment_duration: float,
        seek: int,
    ) -> List[List[int]]:
        current_segments = []
        single_timestamp_ending = (
            len(tokens) >= 2 and tokens[-2] < tokenizer.timestamp_begin <= tokens[-1]
        )

        consecutive_timestamps = [
            i
            for i in range(len(tokens))
            if i > 0
            and tokens[i] >= tokenizer.timestamp_begin
            and tokens[i - 1] >= tokenizer.timestamp_begin
        ]

        if len(consecutive_timestamps) > 0:
            slices = list(consecutive_timestamps)
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = sliced_tokens[0] - tokenizer.timestamp_begin
                end_timestamp_position = sliced_tokens[-1] - tokenizer.timestamp_begin
                start_time = (
                    time_offset + start_timestamp_position * self.time_precision
                )
                end_time = time_offset + end_timestamp_position * self.time_precision

                current_segments.append(
                    dict(
                        seek=seek,
                        start=start_time,
                        end=end_time,
                        tokens=sliced_tokens,
                    )
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                seek += segment_size
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                last_timestamp_position = (
                    tokens[last_slice - 1] - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * self.input_stride

        else:
            duration = segment_duration
            timestamps = [
                token for token in tokens if token >= tokenizer.timestamp_begin
            ]
            if len(timestamps) > 0 and timestamps[-1] != tokenizer.timestamp_begin:
                last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                duration = last_timestamp_position * self.time_precision

            current_segments.append(
                dict(
                    seek=seek,
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                )
            )

            seek += segment_size

        return current_segments, seek, single_timestamp_ending

    def generate_segments(
        self,
        features: torch.Tensor,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        encoder_output: Optional[ctranslate2.StorageView] = None,
    ) -> Iterable[Segment]:
        content_frames = features.shape[-1] - self.feature_extractor.nb_max_frames
        content_duration = float(content_frames * self.feature_extractor.time_per_frame)

        if isinstance(options.clip_timestamps, str):
            options = options._replace(
                clip_timestamps=[
                    float(ts)
                    for ts in (
                        options.clip_timestamps.split(",")
                        if options.clip_timestamps
                        else []
                    )
                ]
            )
        seek_points: List[int] = [
            round(ts * self.frames_per_second) for ts in options.clip_timestamps
        ]
        if len(seek_points) == 0:
            seek_points.append(0)
        if len(seek_points) % 2 == 1:
            seek_points.append(content_frames)
        seek_clips: List[Tuple[int, int]] = list(
            zip(seek_points[::2], seek_points[1::2])
        )

        punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

        idx = 0
        clip_idx = 0
        seek = seek_clips[clip_idx][0]
        all_tokens = []
        prompt_reset_since = 0

        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(options.initial_prompt)

        last_speech_timestamp = 0.0
        # NOTE: This loop is obscurely flattened to make the diff readable.
        # A later commit should turn this into a simpler nested loop.
        # for seek_clip_start, seek_clip_end in seek_clips:
        #     while seek < seek_clip_end
        while clip_idx < len(seek_clips):
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek_clip_end > content_frames:
                seek_clip_end = content_frames
            if seek < seek_clip_start:
                seek = seek_clip_start
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips):
                    seek = seek_clips[clip_idx][0]
                continue
            time_offset = seek * self.feature_extractor.time_per_frame
            window_end_time = float(
                (seek + self.feature_extractor.nb_max_frames)
                * self.feature_extractor.time_per_frame
            )
            segment_size = min(
                self.feature_extractor.nb_max_frames,
                content_frames - seek,
                seek_clip_end - seek,
            )
            segment = features[:, seek : seek + segment_size]
            segment_duration = segment_size * self.feature_extractor.time_per_frame
            segment = pad_or_trim(segment, self.feature_extractor.nb_max_frames)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Processing segment at %s", format_timestamp(time_offset)
                )

            previous_tokens = all_tokens[prompt_reset_since:]

            if encoder_output is None:
                encoder_output = self.encode(segment)

            # Perform language detection at every segment to update task based on output language,
            # if the language is english, task is transcribe,
            # else the task is translate to english (default)
            # or transcribe if 'output_language' is 'hybrid'.
            if options.multilingual:
                results = self.model.detect_language(encoder_output)
                language_token, language_probability = results[0][0]
                language = language_token[2:-2]
                if options.output_language == "en" and language != "en":
                    task = "translate"
                else:
                    task = "transcribe"

                # Update tokenizer based on task and language
                tokenizer.task = tokenizer.tokenizer.token_to_id(f"<|{task}|>")
                tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
                tokenizer.language_code = language
            # Update prompt based on task and language
            prompt = self.get_prompt(
                tokenizer,
                previous_tokens,
                without_timestamps=options.without_timestamps,
                prefix=options.prefix if seek == 0 else None,
                hotwords=options.hotwords,
            )

            if seek > 0 or encoder_output is None:
                encoder_output = self.encode(segment)

            (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            ) = self.generate_with_fallback(encoder_output, prompt, tokenizer, options)

            if options.no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > options.no_speech_threshold

                if (
                    options.log_prob_threshold is not None
                    and avg_logprob > options.log_prob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    self.logger.debug(
                        "No speech threshold is met (%f > %f)",
                        result.no_speech_prob,
                        options.no_speech_threshold,
                    )

                # Skip if the logprob is very low (below the threshold value),
                # despite no_speech_prob being low (ex: Too ambiguous outputs)
                if options.log_prob_low_threshold:
                    if avg_logprob < options.log_prob_low_threshold:
                        should_skip = True
                        self.logger.debug(
                            "log prob low threshold is met (%f > %f)",
                            avg_logprob,
                            options.log_prob_low_threshold,
                        )

                if should_skip:
                    # fast-forward to the next segment boundary
                    seek += segment_size
                    continue

            tokens = result.sequences_ids[0]

            previous_seek = seek

            # anomalous words are very long/short/improbable
            def word_anomaly_score(word: dict) -> float:
                probability = word.get("probability", 0.0)
                duration = word["end"] - word["start"]
                score = 0.0
                if probability < 0.15:
                    score += 1.0
                if duration < 0.133:
                    score += (0.133 - duration) * 15
                if duration > 2.0:
                    score += duration - 2.0
                return score

            def is_segment_anomaly(segment: Optional[dict]) -> bool:
                if segment is None or not segment["words"]:
                    return False
                words = [w for w in segment["words"] if w["word"] not in punctuation]
                words = words[:8]
                score = sum(word_anomaly_score(w) for w in words)
                return score >= 3 or score + 0.01 >= len(words)

            def next_words_segment(segments: List[dict]) -> Optional[dict]:
                return next((s for s in segments if s["words"]), None)

            (
                current_segments,
                seek,
                single_timestamp_ending,
            ) = self._split_segments_by_timestamps(
                tokenizer=tokenizer,
                tokens=tokens,
                time_offset=time_offset,
                segment_size=segment_size,
                segment_duration=segment_duration,
                seek=seek,
            )

            if options.word_timestamps:
                self.add_word_timestamps(
                    [current_segments],
                    tokenizer,
                    encoder_output,
                    segment_size,
                    options.prepend_punctuations,
                    options.append_punctuations,
                    last_speech_timestamp=last_speech_timestamp,
                )
                if not single_timestamp_ending:
                    last_word_end = get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset:
                        seek = round(last_word_end * self.frames_per_second)

                # skip silence before possible hallucinations
                if options.hallucination_silence_threshold is not None:
                    threshold = options.hallucination_silence_threshold

                    # if first segment might be a hallucination, skip leading silence
                    first_segment = next_words_segment(current_segments)
                    if first_segment is not None and is_segment_anomaly(first_segment):
                        gap = first_segment["start"] - time_offset
                        if gap > threshold:
                            seek = previous_seek + round(gap * self.frames_per_second)
                            continue

                    # skip silence before any possible hallucination that is surrounded
                    # by silence or more hallucinations
                    hal_last_end = last_speech_timestamp
                    for si in range(len(current_segments)):
                        segment = current_segments[si]
                        if not segment["words"]:
                            continue
                        if is_segment_anomaly(segment):
                            next_segment = next_words_segment(
                                current_segments[si + 1 :]
                            )
                            if next_segment is not None:
                                hal_next_start = next_segment["words"][0]["start"]
                            else:
                                hal_next_start = time_offset + segment_duration
                            silence_before = (
                                segment["start"] - hal_last_end > threshold
                                or segment["start"] < threshold
                                or segment["start"] - time_offset < 2.0
                            )
                            silence_after = (
                                hal_next_start - segment["end"] > threshold
                                or is_segment_anomaly(next_segment)
                                or window_end_time - segment["end"] < 2.0
                            )
                            if silence_before and silence_after:
                                seek = round(
                                    max(time_offset + 1, segment["start"])
                                    * self.frames_per_second
                                )
                                if content_duration - segment["end"] < threshold:
                                    seek = content_frames
                                current_segments[si:] = []
                                break
                        hal_last_end = segment["end"]

                last_word_end = get_end(current_segments)
                if last_word_end is not None:
                    last_speech_timestamp = last_word_end
            for segment in current_segments:
                tokens = segment["tokens"]
                text = tokenizer.decode(tokens)

                if segment["start"] == segment["end"] or not text.strip():
                    continue

                all_tokens.extend(tokens)
                idx += 1

                yield Segment(
                    id=idx,
                    seek=seek,
                    start=segment["start"],
                    end=segment["end"],
                    text=text,
                    tokens=tokens,
                    temperature=temperature,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio,
                    no_speech_prob=result.no_speech_prob,
                    words=(
                        [Word(**word) for word in segment["words"]]
                        if options.word_timestamps
                        else None
                    ),
                )

            if (
                not options.condition_on_previous_text
                or temperature > options.prompt_reset_on_temperature
            ):
                if options.condition_on_previous_text:
                    self.logger.debug(
                        "Reset prompt. prompt_reset_on_temperature threshold is met %f > %f",
                        temperature,
                        options.prompt_reset_on_temperature,
                    )

                prompt_reset_since = len(all_tokens)

    def encode(self, features: torch.Tensor) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

        if features.ndim == 2:
            features = features.unsqueeze(0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

    def generate_with_fallback(
        self,
        encoder_output: ctranslate2.StorageView,
        prompt: List[int],
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
    ) -> Tuple[ctranslate2.models.WhisperGenerationResult, float, float, float]:
        decode_result = None
        all_results = []
        below_cr_threshold_results = []

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )
        if options.max_new_tokens is not None:
            max_length = len(prompt) + options.max_new_tokens
        else:
            max_length = self.max_length

        if max_length > self.max_length:
            raise ValueError(
                f"The length of the prompt is {len(prompt)}, and the `max_new_tokens` "
                f"{max_length - len(prompt)}. Thus, the combined length of the prompt "
                f"and `max_new_tokens` is: {max_length}. This exceeds the "
                f"`max_length` of the Whisper model: {self.max_length}. "
                "You should either reduce the length of your prompt, or "
                "reduce the value of `max_new_tokens`, "
                f"so that their combined length is less that {self.max_length}."
            )

        for temperature in options.temperatures:
            if temperature > 0:
                kwargs = {
                    "beam_size": 1,
                    "num_hypotheses": options.best_of,
                    "sampling_topk": 0,
                    "sampling_temperature": temperature,
                }
            else:
                kwargs = {
                    "beam_size": options.beam_size,
                    "patience": options.patience,
                }

            result = self.model.generate(
                encoder_output,
                [prompt],
                length_penalty=options.length_penalty,
                repetition_penalty=options.repetition_penalty,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                max_length=max_length,
                return_scores=True,
                return_no_speech_prob=True,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                max_initial_timestamp_index=max_initial_timestamp_index,
                **kwargs,
            )[0]

            tokens = result.sequences_ids[0]

            # Recover the average log prob from the returned score.
            seq_len = len(tokens)
            cum_logprob = result.scores[0] * (seq_len**options.length_penalty)
            avg_logprob = cum_logprob / (seq_len + 1)

            text = tokenizer.decode(tokens).strip()
            compression_ratio = get_compression_ratio(text)

            decode_result = (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            )
            all_results.append(decode_result)

            needs_fallback = False

            if options.compression_ratio_threshold is not None:
                if compression_ratio > options.compression_ratio_threshold:
                    needs_fallback = True  # too repetitive

                    self.logger.debug(
                        "Compression ratio threshold is not met with temperature %.1f (%f > %f)",
                        temperature,
                        compression_ratio,
                        options.compression_ratio_threshold,
                    )
                else:
                    below_cr_threshold_results.append(decode_result)

            if (
                options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = True  # average log probability is too low

                self.logger.debug(
                    "Log probability threshold is not met with temperature %.1f (%f < %f)",
                    temperature,
                    avg_logprob,
                    options.log_prob_threshold,
                )

            if (
                options.no_speech_threshold is not None
                and result.no_speech_prob > options.no_speech_threshold
                and options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = False  # silence

            if not needs_fallback:
                break
        else:
            # all failed, select the result with the highest average log probability
            decode_result = max(
                below_cr_threshold_results or all_results, key=lambda x: x[1]
            )
            # to pass final temperature for prompt_reset_on_temperature
            decode_result = (
                decode_result[0],
                decode_result[1],
                temperature,
                decode_result[3],
            )

        return decode_result

    def get_prompt(
        self,
        tokenizer: Tokenizer,
        previous_tokens: List[int],
        without_timestamps: bool = False,
        prefix: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> List[int]:
        prompt = []

        if previous_tokens or (hotwords and not prefix):
            prompt.append(tokenizer.sot_prev)
            if hotwords and not prefix:
                hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
                if len(hotwords_tokens) >= self.max_length // 2:
                    hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
                prompt.extend(hotwords_tokens)
            if previous_tokens:
                prompt.extend(previous_tokens[-(self.max_length // 2 - 1) :])

        prompt.extend(tokenizer.sot_sequence)

        if without_timestamps:
            prompt.append(tokenizer.no_timestamps)

        if prefix:
            prefix_tokens = tokenizer.encode(" " + prefix.strip())
            if len(prefix_tokens) >= self.max_length // 2:
                prefix_tokens = prefix_tokens[: self.max_length // 2 - 1]
            if not without_timestamps:
                prompt.append(tokenizer.timestamp_begin)
            prompt.extend(prefix_tokens)

        return prompt

    def add_word_timestamps(
        self,
        segments: List[dict],
        tokenizer: Tokenizer,
        encoder_output: ctranslate2.StorageView,
        num_frames: int,
        prepend_punctuations: str,
        append_punctuations: str,
        last_speech_timestamp: float,
    ) -> float:
        if len(segments) == 0:
            return

        text_tokens = []
        text_tokens_per_segment = []
        for segment in segments:
            segment_tokens = [
                [token for token in subsegment["tokens"] if token < tokenizer.eot]
                for subsegment in segment
            ]
            text_tokens.append(list(itertools.chain.from_iterable(segment_tokens)))
            text_tokens_per_segment.append(segment_tokens)

        alignments = self.find_alignment(
            tokenizer, text_tokens, encoder_output, num_frames
        )
        median_max_durations = []
        for alignment in alignments:
            word_durations = np.array(
                [word["end"] - word["start"] for word in alignment]
            )
            word_durations = word_durations[word_durations.nonzero()]
            median_duration = (
                np.median(word_durations) if len(word_durations) > 0 else 0.0
            )
            median_duration = min(0.7, float(median_duration))
            max_duration = median_duration * 2

            # hack: truncate long words at sentence boundaries.
            # a better segmentation algorithm based on VAD should be able to replace this.
            if len(word_durations) > 0:
                sentence_end_marks = ".。!！?？"
                # ensure words at sentence boundaries
                # are not longer than twice the median word duration.
                for i in range(1, len(alignment)):
                    if alignment[i]["end"] - alignment[i]["start"] > max_duration:
                        if alignment[i]["word"] in sentence_end_marks:
                            alignment[i]["end"] = alignment[i]["start"] + max_duration
                        elif alignment[i - 1]["word"] in sentence_end_marks:
                            alignment[i]["start"] = alignment[i]["end"] - max_duration

            merge_punctuations(alignment, prepend_punctuations, append_punctuations)
            median_max_durations.append((median_duration, max_duration))

        for segment_idx, segment in enumerate(segments):
            word_index = 0
            time_offset = segment[0]["start"]
            median_duration, max_duration = median_max_durations[segment_idx]
            for subsegment_idx, subsegment in enumerate(segment):
                saved_tokens = 0
                words = []

                while word_index < len(alignments[segment_idx]) and saved_tokens < len(
                    text_tokens_per_segment[segment_idx][subsegment_idx]
                ):
                    timing = alignments[segment_idx][word_index]

                    if timing["word"]:
                        words.append(
                            dict(
                                word=timing["word"],
                                start=round(time_offset + timing["start"], 2),
                                end=round(time_offset + timing["end"], 2),
                                probability=timing["probability"],
                            )
                        )

                    saved_tokens += len(timing["tokens"])
                    word_index += 1

                # hack: truncate long words at segment boundaries.
                # a better segmentation algorithm based on VAD should be able to replace this.
                if len(words) > 0:
                    # ensure the first and second word after a pause is not longer than
                    # twice the median word duration.
                    if words[0][
                        "end"
                    ] - last_speech_timestamp > median_duration * 4 and (
                        words[0]["end"] - words[0]["start"] > max_duration
                        or (
                            len(words) > 1
                            and words[1]["end"] - words[0]["start"] > max_duration * 2
                        )
                    ):
                        if (
                            len(words) > 1
                            and words[1]["end"] - words[1]["start"] > max_duration
                        ):
                            boundary = max(
                                words[1]["end"] / 2, words[1]["end"] - max_duration
                            )
                            words[0]["end"] = words[1]["start"] = boundary
                        words[0]["start"] = max(0, words[0]["end"] - max_duration)

                    # prefer the segment-level start timestamp if the first word is too long.
                    if (
                        subsegment["start"] < words[0]["end"]
                        and subsegment["start"] - 0.5 > words[0]["start"]
                    ):
                        words[0]["start"] = max(
                            0,
                            min(words[0]["end"] - median_duration, subsegment["start"]),
                        )
                    else:
                        subsegment["start"] = words[0]["start"]

                    # prefer the segment-level end timestamp if the last word is too long.
                    if (
                        subsegment["end"] > words[-1]["start"]
                        and subsegment["end"] + 0.5 < words[-1]["end"]
                    ):
                        words[-1]["end"] = max(
                            words[-1]["start"] + median_duration, subsegment["end"]
                        )
                    else:
                        subsegment["end"] = words[-1]["end"]

                    last_speech_timestamp = subsegment["end"]
                segments[segment_idx][subsegment_idx]["words"] = words
        return last_speech_timestamp

    def find_alignment(
        self,
        tokenizer: Tokenizer,
        text_tokens: List[int],
        encoder_output: ctranslate2.StorageView,
        num_frames: int,
        median_filter_width: int = 7,
    ) -> List[dict]:
        if len(text_tokens) == 0:
            return []

        results = self.model.align(
            encoder_output,
            tokenizer.sot_sequence,
            text_tokens,
            num_frames,
            median_filter_width=median_filter_width,
        )
        return_list = []
        for result, text_token in zip(results, text_tokens):
            text_token_probs = result.text_token_probs
            alignments = result.alignments
            text_indices = np.array([pair[0] for pair in alignments])
            time_indices = np.array([pair[1] for pair in alignments])

            words, word_tokens = tokenizer.split_to_word_tokens(
                text_token + [tokenizer.eot]
            )
            if len(word_tokens) <= 1:
                # return on eot only
                # >>> np.pad([], (1, 0))
                # array([0.])
                # This results in crashes when we lookup jump_times with float, like
                # IndexError: arrays used as indices must be of integer (or boolean) type
                return []
            word_boundaries = np.pad(
                np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0)
            )
            if len(word_boundaries) <= 1:
                return []

            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(
                bool
            )
            jump_times = time_indices[jumps] / self.tokens_per_second
            start_times = jump_times[word_boundaries[:-1]]
            end_times = jump_times[word_boundaries[1:]]
            word_probabilities = [
                np.mean(text_token_probs[i:j])
                for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
            ]

            return_list.append(
                [
                    dict(
                        word=word,
                        tokens=tokens,
                        start=start,
                        end=end,
                        probability=probability,
                    )
                    for word, tokens, start, end, probability in zip(
                        words, word_tokens, start_times, end_times, word_probabilities
                    )
                ]
            )
        return return_list

    def generate_segment_batched(
        self,
        features: torch.Tensor,
        tokenizer: Tokenizer,
        options: dict,
    ):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0

        if options["initial_prompt"] is not None:
            initial_prompt = " " + options["initial_prompt"].strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options["without_timestamps"],
            prefix=options["prefix"],
        )

        encoder_output = self.encode(features)

        result = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            beam_size=options["beam_size"],
            patience=options["patience"],
            length_penalty=options["length_penalty"],
            max_length=self.max_length,
            suppress_blank=options["suppress_blank"],
            suppress_tokens=options["suppress_tokens"],
            return_scores=True,
            return_no_speech_prob=True,
        )

        output = []
        for res in result:
            output.append({})
            # return scores
            seq_len = len(res.sequences_ids[0])
            cum_logprob = res.scores[0] * (seq_len ** options["length_penalty"])
            output[-1]["avg_logprob"] = cum_logprob / (seq_len + 1)

            # return no speech prob
            output[-1]["no_speech_prob"] = res.no_speech_prob
            output[-1]["tokens"] = res.sequences_ids[0]

        return encoder_output, output

    def detect_language(self, audio: torch.Tensor):
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        segment = self.feature_extractor(audio, padding=True, to_cpu=to_cpu)[
            :, : self.feature_extractor.nb_max_frames
        ]
        encoder_output = self.encode(segment)
        results = self.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        self.logger.info(
            f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio..."
        )
        all_language_probs = [(token[2:-2], prob) for (token, prob) in results[0]]
        return language, language_probability, all_language_probs

    def detect_language_multi_segment(
        self, audio: Union[str, BinaryIO, torch.Tensor], params: Optional[dict] = None
    ):
        """
        Detect language based on N highly-confident segments of a language.
        """
        # The threshold is used to decide if the audio is silence or not.
        # The default is 0.02 (2.0%) i.e, if more than 2.0% of the audio is silent,
        # the audio is considered as silence.
        if not params:
            params = {
                "multilingual": False,
                "speech_percentage_threshold": 0.02,
                "language_detection_segments": 4,
                "vad_filter": True,
                "vad_min_silence_duration": 2500,
                "language_threshold": 0.7,
            }

        if params.get("multilingual", False):
            logging.warning(
                "lang_id is not supported for multilingual audios, detecting the major language."
            )

        speech_percentage_threshold = params.get("speech_percentage_threshold", 0.02)
        language_threshold = params.get("language_threshold", 0.7)
        num_detection_segments = params.get("language_detection_segments", 4)
        vad_filter_enabled = params.get("vad_filter", True)
        vad_params = dict(
            min_silence_duration_ms=params.get("vad_min_silence_duration", 2500)
        )

        if vad_filter_enabled:
            vad_params = VadOptions(**vad_params)

        # decode audio if it is not decoded already
        sampling_rate = self.feature_extractor.sampling_rate
        if not isinstance(audio, torch.Tensor):
            audio: torch.Tensor = decode_audio(audio, sampling_rate=sampling_rate)

        # calculate duration of audio as number of seconds
        # audio.shape[0] is the number of samples in the audio
        # sampling_rate is the number of samples per second
        # if we divide the number of samples by the number of samples per second,
        # we get the duration in seconds
        duration = audio.shape[0] / sampling_rate

        # Check if vad is enabled, and collect voiced segments
        if vad_filter_enabled:
            # get chunks of audio that contain speech
            speech_chunks = get_speech_timestamps(audio, vad_params)
            # merge chunks of audio that contain speech into a single array
            audio = collect_chunks(audio, speech_chunks)

            # calculate new duration of audio without silence
            duration_vad = audio.shape[0] / sampling_rate

            logging.debug(
                f"Lang ID: VAD filter removed {duration - duration_vad} sec of audio"
            )

            # if the audio after VAD is less than 2% of the original audio, consider it as silence
            if duration_vad / duration < speech_percentage_threshold:
                return {"language_code": None, "language_confidence": 1.0}

            # update duration to be the duration after VAD
            duration = duration_vad

        # if the duration of the audio is less than 1 second, consider it as silence
        if duration < 1.0:
            return {"language_code": None, "language_confidence": 1.0}

        # number of feature frames in 30 seconds of audio is 3000
        nb_max_frames = self.feature_extractor.nb_max_frames

        # extract features from audio with padding (default)
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        features = self.feature_extractor(audio, to_cpu=to_cpu)

        # number of segments in the audio
        num_segments = features.shape[-1] // nb_max_frames
        # more number of segments than possible with the duration of file
        if num_detection_segments > num_segments:
            logging.warning(
                f"Lang ID: Can not have more segments, setting {num_segments} segments."
            )
            num_detection_segments = num_segments

        # create a list of indices to randomly select segments from
        indices = list(range(num_detection_segments))

        # fix seed to get deterministic results
        random.seed(0)
        random.shuffle(indices)

        detected_languages = []
        all_language_probabilities = defaultdict(list)
        confident_language_probabilities = defaultdict(list)
        num_confident_segments_per_language = defaultdict(int)

        # Iterate over the randomly selected indices of the segments.
        #
        # For each segment, extract features and detect language.
        #
        # If the language is confident, add it to the list of confident segments for that language.
        #
        # If the number of confident segments for a language
        # is greater than or equal to the number of detection segments,
        # return the language and the average probability of the language.
        #
        # If we are unable to get sufficient number of confident predcitions,
        # return the most frequently detected language with maximum probability.
        #
        # We need to get sufficient number of confident predictions per language, not in total.

        for i in indices:
            segment_features = features[:, i * nb_max_frames : (i + 1) * nb_max_frames]
            try:
                encoder_output = self.encode(segment_features)
                results = self.model.detect_language(encoder_output)[0]

            except ValueError as e:  # or RuntimeError
                logging.error(f"Inference error:{e}")

            # results is the list of classes (languages) and their probabilities (descending),
            # for eg: [('<|de|>', 0.482177734375),('<|en|>', 0.283447265625),...]

            # take top language token and probability
            # and parse language token to strip out markers
            # for eg: '<|de|>' -> 'de'

            language_token = results[0][0]
            language = language_token[2:-2]

            language_probability = results[0][1]

            detected_languages.append(language)
            all_language_probabilities[language].append(language_probability)

            # only consider if the language prediction is confident
            if language_probability > language_threshold:
                num_confident_segments_per_language[language] += 1

                # Add language and probability to the list of languages when it is confident
                confident_language_probabilities[language].append(language_probability)

                # return the language when sufficient number of confident segments is achieved
                if (
                    num_confident_segments_per_language[language]
                    >= num_detection_segments
                ):
                    # Considering the average probability of only confident segments
                    mean = sum(confident_language_probabilities[language]) / len(
                        confident_language_probabilities[language]
                    )
                    return {
                        "language_code": language,
                        "language_confidence": mean,
                    }

        # if we are unable to get sufficient number of confident predictions,
        # return the most frequently detected language.
        # if there is a tie, return the one with maximum average probability.
        counter = Counter(detected_languages)

        # Define the key function to select frequent language with attached probabilities
        def key_func(language):
            # Calculate the frequency of the language
            frequency = counter[language]

            # Calculate the average probability of the language
            prob_avg = sum(all_language_probabilities[language]) / len(
                all_language_probabilities[language]
            )

            return frequency, prob_avg

        if detected_languages:
            # Use the key function to find the language with maximum frequency and probability
            max_language = max(detected_languages, key=key_func)
            max_probability = sum(all_language_probabilities[max_language]) / len(
                all_language_probabilities[max_language]
            )

            # Do additional checks for silence for non-confident case
            # calculate RMS amplitude and DC offset
            dc_offset = audio.mean()
            audio_minus_dc_offset = audio - dc_offset
            is_silent = (
                torch.all(audio.abs() < 0.01)
                or torch.sqrt(torch.mean(audio_minus_dc_offset**2)) < 0.01
            )

            if is_silent:
                return {"language_code": None, "language_confidence": 1.0}

            return {
                "language_code": max_language,
                "language_confidence": max_probability,
            }

        # Language is not detected for any segment and none of prev conditions met
        return {"language_code": None, "language_confidence": 1.0}


def restore_speech_timestamps(
    segments: Iterable[Segment],
    speech_chunks: List[dict],
    sampling_rate: int,
) -> Iterable[Segment]:
    ts_map = SpeechTimestampsMap(speech_chunks, sampling_rate)

    for segment in segments:
        if segment.words:
            words = []
            for word in segment.words:
                # Ensure the word start and end times are resolved to the same chunk.
                middle = (word.start + word.end) / 2
                chunk_index = ts_map.get_chunk_index(middle)
                word = word._replace(
                    start=ts_map.get_original_time(word.start, chunk_index),
                    end=ts_map.get_original_time(word.end, chunk_index),
                )
                words.append(word)

            segment = segment._replace(
                start=words[0].start,
                end=words[-1].end,
                words=words,
            )

        else:
            segment = segment._replace(
                start=ts_map.get_original_time(segment.start),
                end=ts_map.get_original_time(segment.end),
            )

        yield segment


def get_ctranslate2_storage(segment: torch.Tensor) -> ctranslate2.StorageView:
    segment = segment.contiguous()
    segment = ctranslate2.StorageView.from_array(
        segment if segment.is_cuda else segment.numpy()
    )  # torch cpu tensors don't implement __array_interface__
    # https://github.com/pytorch/pytorch/issues/51156
    return segment


def get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def get_suppressed_tokens(
    tokenizer: Tokenizer,
    suppress_tokens: Tuple[int],
) -> Optional[List[int]]:
    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []  # interpret empty string as an empty list
    else:
        assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

    suppress_tokens.extend(
        [
            tokenizer.transcribe,
            tokenizer.translate,
            tokenizer.sot,
            tokenizer.sot_prev,
            tokenizer.sot_lm,
        ]
    )

    return tuple(sorted(set(suppress_tokens)))


def merge_punctuations(alignment: List[dict], prepended: str, appended: str) -> None:
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous["word"].startswith(" ") and previous["word"].strip() in prepended:
            # prepend it to the following word
            following["word"] = previous["word"] + following["word"]
            if "tokens" in alignment[0].keys():
                following["tokens"] = previous["tokens"] + following["tokens"]
                previous["tokens"] = []
            previous["word"] = ""

        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous["word"].endswith(" ") and following["word"] in appended:
            # append it to the previous word
            previous["word"] = previous["word"] + following["word"]
            if "tokens" in alignment[0].keys():
                previous["tokens"] = previous["tokens"] + following["tokens"]
                following["tokens"] = []
            following["word"] = ""

        else:
            i = j
        j += 1
