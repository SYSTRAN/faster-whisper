import itertools
import json
import logging
import os
import random
import zlib

from collections import Counter, defaultdict
from inspect import signature
from math import ceil
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
from dataclasses import dataclass, field

import ctranslate2
import numpy as np
import tokenizers
import torch

from tqdm import tqdm

from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
from faster_whisper.utils import download_model, format_timestamp, get_end, get_logger
from faster_whisper.vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
    merge_segments,
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


@dataclass
class TranscriptionConfig:
    # Core parameters
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # Threshold parameters
    log_prob_threshold: Optional[float] = -1.0
    log_prob_low_threshold: Optional[float] = None
    no_speech_threshold: Optional[float] = 0.6
    compression_ratio_threshold: Optional[float] = 2.4
    language_threshold: float = 0.7
    speech_percentage_threshold: float = 0.02
    
    # Temperature settings
    prompt_reset_on_temperature: float = 0.5
    temperatures: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    
    # Text processing
    initial_prompt: Optional[Union[str, Iterable[int]]] = None
    prefix: Optional[str] = None
    suppress_blank: bool = True
    suppress_tokens: Optional[List[int]] = field(default_factory=lambda: [-1])
    without_timestamps: bool = True
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'"¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：")]}、"
    
    # Language and processing options
    multilingual: bool = False
    output_language: Optional[str] = None
    vad_filter: bool = False
    vad_parameters: Optional[Union[dict, VadOptions]] = None
    language_detection_segments: int = 4
    vad_min_silence_duration: int = 2500
    
    # Size and timing parameters
    max_new_tokens: Optional[int] = None
    max_initial_timestamp: float = 0.0
    chunk_length: Optional[int] = None
    clip_timestamps: Union[str, List[float]] = "0"
    
    # Advanced options
    condition_on_previous_text: bool = False
    hallucination_silence_threshold: Optional[float] = None
    hotwords: Optional[str] = None

    def __post_init__(self):
        if self.beam_size < 1:
            raise ValueError("beam_size must be at least 1")
        if not isinstance(self.temperatures, (list, tuple)):
            raise TypeError("temperatures must be a list or tuple of floats")
        if any(not isinstance(t, (int, float)) for t in self.temperatures):
            raise TypeError("all temperatures must be numeric")


class BatchedInferencePipeline:

    def __init__(
        self,
        model,
        config: Optional[TranscriptionConfig] = None,
        tokenizer=None,
        language: Optional[str] = None,
    ):
        self.model: WhisperModel = model
        self.tokenizer = tokenizer
        self.config = config or TranscriptionConfig()
        self.preset_language = language
        self.last_speech_timestamp = 0.0

    def forward(
        self,
        features: torch.Tensor,
        chunks_metadata: List[dict],
    ) -> Tuple[ctranslate2.StorageView, List[dict]]:
        encoder_output, outputs = self.model.generate_segment_batched(
            features, 
            self.tokenizer, 
            self.config
        )

        segmented_outputs = []
        segment_sizes = []
        for chunk_metadata, output in zip(chunks_metadata, outputs):
            duration = chunk_metadata["end_time"] - chunk_metadata["start_time"]
            segment_size = int(ceil(duration) * self.model.frames_per_second)
            segment_sizes.append(segment_size)
            
            subsegments, seek, single_timestamp_ending = (
                self.model._split_segments_by_timestamps(
                    tokenizer=self.tokenizer,
                    tokens=output["tokens"],
                    time_offset=chunk_metadata["start_time"],
                    segment_size=segment_size,
                    segment_duration=duration,
                    seek=0,
                )
            )
            
            segmented_outputs.append([
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
            ])
            
        if self.config.word_timestamps:
            self.last_speech_timestamp = self.model.add_word_timestamps(
                segmented_outputs,
                self.tokenizer,
                encoder_output,
                segment_sizes,
                self.config.prepend_punctuations,
                self.config.append_punctuations,
                self.last_speech_timestamp,
            )

        return encoder_output, segmented_outputs

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

    def transcribe(
        self,
        audio: Union[str, BinaryIO, torch.Tensor, np.ndarray],
        log_progress: bool = False,
        batch_size: int = 16,
    ) -> Tuple[Iterable[Segment], 'TranscriptionInfo']:
        options = self.config
        sampling_rate = self.model.feature_extractor.sampling_rate

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        elif not isinstance(audio, torch.Tensor):
            audio = decode_audio(audio, sampling_rate=sampling_rate)
        duration = audio.shape[0] / sampling_rate

        chunk_length = options.chunk_length or self.model.feature_extractor.chunk_length
        clip_timestamps = options.clip_timestamps
        if not clip_timestamps:
            vad_parameters = None
            if options.vad_filter:
                if options.vad_parameters is None:
                    vad_parameters = VadOptions(
                        max_speech_duration_s=chunk_length,
                        min_silence_duration_ms=160,
                    )
                elif isinstance(options.vad_parameters, dict):
                    if "max_speech_duration_s" in options.vad_parameters.keys():
                        options.vad_parameters.pop("max_speech_duration_s")

                    vad_parameters = VadOptions(
                        **options.vad_parameters, max_speech_duration_s=chunk_length
                    )

                active_segments = get_speech_timestamps(audio, vad_parameters)
                clip_timestamps = merge_segments(active_segments, vad_parameters)
            elif duration < chunk_length:
                clip_timestamps = [{"start": 0, "end": audio.shape[0]}]
            else:
                raise RuntimeError(
                    "No clip timestamps found. "
                    "Set 'vad_filter' to True or provide 'clip_timestamps'."
                )
        if self.model.model.is_multilingual:
            language = options.output_language or self.preset_language
        elif options.output_language != "en":
            if options.output_language is not None:
                self.model.logger.warning(
                    f"English-only model is used, but {options.output_language} language is"
                    " chosen, setting language to 'en'."
                )
            language = "en"
        else:
            language = "en"

        (
            language,
            language_probability,
            task,
            all_language_probs,
        ) = self.get_language_and_tokenizer(audio, None, language)

        duration_after_vad = (
            sum((segment["end"] - segment["start"]) for segment in clip_timestamps)
            / sampling_rate
        )

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=options,
            vad_options=options.vad_parameters,
            all_language_probs=all_language_probs,
        )

        audio_chunks, chunks_metadata = collect_chunks(audio, clip_timestamps)
        to_cpu = (
            self.model.model.device == "cuda" and len(self.model.model.device_index) > 1
        )
        features = (
            torch.stack(
                [
                    self.model.feature_extractor(chunk, to_cpu=to_cpu)[
                        ..., : self.model.feature_extractor.nb_max_frames
                    ]
                    for chunk in audio_chunks
                ]
            )
            if duration_after_vad
            else []
        )

        segments = self._batched_segments_generator(
            features,
            chunks_metadata,
            batch_size,
            options,
            log_progress,
        )

        return segments, info

    def _batched_segments_generator(
        self, 
        features: torch.Tensor,
        chunks_metadata: List[dict],
        batch_size: int,
        options: TranscriptionConfig,
        log_progress: bool
    ) -> Iterator[Segment]:
        if len(features) == 0:
            return
            
        pbar = tqdm(total=len(features), disable=not log_progress, position=0)
        seg_idx = 0
        
        for i in range(0, len(features), batch_size):
            encoder_output, segmented_outputs = self.forward(
                features[i : i + batch_size],
                chunks_metadata[i : i + batch_size],
            )

            for batch_segments in segmented_outputs:
                for segment in batch_segments:
                    seg_idx += 1
                    yield Segment(
                        seek=int(batch_segments[-1]["end"] * self.model.frames_per_second),
                        id=seg_idx,
                        text=segment["text"],
                        start=round(segment["start"], 3),
                        end=round(segment["end"], 3),
                        words=(
                            None
                            if not options.word_timestamps
                            else [Word(**word) for word in segment.get("words", [])]
                        ),
                        tokens=segment["tokens"],
                        avg_logprob=segment["avg_logprob"],
                        no_speech_prob=segment["no_speech_prob"],
                        compression_ratio=segment["compression_ratio"],
                    )
                pbar.update(1)

        pbar.close()
        if self.preset_language is None:
            self.tokenizer = None
        self.last_speech_timestamp = 0.0


class TranscriptionInfo(NamedTuple):
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionConfig
    vad_options: Optional[VadOptions]


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
        config: Optional[TranscriptionConfig] = None,
        **model_kwargs,
    ):
        self.logger = get_logger()
        self.config = config or TranscriptionConfig()

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
        config: Optional[TranscriptionConfig] = None,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        config = config or self.config
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

        if config.vad_filter and config.clip_timestamps == "0":
            vad_parameters = None
            if config.vad_parameters is None:
                vad_parameters = VadOptions()
            elif isinstance(config.vad_parameters, dict):
                vad_parameters = VadOptions(**config.vad_parameters)
            speech_chunks = get_speech_timestamps(audio, vad_parameters)
            audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
            audio = torch.cat(audio_chunks, dim=0)
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
            audio, chunk_length=config.chunk_length, to_cpu=to_cpu
        )

        encoder_output = None
        all_language_probs = None

        language = config.output_language
        language_probability = 1

        if language is None:
            if not self.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                language, language_probability, all_language_probs = self.detect_language(
                    audio
                )

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

        tokenizer = Tokenizer(
            self.hf_tokenizer,
            self.model.is_multilingual,
            task="transcribe",
            language=language,
        )

        segments = self.generate_segments(features, tokenizer, config, encoder_output)

        if speech_chunks:
            segments = restore_speech_timestamps(segments, speech_chunks, sampling_rate)

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=config,
            vad_options=config.vad_parameters,
            all_language_probs=all_language_probs,
        )
        return segments, info

    def encode(self, features: torch.Tensor) -> ctranslate2.StorageView:
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

        if features.ndim == 2:
            features = features.unsqueeze(0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

    def generate_segments(
        self,
        features: torch.Tensor,
        tokenizer: Tokenizer,
        config: TranscriptionConfig,
        encoder_output: Optional[ctranslate2.StorageView] = None,
    ) -> Iterable[Segment]:
        content_frames = features.shape[-1] - self.feature_extractor.nb_max_frames
        content_duration = float(
            content_frames * self.feature_extractor.time_per_frame
        )

        if isinstance(config.clip_timestamps, str):
            config.clip_timestamps = [
                float(ts) for ts in config.clip_timestamps.split(",") if ts
            ]
        seek_points: List[int] = [
            round(ts * self.frames_per_second) for ts in config.clip_timestamps
        ]
        if len(seek_points) == 0:
            seek_points.append(0)
        if len(seek_points) % 2 == 1:
            seek_points.append(content_frames)
        seek_clips: List[Tuple[int, int]] = list(
            zip(seek_points[::2], seek_points[1::2])
        )

        idx = 0
        clip_idx = 0
        seek = seek_clips[clip_idx][0]
        all_tokens = []
        prompt_reset_since = 0

        if config.initial_prompt is not None:
            if isinstance(config.initial_prompt, str):
                initial_prompt = " " + config.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(config.initial_prompt)

        last_speech_timestamp = 0.0
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

            prompt = self.get_prompt(
                tokenizer,
                previous_tokens,
                without_timestamps=config.without_timestamps,
                prefix=config.prefix if seek == 0 else None,
                hotwords=config.hotwords,
            )

            if seek > 0 or encoder_output is None:
                encoder_output = self.encode(segment)

            (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            ) = self.generate_with_fallback(encoder_output, prompt, tokenizer, config)

            if config.no_speech_threshold is not None:
                should_skip = result.no_speech_prob > config.no_speech_threshold

                if (
                    config.log_prob_threshold is not None
                    and avg_logprob > config.log_prob_threshold
                ):
                    should_skip = False

                if should_skip:
                    self.logger.debug(
                        "No speech threshold is met (%f > %f)",
                        result.no_speech_prob,
                        config.no_speech_threshold,
                    )

                if config.log_prob_low_threshold:
                    if avg_logprob < config.log_prob_low_threshold:
                        should_skip = True
                        self.logger.debug(
                            "log prob low threshold is met (%f > %f)",
                            avg_logprob,
                            config.log_prob_low_threshold,
                        )

                if should_skip:
                    seek += segment_size
                    continue

            tokens = result.sequences_ids[0]

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

            if config.word_timestamps:
                last_speech_timestamp = self.add_word_timestamps(
                    [current_segments],
                    tokenizer,
                    encoder_output,
                    segment_size,
                    config.prepend_punctuations,
                    config.append_punctuations,
                    last_speech_timestamp=last_speech_timestamp,
                )
                if not single_timestamp_ending:
                    last_word_end = get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset:
                        seek = round(last_word_end * self.frames_per_second)

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
                        [Word(**word) for word in segment.get("words", [])]
                        if config.word_timestamps
                        else None
                    ),
                )

            if (
                not config.condition_on_previous_text
                or temperature > config.prompt_reset_on_temperature
            ):
                if config.condition_on_previous_text:
                    self.logger.debug(
                        "Reset prompt. prompt_reset_on_temperature threshold is met %f > %f",
                        temperature,
                        config.prompt_reset_on_temperature,
                    )

                prompt_reset_since = len(all_tokens)

    def generate_with_fallback(
        self,
        encoder_output: ctranslate2.StorageView,
        prompt: List[int],
        tokenizer: Tokenizer,
        config: TranscriptionConfig,
    ) -> Tuple[ctranslate2.models.WhisperGenerationResult, float, float, float]:
        decode_result = None
        all_results = []
        below_cr_threshold_results = []

        max_initial_timestamp_index = int(
            round(config.max_initial_timestamp / self.time_precision)
        )
        if config.max_new_tokens is not None:
            max_length = len(prompt) + config.max_new_tokens
        else:
            max_length = self.max_length

        if max_length > self.max_length:
            raise ValueError(
                f"The length of the prompt is {len(prompt)}, and the max_new_tokens "
                f"{max_length - len(prompt)}. Thus, the combined length of the prompt "
                f"and max_new_tokens is: {max_length}. This exceeds the "
                f"max_length of the Whisper model: {self.max_length}. "
                "You should either reduce the length of your prompt, or "
                "reduce the value of max_new_tokens, "
                f"so that their combined length is less that {self.max_length}."
            )

        for temperature in config.temperatures:
            if temperature > 0:
                kwargs = {
                    "beam_size": 1,
                    "num_hypotheses": config.best_of,
                    "sampling_topk": 0,
                    "sampling_temperature": temperature,
                }
            else:
                kwargs = {
                    "beam_size": config.beam_size,
                    "patience": config.patience,
                }

            result = self.model.generate(
                encoder_output,
                [prompt],
                length_penalty=config.length_penalty,
                repetition_penalty=config.repetition_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                max_length=max_length,
                return_scores=True,
                return_no_speech_prob=True,
                suppress_blank=config.suppress_blank,
                suppress_tokens=get_suppressed_tokens(tokenizer, config.suppress_tokens),
                max_initial_timestamp_index=max_initial_timestamp_index,
                **kwargs,
            )[0]

            tokens = result.sequences_ids[0]

            seq_len = len(tokens)
            cum_logprob = result.scores[0] * (seq_len ** config.length_penalty)
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

            if config.compression_ratio_threshold is not None:
                if compression_ratio > config.compression_ratio_threshold:
                    needs_fallback = True

                    self.logger.debug(
                        "Compression ratio threshold is not met with temperature %.1f (%f > %f)",
                        temperature,
                        compression_ratio,
                        config.compression_ratio_threshold,
                    )
                else:
                    below_cr_threshold_results.append(decode_result)

            if (
                config.log_prob_threshold is not None
                and avg_logprob < config.log_prob_threshold
            ):
                needs_fallback = True

                self.logger.debug(
                    "Log probability threshold is not met with temperature %.1f (%f < %f)",
                    temperature,
                    avg_logprob,
                    config.log_prob_threshold,
                )

            if (
                config.no_speech_threshold is not None
                and result.no_speech_prob > config.no_speech_threshold
                and config.log_prob_threshold is not None
                and avg_logprob < config.log_prob_threshold
            ):
                needs_fallback = False

            if not needs_fallback:
                break
        else:
            decode_result = max(
                below_cr_threshold_results or all_results, key=lambda x: x[1]
            )
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

    def _split_segments_by_timestamps(
        self,
        tokenizer: Tokenizer,
        tokens: List[int],
        time_offset: float,
        segment_size: int,
        segment_duration: float,
        seek: int,
    ) -> Tuple[List[dict], int, bool]:
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
                seek += segment_size
            else:
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

    def add_word_timestamps(
        self,
        segments: List[List[dict]],
        tokenizer: Tokenizer,
        encoder_output: ctranslate2.StorageView,
        num_frames: Union[int, List[int]],
        prepend_punctuations: str,
        append_punctuations: str,
        last_speech_timestamp: float,
    ) -> float:
        if len(segments) == 0:
            return last_speech_timestamp

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

            if len(word_durations) > 0:
                sentence_end_marks = ".。!！?？"
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

                if len(words) > 0:
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
        text_tokens: List[List[int]],
        encoder_output: ctranslate2.StorageView,
        num_frames: Union[int, List[int]],
        median_filter_width: int = 7,
    ) -> List[List[dict]]:
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
        for result, tokens in zip(results, text_tokens):
            text_token_probs = result.text_token_probs
            alignments = result.alignments
            text_indices = np.array([pair[0] for pair in alignments])
            time_indices = np.array([pair[1] for pair in alignments])

            words, word_tokens = tokenizer.split_to_word_tokens(tokens + [tokenizer.eot])
            if len(word_tokens) <= 1:
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
                        tokens=wtokens,
                        start=start,
                        end=end,
                        probability=probability,
                    )
                    for word, wtokens, start, end, probability in zip(
                        words, word_tokens, start_times, end_times, word_probabilities
                    )
                ]
            )
        return return_list

    def generate_segment_batched(
        self,
        features: torch.Tensor,
        tokenizer: Tokenizer,
        options: TranscriptionConfig,
    ):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0

        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(options.initial_prompt)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        result = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            beam_size=options.beam_size,
            patience=options.patience,
            length_penalty=options.length_penalty,
            max_length=self.max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=get_suppressed_tokens(tokenizer, options.suppress_tokens),
            return_scores=True,
            return_no_speech_prob=True,
        )

        output = []
        for res in result:
            seq_len = len(res.sequences_ids[0])
            cum_logprob = res.scores[0] * (seq_len ** options.length_penalty)
            avg_logprob = cum_logprob / (seq_len + 1)
            output.append({
                "avg_logprob": avg_logprob,
                "no_speech_prob": res.no_speech_prob,
                "tokens": res.sequences_ids[0],
            })

        return encoder_output, output

    def detect_language(
        self, audio: torch.Tensor
    ) -> Tuple[str, float, Optional[List[Tuple[str, float]]]]:
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

    def detect_language_multi_segment(self, audio: Union[str, BinaryIO, torch.Tensor]):
        config = self.config

        if not config.multilingual:
            self.logger.warning(
                "Language detection is not supported for non-multilingual models; defaulting to the major language."
            )

        speech_percentage_threshold = config.speech_percentage_threshold
        language_threshold = config.language_threshold
        num_detection_segments = config.language_detection_segments
        vad_filter_enabled = config.vad_filter
        vad_params = {
            'min_silence_duration_ms': config.vad_min_silence_duration
        }

        if vad_filter_enabled:
            vad_options = VadOptions(**vad_params)
        else:
            vad_options = None

        sampling_rate = self.feature_extractor.sampling_rate
        if not isinstance(audio, torch.Tensor):
            audio = decode_audio(audio, sampling_rate=sampling_rate)

        duration = audio.shape[0] / sampling_rate

        if vad_filter_enabled:
            speech_chunks = get_speech_timestamps(audio, vad_options)
            audio_chunks, _ = collect_chunks(audio, speech_chunks)
            audio = torch.cat(audio_chunks, dim=0)

            duration_vad = audio.shape[0] / sampling_rate

            self.logger.debug(
                f"Language detection: VAD filter removed {duration - duration_vad:.2f} sec of audio"
            )

            if duration_vad / duration < speech_percentage_threshold:
                return {"language_code": None, "language_confidence": 1.0}

            duration = duration_vad

        if duration < 1.0:
            return {"language_code": None, "language_confidence": 1.0}

        nb_max_frames = self.feature_extractor.nb_max_frames

        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        features = self.feature_extractor(audio, to_cpu=to_cpu)

        num_segments = features.shape[-1] // nb_max_frames
        if num_detection_segments > num_segments:
            self.logger.warning(
                f"Language detection: Cannot have more segments; setting to {num_segments} segments."
            )
            num_detection_segments = num_segments

        indices = list(range(num_segments))
        random.seed(0)
        random.shuffle(indices)
        indices = indices[:num_detection_segments]

        detected_languages = []
        all_language_probabilities = defaultdict(list)
        confident_language_probabilities = defaultdict(list)
        num_confident_segments_per_language = defaultdict(int)

        for i in indices:
            segment_features = features[:, i * nb_max_frames : (i + 1) * nb_max_frames]
            try:
                encoder_output = self.encode(segment_features)
                results = self.model.detect_language(encoder_output)[0]
            except ValueError as e:
                self.logger.error(f"Inference error: {e}")
                continue

            language_token = results[0][0]
            language = language_token[2:-2]
            language_probability = results[0][1]

            detected_languages.append(language)
            all_language_probabilities[language].append(language_probability)

            if language_probability > language_threshold:
                num_confident_segments_per_language[language] += 1
                confident_language_probabilities[language].append(language_probability)

                if (
                    num_confident_segments_per_language[language]
                    >= num_detection_segments
                ):
                    mean_confidence = sum(confident_language_probabilities[language]) / len(
                        confident_language_probabilities[language]
                    )
                    return {
                        "language_code": language,
                        "language_confidence": mean_confidence,
                    }

        counter = Counter(detected_languages)

        def key_func(lang):
            frequency = counter[lang]
            prob_avg = sum(all_language_probabilities[lang]) / len(
                all_language_probabilities[lang]
            )
            return frequency, prob_avg

        if detected_languages:
            max_language = max(detected_languages, key=key_func)
            max_probability = sum(all_language_probabilities[max_language]) / len(
                all_language_probabilities[max_language]
            )

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

        return {"language_code": None, "language_confidence": 1.0}


def get_ctranslate2_storage(segment: torch.Tensor) -> ctranslate2.StorageView:
    segment = segment.contiguous()
    segment = ctranslate2.StorageView.from_array(
        segment if segment.is_cuda else segment.numpy()
    )
    return segment

def get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))

def get_suppressed_tokens(
    tokenizer: Tokenizer,
    suppress_tokens: Optional[List[int]],
) -> Optional[List[int]]:
    if suppress_tokens is None:
        suppress_tokens = []
    elif -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
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

def merge_punctuations(alignment: List[dict], prepended: str, appended: str) -> None:
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous["word"].startswith(" ") and previous["word"].strip() in prepended:
            following["word"] = previous["word"] + following["word"]
            if "tokens" in alignment[0].keys():
                following["tokens"] = previous["tokens"] + following["tokens"]
                previous["tokens"] = []
            previous["word"] = ""

        else:
            j = i
        i -= 1

    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous["word"].endswith(" ") and following["word"] in appended:
            previous["word"] = previous["word"] + following["word"]
            if "tokens" in alignment[0].keys():
                previous["tokens"] = previous["tokens"] + following["tokens"]
                following["tokens"] = []
            following["word"] = ""

        else:
            i = j
        j += 1
