import itertools
import logging
import math
import os
import zlib

from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union

import ctranslate2
import numpy as np
import tokenizers
import torch

from faster_whisper.audio import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
from faster_whisper.utils import download_model, format_timestamp, get_logger
from faster_whisper.vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
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
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]


class TranscriptionOptions(NamedTuple):
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
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


class TranscriptionInfo(NamedTuple):
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionOptions
    vad_options: VadOptions


class WhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        device: str = "auto",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
    ):
        """Initializes the Whisper model.

        Args:
          model_size_or_path: Size of the model to use (tiny, tiny.en, base, base.en,
            small, small.en, medium, medium.en, large-v1, large-v2, or large), a path to a converted
            model directory, or a CTranslate2-converted Whisper model ID from the Hugging Face Hub.
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
        """
        self.logger = get_logger()

        if os.path.isdir(model_size_or_path):
            model_path = model_size_or_path
        else:
            model_path = download_model(
                model_size_or_path,
                local_files_only=local_files_only,
                cache_dir=download_root,
            )

        self.model = ctranslate2.models.Whisper(
            model_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            intra_threads=cpu_threads,
            inter_threads=num_workers,
        )

        tokenizer_file = os.path.join(model_path, "tokenizer.json")
        if os.path.isfile(tokenizer_file):
            self.hf_tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
        else:
            self.hf_tokenizer = tokenizers.Tokenizer.from_pretrained(
                "openai/whisper-tiny" + ("" if self.model.is_multilingual else ".en")
            )

        self.feature_extractor = FeatureExtractor()
        self.num_samples_per_token = self.feature_extractor.hop_length * 2
        self.frames_per_second = (
            self.feature_extractor.sampling_rate // self.feature_extractor.hop_length
        )
        self.tokens_per_second = (
            self.feature_extractor.sampling_rate // self.num_samples_per_token
        )
        self.input_stride = 2
        self.time_precision = 0.02
        self.max_length = 448

    @property
    def supported_languages(self) -> List[str]:
        """The languages supported by the model."""
        return list(_LANGUAGE_CODES) if self.model.is_multilingual else ["en"]

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        batch_size: int = 2,
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
        vad_filter: bool = False,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribes an input file.

        Arguments:
          audio: Path to the input file (or a file-like object), or the audio waveform.
          language: The language spoken in the audio. It should be a language code such
            as "en" or "fr". If not set, the language will be detected in the first 30 seconds
            of audio.
          task: Task to execute (transcribe or translate).
          beam_size: Beam size to use for decoding.
          batch_size: Batch size to use for decoding.
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
            of symbols as defined in the model config.json file.
          without_timestamps: Only sample text tokens.
          max_initial_timestamp: The initial timestamp cannot be later than this.
          word_timestamps: Extract word-level timestamps using the cross-attention pattern
            and dynamic time warping, and include the timestamps for each word in each segment.
          prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the next word
          append_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the previous word
          vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
            without speech. This step is using the Silero VAD model
            https://github.com/snakers4/silero-vad.
          vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
            parameters and default values in the class `VadOptions`).

        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """
        sampling_rate = self.feature_extractor.sampling_rate

        if not isinstance(audio, np.ndarray):
            audio = decode_audio(audio, sampling_rate=sampling_rate)

        duration = audio.shape[0] / sampling_rate
        duration_after_vad = duration

        self.logger.info(
            "Processing audio with duration %s", format_timestamp(duration)
        )

        if vad_filter:
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

        features = self.feature_extractor(audio)
        features, frame_offsets = split_audio(features, batch_size)
        time_offsets = [
            frame_offset * self.feature_extractor.time_per_frame
            for frame_offset in frame_offsets
        ]

        encoder_output = None
        all_language_probs = None
        if language is None:
            if not self.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                segment = features[:, :, : self.feature_extractor.nb_max_frames]
                encoder_output = self.encode(segment)

                # results is a list of tuple[str, float] with language names and
                # probabilities.
                results = self.model.detect_language(encoder_output)[0]
                # Parse language names to strip out markers
                all_language_probs = [(token[2:-2], prob) for (token, prob) in results]
                # Get top language token and probability
                language, language_probability = all_language_probs[0]

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
            suppress_tokens=get_suppressed_tokens(tokenizer, suppress_tokens),
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
        )

        g_batch_segments = self.generate_segments(
            features, tokenizer, options, encoder_output
        )
        segments = from_batch_to_single(g_batch_segments, time_offsets)

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

    def generate_segments(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        encoder_output: Optional[ctranslate2.StorageView] = None,
    ) -> Iterable[Iterable[Segment]]:
        g_batch_content_frames = (
            np.count_nonzero(features[:, 0, :], axis=-1)
            - self.feature_extractor.nb_max_frames
        )
        batch_size = features.shape[0]
        g_idx = [0 for _ in range(batch_size)]
        g_batch_seek = [0 for _ in range(batch_size)]
        g_batch_previous_seek = [0 for _ in range(batch_size)]
        g_all_tokens = [[] for _ in range(batch_size)]
        g_batch_segments = [[] for _ in range(batch_size)]
        g_prompt_reset_since = [0 for _ in range(batch_size)]
        g_batch_is_done = [False for _ in range(batch_size)]

        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                g_all_tokens[0].extend(initial_prompt_tokens)
            else:
                g_all_tokens[0].extend(options.initial_prompt)

        g_batch_last_speech_timestamp = [0 for _ in range(batch_size)]
        while any(
            seek < content_frames
            for seek, content_frames in zip(g_batch_seek, g_batch_content_frames)
        ):
            g_batch_is_done = [
                is_done or seek >= content_frames
                for is_done, seek, content_frames in zip(
                    g_batch_is_done, g_batch_seek, g_batch_content_frames
                )
            ]
            g_batch_index_conversion = [
                i for i, is_done in enumerate(g_batch_is_done) if not is_done
            ]
            batch_time_offset = [
                seek * self.feature_extractor.time_per_frame
                for seek, is_done in zip(g_batch_seek, g_batch_is_done)
                if not is_done
            ]
            batch_last_speech_timestamp = [
                last_speech_timestamp
                for last_speech_timestamp, is_done in zip(
                    g_batch_last_speech_timestamp, g_batch_is_done
                )
                if not is_done
            ]

            segment_size = min(
                [
                    min(self.feature_extractor.nb_max_frames, content_frames - seek)
                    for seek, content_frames, is_done in zip(
                        g_batch_seek, g_batch_content_frames, g_batch_is_done
                    )
                    if not is_done
                ]
            )
            segment = np.stack(
                [
                    features[i, :, seek : seek + segment_size]
                    for i, (seek, is_done) in enumerate(
                        zip(g_batch_seek, g_batch_is_done)
                    )
                    if not is_done
                ]
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Processing segment at %s", format_timestamp(batch_time_offset)
                )

            batch_previous_tokens = [
                tokens[g_prompt_reset_since[batch_index] :]
                for batch_index, (tokens, is_done) in enumerate(
                    zip(g_all_tokens, g_batch_is_done)
                )
                if not is_done
            ]
            batch_prompt = self.get_prompt(
                tokenizer,
                batch_previous_tokens,
                without_timestamps=options.without_timestamps,
                prefix=options.prefix if g_batch_seek[0] == 0 else None,
            )

            # not first step of decoding or no prompt
            if g_batch_seek[0] > 0 or encoder_output is None:
                encoder_output = self.encode(segment)

            batch_results = self.generate_with_fallback(
                encoder_output, batch_prompt, tokenizer, options
            )
            batch_current_segments = [None for _ in g_batch_index_conversion]
            batch_tokens = [result[0].sequences_ids[0] for result in batch_results]
            skipped_batch_index = set()
            for batch_index, (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            ) in enumerate(batch_results):
                global_batch_index = g_batch_index_conversion[batch_index]
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

                        # fast-forward to the next segment boundary
                        g_batch_seek[global_batch_index] += segment_size
                        skipped_batch_index.add(batch_index)
                        continue

                g_batch_previous_seek[global_batch_index] = g_batch_seek[
                    global_batch_index
                ]

                current_segments, seek = self.compute_current_segments_and_next_seek(
                    g_batch_seek[global_batch_index],
                    batch_tokens[batch_index],
                    tokenizer.timestamp_begin,
                    segment_size,
                    batch_time_offset[batch_index],
                )
                g_batch_seek[global_batch_index] = seek
                batch_current_segments[batch_index] = current_segments

            if len(g_batch_index_conversion) == len(skipped_batch_index):
                continue

            partial_encoder_output = encoder_output
            partial_batch_current_segments = batch_current_segments
            partial_batch_time_offset = batch_time_offset

            if options.word_timestamps:
                self.add_word_timestamps(
                    partial_batch_current_segments,
                    tokenizer,
                    partial_encoder_output,
                    partial_batch_time_offset,
                    segment_size,
                    options.prepend_punctuations,
                    options.append_punctuations,
                    batch_last_speech_timestamp,
                    skipped_batch_index,
                )

                for partial_batch_index, current_segments in enumerate(
                    partial_batch_current_segments
                ):
                    if partial_batch_index in skipped_batch_index:
                        continue
                    global_batch_index = g_batch_index_conversion[partial_batch_index]
                    word_end_timestamps = [
                        w["end"] for s in current_segments for w in s["words"]
                    ]
                    single_timestamp_ending = ends_with_single_timestamp(
                        batch_tokens[partial_batch_index],
                        tokenizer.timestamp_begin,
                    )

                    if len(word_end_timestamps) > 0:
                        g_batch_last_speech_timestamp[
                            global_batch_index
                        ] = word_end_timestamps[-1]
                    if not single_timestamp_ending and len(word_end_timestamps) > 0:
                        seek_shift = round(
                            (
                                word_end_timestamps[-1]
                                - batch_time_offset[partial_batch_index]
                            )
                            * self.frames_per_second
                        )

                        if seek_shift > 0:
                            g_batch_seek[global_batch_index] = (
                                g_batch_previous_seek[global_batch_index] + seek_shift
                            )

            for partial_batch_index, current_segments in enumerate(
                partial_batch_current_segments
            ):
                if partial_batch_index in skipped_batch_index:
                    continue

                def segment_iterator(current_segments, g_batch_index):
                    for segment in current_segments:
                        tokens = segment["tokens"]
                        text = tokenizer.decode(tokens)

                        if segment["start"] == segment["end"] or not text.strip():
                            continue

                        g_all_tokens[g_batch_index].extend(tokens)
                        g_idx[g_batch_index] += 1

                        yield Segment(
                            id=g_idx[g_batch_index],
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

                g_batch_segments[g_batch_index_conversion[partial_batch_index]].append(
                    segment_iterator(
                        current_segments,
                        g_batch_index_conversion[partial_batch_index],
                    )
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

                    g_prompt_reset_since[
                        g_batch_index_conversion[partial_batch_index]
                    ] = len(g_all_tokens)
        return [
            itertools.chain.from_iterable(segments) for segments in g_batch_segments
        ]

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

        # features = np.expand_dims(features, 0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

    def generate_with_fallback(
        self,
        encoder_output: ctranslate2.StorageView,
        batch_prompt: List[List[int]],
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
    ) -> List[Tuple[ctranslate2.models.WhisperGenerationResult, float, float, float]]:
        batch_decode_results = [None for _ in range(encoder_output.shape[0])]
        all_results = [[] for _ in range(encoder_output.shape[0])]
        below_cr_threshold_results = []

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )
        convert_index = list(range(encoder_output.shape[0]))
        index_with_fallback = list(range(encoder_output.shape[0]))
        torch_encoder_output = None
        unfinished_memory = (None, None)
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

            unfinished_encoder_output = encoder_output
            unfinished_batch_prompt = batch_prompt
            if len(convert_index) != encoder_output.shape[0]:
                if torch_encoder_output is None:
                    torch_encoder_output = torch.as_tensor(
                        encoder_output, device=encoder_output.device
                    ).contiguous()
                if unfinished_memory[0] != tuple(convert_index):
                    unfinished_memory = (
                        tuple(convert_index),
                        ctranslate2.StorageView.from_array(
                            torch_encoder_output[convert_index]
                        ),
                    )
                unfinished_encoder_output = unfinished_memory[1]
                unfinished_batch_prompt = [batch_prompt[i] for i in convert_index]
            results = self.model.generate(
                unfinished_encoder_output,
                unfinished_batch_prompt,
                length_penalty=options.length_penalty,
                repetition_penalty=options.repetition_penalty,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                max_length=self.max_length,
                return_scores=True,
                return_no_speech_prob=True,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                max_initial_timestamp_index=max_initial_timestamp_index,
                **kwargs,
            )
            index_with_fallback = set()

            for batch_index, result in zip(convert_index, results):
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
                all_results[batch_index].append(decode_result)

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
                ):
                    needs_fallback = False  # silence

                batch_decode_results[batch_index] = decode_result
                if needs_fallback:
                    # all failed, select the result with the highest average log probability
                    decode_result = max(
                        below_cr_threshold_results or all_results[batch_index],
                        key=lambda x: x[1],
                    )
                    index_with_fallback.add(batch_index)
            convert_index = list(index_with_fallback)
            if not index_with_fallback:
                break
        return batch_decode_results

    def get_prompt(
        self,
        tokenizer: Tokenizer,
        batch_previous_tokens: List[List[int]],
        without_timestamps: bool = False,
        prefix: Optional[str] = None,
    ) -> List[List[int]]:
        batch_prompt = []
        # in ctranslate2 the <|startoftranscript|> must be at the same posiion
        min_length_previous_tokens = min(
            [len(previous_tokens) for previous_tokens in batch_previous_tokens]
        )
        for previous_tokens in batch_previous_tokens:
            previous_tokens = (
                previous_tokens[-min_length_previous_tokens:]
                if min_length_previous_tokens
                else []
            )
            prompt = []

            if previous_tokens:
                prompt.append(tokenizer.sot_prev)
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

            batch_prompt.append(prompt)
        return batch_prompt

    def add_word_timestamps(
        self,
        g_batch_segments: List[List[dict]],
        tokenizer: Tokenizer,
        encoder_output: ctranslate2.StorageView,
        batch_time_offset: List[int],
        num_frames: int,
        prepend_punctuations: str,
        append_punctuations: str,
        last_speech_timestamp: List[float],
        skipped_batch_index: set,
    ) -> None:
        if all(
            (segments is None or len(segments) == 0) for segments in g_batch_segments
        ):
            return

        batch_text_tokens_per_segment = [
            [
                [token for token in segment["tokens"] if token < tokenizer.eot]
                for segment in segments
            ]
            if segments
            else []
            for segments in g_batch_segments
        ]

        text_tokens = [
            list(itertools.chain.from_iterable(text_tokens_per_segment))
            for text_tokens_per_segment in batch_text_tokens_per_segment
        ]

        for batch_index, tokens in enumerate(text_tokens):
            if not tokens:
                text_tokens[batch_index] = [tokenizer.eot - 1]
                skipped_batch_index.add(batch_index)

        batch_alignment = self.find_alignment(
            tokenizer, text_tokens, encoder_output, num_frames
        )

        for batch_index, segments in enumerate(g_batch_segments):
            if batch_index in skipped_batch_index:
                continue
            word_durations = np.array(
                [word["end"] - word["start"] for word in batch_alignment[batch_index]]
            )
            word_durations = word_durations[word_durations.nonzero()]
            median_duration = (
                np.median(word_durations) if len(word_durations) > 0 else 0.0
            )
            max_duration = median_duration * 2

            # hack: truncate long words at sentence boundaries.
            # a better segmentation algorithm based on VAD should be able to replace this.
            if len(word_durations) > 0:
                sentence_end_marks = ".。!！?？"
                # ensure words at sentence boundaries
                # are not longer than twice the median word duration.
                for i in range(1, len(batch_alignment[batch_index])):
                    if (
                        batch_alignment[batch_index][i]["end"]
                        - batch_alignment[batch_index][i]["start"]
                        > max_duration
                    ):
                        if (
                            batch_alignment[batch_index][i]["word"]
                            in sentence_end_marks
                        ):
                            batch_alignment[batch_index][i]["end"] = (
                                batch_alignment[batch_index][i]["start"] + max_duration
                            )
                        elif (
                            batch_alignment[batch_index][i - 1]["word"]
                            in sentence_end_marks
                        ):
                            batch_alignment[batch_index][i]["start"] = (
                                batch_alignment[batch_index][i]["end"] - max_duration
                            )

            merge_punctuations(
                batch_alignment[batch_index],
                prepend_punctuations,
                append_punctuations,
            )

            batch_time_offset[batch_index] = (
                segments[0]["seek"]
                * self.feature_extractor.hop_length
                / self.feature_extractor.sampling_rate
            )

            word_index = 0

            for segment_index, (segment, text_tokens) in enumerate(
                zip(segments, batch_text_tokens_per_segment[batch_index])
            ):
                saved_tokens = 0
                words = []

                while word_index < len(
                    batch_alignment[batch_index]
                ) and saved_tokens < len(text_tokens):
                    timing = batch_alignment[batch_index][word_index]

                    if timing["word"]:
                        words.append(
                            dict(
                                word=timing["word"],
                                start=round(
                                    batch_time_offset[batch_index] + timing["start"], 2
                                ),
                                end=round(
                                    batch_time_offset[batch_index] + timing["end"], 2
                                ),
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
                    if words[0]["end"] - last_speech_timestamp[
                        batch_index
                    ] > median_duration * 4 and (
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
                        segment["start"] < words[0]["end"]
                        and segment["start"] - 0.5 > words[0]["start"]
                    ):
                        words[0]["start"] = max(
                            0, min(words[0]["end"] - median_duration, segment["start"])
                        )
                    else:
                        segment["start"] = words[0]["start"]

                    # prefer the segment-level end timestamp if the last word is too long.
                    if (
                        segment["end"] > words[-1]["start"]
                        and segment["end"] + 0.5 < words[-1]["end"]
                    ):
                        words[-1]["end"] = max(
                            words[-1]["start"] + median_duration, segment["end"]
                        )
                    else:
                        segment["end"] = words[-1]["end"]

                    last_speech_timestamp[batch_index] = segment["end"]

                g_batch_segments[batch_index][segment_index]["words"] = words

    def find_alignment(
        self,
        tokenizer: Tokenizer,
        batch_text_tokens: List[List[int]],
        encoder_output: ctranslate2.StorageView,
        num_frames: int,
        median_filter_width: int = 7,
    ) -> List[List[dict]]:
        if len(batch_text_tokens) == 0:
            return []

        results = self.model.align(
            encoder_output,
            tokenizer.sot_sequence,
            batch_text_tokens,
            num_frames,
            median_filter_width=median_filter_width,
        )
        batch_alignment = []

        for text_tokens, result in zip(batch_text_tokens, results):
            text_token_probs = result.text_token_probs

            alignments = result.alignments
            text_indices = np.array([pair[0] for pair in alignments])
            time_indices = np.array([pair[1] for pair in alignments])

            words, word_tokens = tokenizer.split_to_word_tokens(
                text_tokens + [tokenizer.eot]
            )
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

            batch_alignment.append(
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
        return batch_alignment

    def compute_current_segments_and_next_seek(
        self, seek, tokens, timestamp_begin, segment_size, time_offset
    ):
        current_segments = []
        segment_duration = segment_size * self.feature_extractor.time_per_frame

        single_timestamp_ending = ends_with_single_timestamp(tokens, timestamp_begin)

        consecutive_timestamps = index_of_consecutive_timestamps(
            tokens, timestamp_begin
        )

        if len(consecutive_timestamps) > 0:
            slices = list(consecutive_timestamps)
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = sliced_tokens[0] - timestamp_begin
                end_timestamp_position = sliced_tokens[-1] - timestamp_begin
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
                return current_segments, seek + segment_size
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                last_timestamp_position = tokens[last_slice - 1] - timestamp_begin
                return (
                    current_segments,
                    seek + last_timestamp_position * self.input_stride,
                )

        else:
            duration = segment_duration
            timestamps = [token for token in tokens if token >= timestamp_begin]
            if len(timestamps) > 0 and timestamps[-1] != timestamp_begin:
                last_timestamp_position = timestamps[-1] - timestamp_begin
                duration = last_timestamp_position * self.time_precision

            current_segments.append(
                dict(
                    seek=seek,
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                )
            )

            return current_segments, seek + segment_size


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


def get_ctranslate2_storage(segment: np.ndarray) -> ctranslate2.StorageView:
    segment = np.ascontiguousarray(segment)
    segment = ctranslate2.StorageView.from_array(segment)
    return segment


def get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def get_suppressed_tokens(
    tokenizer: Tokenizer,
    suppress_tokens: Optional[List[int]],
) -> Optional[List[int]]:
    if not suppress_tokens or -1 in suppress_tokens:
        return suppress_tokens

    suppress_tokens = list(suppress_tokens)

    # Ensure the following special tokens are suppressed when the user does
    # not use the default set (-1).
    suppress_tokens.extend(
        [
            tokenizer.transcribe,
            tokenizer.translate,
            tokenizer.sot,
            tokenizer.sot_prev,
            tokenizer.sot_lm,
        ]
    )

    return sorted(set(suppress_tokens))


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
            following["tokens"] = previous["tokens"] + following["tokens"]
            previous["word"] = ""
            previous["tokens"] = []
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
            previous["tokens"] = previous["tokens"] + following["tokens"]
            following["word"] = ""
            following["tokens"] = []
        else:
            i = j
        j += 1


def split_audio(features, batch_size):
    remainder = features.shape[1] % batch_size
    if remainder != 0:
        remainder = batch_size - remainder
    features = np.pad(features, ((0, 0), (0, remainder)))
    offsets = [i * features.shape[-1] for i in range(features.shape[0])]
    return (
        np.swapaxes(
            np.swapaxes(features, 0, 1).reshape(
                (
                    batch_size,
                    math.ceil(features.shape[1] / batch_size),
                    features.shape[0],
                ),
            ),
            1,
            2,
        ),
        offsets,
    )


def ends_with_single_timestamp(tokens, timestamp_begin):
    return (
        len(tokens) >= 2
        and tokens[-2] < timestamp_begin
        and tokens[-1] >= timestamp_begin
    )


def index_of_consecutive_timestamps(tokens, timestamp_begin):
    return [
        i
        for i in range(len(tokens))
        if i > 0 and tokens[i] >= timestamp_begin and tokens[i - 1] >= timestamp_begin
    ]


def from_batch_to_single(
    g_batch_segments: List[Iterable[Segment]], offsets: List[float]
) -> Iterable[Segment]:
    for batch_index, segments in enumerate(g_batch_segments):
        for segment in segments:
            if segment.words:
                words = []
                for word in segment.words:
                    # Ensure the word start and end times are resolved to the same chunk.
                    word = word._replace(
                        start=word.start + offsets[batch_index],
                        end=word.end + offsets[batch_index],
                    )
                    words.append(word)

                segment = segment._replace(
                    start=words[0].start,
                    end=words[-1].end,
                    words=words,
                )

            else:
                segment = segment._replace(
                    start=segment.start + offsets[batch_index],
                    end=segment.end + offsets[batch_index],
                )

            yield segment
