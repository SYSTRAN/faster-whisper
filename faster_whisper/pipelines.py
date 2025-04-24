import asyncio
from math import ceil
from typing import BinaryIO, Iterable, List, Optional, Tuple, Union, AsyncGenerator

import numpy as np
from ctranslate2._ext import WhisperGenerationResultAsync
from tqdm.asyncio import tqdm as atqdm

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_compression_ratio, TranscriptionOptions, TranscriptionInfo, Segment, \
    get_suppressed_tokens, Word
from faster_whisper.utils import format_timestamp
from faster_whisper.vad import (
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
    merge_segments,
)


class AsyncBatchedInferencePipeline:
    def __init__(
            self,
            model,
    ):
        self.model: WhisperModel = model
        self.last_speech_timestamp = 0.0

    async def forward(self, features, tokenizer, chunks_metadata, options):
        encoder_output, outputs = await self.generate_segment_batched(
            features, tokenizer, options
        )

        segmented_outputs = []
        segment_sizes = []

        # Асинхронно обрабатываем каждый чанк
        async def process_chunk(chunk_metadata, output):
            duration = chunk_metadata["end_time"] - chunk_metadata["start_time"]
            segment_size = int(ceil(duration) * self.model.frames_per_second)

            (
                subsegments,
                seek,
                single_timestamp_ending,
            ) = self.model._split_segments_by_timestamps(
                tokenizer=tokenizer,
                tokens=output["tokens"],
                time_offset=chunk_metadata["start_time"],
                segment_size=segment_size,
                segment_duration=duration,
                seek=0,
            )

            return (
                segment_size,
                [
                    dict(
                        text=tokenizer.decode(subsegment["tokens"]),
                        avg_logprob=output["avg_logprob"],
                        no_speech_prob=output["no_speech_prob"],
                        tokens=subsegment["tokens"],
                        start=subsegment["start"],
                        end=subsegment["end"],
                        compression_ratio=get_compression_ratio(
                            tokenizer.decode(subsegment["tokens"])
                        ),
                        seek=int(
                            chunk_metadata["start_time"] * self.model.frames_per_second
                        ),
                    )
                    for subsegment in subsegments
                ]
            )

        # Запускаем обработку всех чанков параллельно
        tasks = [process_chunk(chunk_metadata, output)
                 for chunk_metadata, output in zip(chunks_metadata, outputs)]
        results = await asyncio.gather(*tasks)

        for segment_size, output in results:
            segment_sizes.append(segment_size)
            segmented_outputs.append(output)

        if options.word_timestamps:
            self.last_speech_timestamp = await self.add_word_timestamps_async(
                segmented_outputs,
                tokenizer,
                encoder_output,
                segment_sizes,
                options.prepend_punctuations,
                options.append_punctuations,
                self.last_speech_timestamp,
            )

        return segmented_outputs

    async def add_word_timestamps_async(self, *args, **kwargs):
        # Асинхронная обертка для метода add_word_timestamps
        # Предполагается, что внутренняя реализация может быть интенсивной
        return await asyncio.to_thread(
            self.model.add_word_timestamps, *args, **kwargs
        )

    async def generate_segment_batched(
            self,
            features: np.ndarray,
            tokenizer: Tokenizer,
            options: TranscriptionOptions,
    ):
        batch_size = features.shape[0]

        prompt = self.model.get_prompt(
            tokenizer,
            previous_tokens=(
                tokenizer.encode(options.initial_prompt)
                if options.initial_prompt is not None
                else []
            ),
            without_timestamps=options.without_timestamps,
            hotwords=options.hotwords,
        )

        if options.max_new_tokens is not None:
            max_length = len(prompt) + options.max_new_tokens
        else:
            max_length = self.model.max_length

        if max_length > self.model.max_length:
            raise ValueError(
                f"The length of the prompt is {len(prompt)}, and the `max_new_tokens` "
                f"{max_length - len(prompt)}. Thus, the combined length of the prompt "
                f"and `max_new_tokens` is: {max_length}. This exceeds the "
                f"`max_length` of the Whisper model: {self.model.max_length}. "
                "You should either reduce the length of your prompt, or "
                "reduce the value of `max_new_tokens`, "
                f"so that their combined length is less that {self.model.max_length}."
            )

        encoder_output = self.model.encode(features)
        prompts = [prompt.copy() for _ in range(batch_size)]

        if options.multilingual:
            language_tokens = [
                tokenizer.tokenizer.token_to_id(segment_langs[0][0])
                for segment_langs in self.model.model.detect_language(encoder_output)
            ]
            language_token_index = prompt.index(tokenizer.language)

            for i, language_token in enumerate(language_tokens):
                prompts[i][language_token_index] = language_token

        futures: List[WhisperGenerationResultAsync] = self.model.model.generate(
            encoder_output,
            prompts,
            beam_size=options.beam_size,
            patience=options.patience,
            length_penalty=options.length_penalty,
            max_length=max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
            return_scores=True,
            return_no_speech_prob=True,
            sampling_temperature=options.temperatures[0],
            repetition_penalty=options.repetition_penalty,
            no_repeat_ngram_size=options.no_repeat_ngram_size,
            asynchronous=True
        )

        async def await_result(future):
            while not future.done():
                await asyncio.sleep(0.001)
            return future.result()

        results = await asyncio.gather(*[await_result(future) for future in futures])

        output = []
        for result in results:
            # return scores
            seq_len = len(result.sequences_ids[0])
            cum_logprob = result.scores[0] * (seq_len ** options.length_penalty)

            output.append(
                dict(
                    avg_logprob=cum_logprob / (seq_len + 1),
                    no_speech_prob=result.no_speech_prob,
                    tokens=result.sequences_ids[0],
                )
            )

        return encoder_output, output

    async def transcribe(
            self,
            audio: Union[str, BinaryIO, np.ndarray],
            language: Optional[str] = None,
            task: str = "transcribe",
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
            no_speech_threshold: Optional[float] = 0.6,
            condition_on_previous_text: bool = True,
            prompt_reset_on_temperature: float = 0.5,
            initial_prompt: Optional[Union[str, Iterable[int]]] = None,
            prefix: Optional[str] = None,
            suppress_blank: bool = True,
            suppress_tokens: Optional[List[int]] = [-1],
            without_timestamps: bool = True,
            max_initial_timestamp: float = 1.0,
            word_timestamps: bool = False,
            prepend_punctuations: str = "\"'“¿([{-",
            append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
            multilingual: bool = False,
            vad_filter: bool = True,
            vad_parameters: Optional[Union[dict, VadOptions]] = None,
            max_new_tokens: Optional[int] = None,
            chunk_length: Optional[int] = None,
            clip_timestamps: Optional[List[dict]] = None,
            hallucination_silence_threshold: Optional[float] = None,
            batch_size: int = 8,
            hotwords: Optional[str] = None,
            language_detection_threshold: Optional[float] = 0.5,
            language_detection_segments: int = 1,
    ) -> Tuple[AsyncGenerator[Segment, None], TranscriptionInfo]:
        """transcribe audio in chunks in batched fashion and return with language info.

        Arguments:
            audio: Path to the input file (or a file-like object), or the audio waveform.
            language: The language spoken in the audio. It should be a language code such
                as "en" or "fr". If not set, the language will be detected in the first 30 seconds
                of audio.
            task: Task to execute (transcribe or translate).
            log_progress: whether to show progress bar or not.
            beam_size: Beam size to use for decoding.
            best_of: Number of candidates when sampling with non-zero temperature.
            patience: Beam search patience factor.
            length_penalty: Exponential length penalty constant.
            repetition_penalty: Penalty applied to the score of previously generated tokens
                (set > 1 to penalize).
            no_repeat_ngram_size: Prevent repetitions of ngrams with this size (set 0 to disable).
            temperature: Temperature for sampling. If a list or tuple is passed,
                only the first value is used.
            initial_prompt: Optional text string or iterable of token ids to provide as a
                prompt for the each window.
            suppress_blank: Suppress blank outputs at the beginning of the sampling.
            suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
                of symbols as defined in `tokenizer.non_speech_tokens()`.
            without_timestamps: Only sample text tokens.
            word_timestamps: Extract word-level timestamps using the cross-attention pattern
                and dynamic time warping, and include the timestamps for each word in each segment.
                Set as False.
            prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
                with the next word
            append_punctuations: If word_timestamps is True, merge these punctuation symbols
                with the previous word
            multilingual: Perform language detection on every segment.
            vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
                without speech. This step is using the Silero VAD model
                https://github.com/snakers4/silero-vad.
            vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
                parameters and default values in the class `VadOptions`).
            max_new_tokens: Maximum number of new tokens to generate per-chunk. If not set,
                the maximum will be set by the default max_length.
            chunk_length: The length of audio segments. If it is not None, it will overwrite the
                default chunk_length of the FeatureExtractor.
            clip_timestamps: Optionally provide list of dictionaries each containing "start" and
                "end" keys that specify the start and end of the voiced region within
                `chunk_length` boundary. vad_filter will be ignored if clip_timestamps is used.
            batch_size: the maximum number of parallel requests to model for decoding.
            hotwords:
                Hotwords/hint phrases to the model. Has no effect if prefix is not None.
            language_detection_threshold: If the maximum probability of the language tokens is
                higher than this value, the language is detected.
            language_detection_segments: Number of segments to consider for the language detection.

        Unused Arguments
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
                such as repetition looping or timestamps going out of sync. Set as False
            prompt_reset_on_temperature: Resets prompt if temperature is above this value.
                Arg has effect only if condition_on_previous_text is True. Set at 0.5
            prefix: Optional text to provide as a prefix at the beginning of each window.
            max_initial_timestamp: The initial timestamp cannot be later than this, set at 0.0.
            hallucination_silence_threshold: Optional[float]
                When word_timestamps is True, skip silent periods longer than this threshold
                (in seconds) when a possible hallucination is detected. set as None.
        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """

        sampling_rate = self.model.feature_extractor.sampling_rate

        if multilingual and not self.model.model.is_multilingual:
            self.model.logger.warning(
                "The current model is English-only but the multilingual parameter is set to"
                "True; setting to False instead."
            )
            multilingual = False

        if not isinstance(audio, np.ndarray):
            #When transmitting np.ndarray works faster since no decoding is needed!
            audio = decode_audio(audio, sampling_rate=sampling_rate)
        duration = audio.shape[0] / sampling_rate

        self.model.logger.info(
            "Processing audio with duration %s", format_timestamp(duration)
        )

        chunk_length = chunk_length or self.model.feature_extractor.chunk_length
        # if no segment split is provided, use vad_model and generate segments
        if not clip_timestamps:
            if vad_filter:
                if vad_parameters is None:
                    vad_parameters = VadOptions(
                        max_speech_duration_s=chunk_length,
                        min_silence_duration_ms=160,
                    )
                elif isinstance(vad_parameters, dict):
                    if "max_speech_duration_s" in vad_parameters.keys():
                        vad_parameters.pop("max_speech_duration_s")

                    vad_parameters = VadOptions(
                        **vad_parameters, max_speech_duration_s=chunk_length
                    )

                active_segments = await asyncio.to_thread(
                    get_speech_timestamps, audio, vad_parameters, 16000
                )
                clip_timestamps = await asyncio.to_thread(
                    merge_segments, active_segments, vad_parameters, 16000
                )
                # -------------------------------------
            # run the audio if it is less than 30 sec even without clip_timestamps
            elif duration < chunk_length:
                clip_timestamps = [{"start": 0, "end": audio.shape[0]}]
            else:
                raise RuntimeError(
                    "No clip timestamps found. "
                    "Set 'vad_filter' to True or provide 'clip_timestamps'."
                )

        duration_after_vad = (
                sum((segment["end"] - segment["start"]) for segment in clip_timestamps)
                / sampling_rate
        )

        self.model.logger.info(
            "VAD filter removed %s of audio",
            format_timestamp(duration - duration_after_vad),
        )
        audio_chunks, chunks_metadata = await asyncio.to_thread(
            collect_chunks, audio, clip_timestamps, 16000
        )

        async def extract_features(chunk):
            feature_result = await asyncio.to_thread(self.model.feature_extractor, chunk)
            return feature_result[..., :-1]

        features = []
        if duration_after_vad:
            extract_tasks = [extract_features(chunk) for chunk in audio_chunks]
            features = await asyncio.gather(*extract_tasks)

        all_language_probs = None
        # detecting the language if not provided
        if language is None:
            if not self.model.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                (
                    language,
                    language_probability,
                    all_language_probs,
                ) = self.model.detect_language(
                    features=np.concatenate(
                        features
                        + [
                            np.full((self.model.model.n_mels, 1), -1.5, dtype="float32")
                        ],
                        axis=1,
                    ),  # add a dummy feature to account for empty audio
                    language_detection_segments=language_detection_segments,
                    language_detection_threshold=language_detection_threshold,
                )

                self.model.logger.info(
                    "Detected language '%s' with probability %.2f",
                    language,
                    language_probability,
                )
        else:
            if not self.model.model.is_multilingual and language != "en":
                self.model.logger.warning(
                    "The current model is English-only but the language parameter is set to '%s'; "
                    "using 'en' instead." % language
                )
                language = "en"

            language_probability = 1

        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task=task,
            language=language,
        )

        features = (
            np.stack([pad_or_trim(feature) for feature in features]) if features else []
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
            temperatures=(
                temperature[:1]
                if isinstance(temperature, (list, tuple))
                else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=(
                await asyncio.to_thread(get_suppressed_tokens, tokenizer, suppress_tokens)
                if suppress_tokens
                else suppress_tokens
            ),
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
            word_timestamps=word_timestamps,
            hallucination_silence_threshold=None,
            condition_on_previous_text=False,
            clip_timestamps=clip_timestamps,
            prompt_reset_on_temperature=0.5,
            multilingual=multilingual,
            without_timestamps=without_timestamps,
            max_initial_timestamp=0.0,
        )

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )

        segments = self._batched_segments_generator(
            features,
            tokenizer,
            chunks_metadata,
            batch_size,
            options,
            log_progress,
        )

        return segments, info

    async def _batched_segments_generator(
            self, features, tokenizer, chunks_metadata, batch_size, options, log_progress
    ):
        """
        Asynchronous generator for batch processing of transcription segments.

        Args:
            features: Extracted audio features.
            tokenizer: Tokenizer for decoding tokens.
            chunks_metadata: Audio chunks metadata.
            batch_size: Batch size for processing.
            options: Transcription options.
            log_progress: Flag to display progress.

        Yields:
            Segment: Transcription segment objects.
        """

        # Инициализируем асинхронный прогресс-бар
        pbar = atqdm(total=len(features), disable=not log_progress, position=0)
        seg_idx = 0

        try:
            # Process the data in packets of a certain size
            for i in range(0, len(features), batch_size):
                results = await self.forward(
                    features[i: i + batch_size],
                    tokenizer,
                    chunks_metadata[i: i + batch_size],
                    options,
                )

                for result in results:
                    for segment in result:
                        seg_idx += 1
                        yield Segment(
                            seek=segment["seek"],
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
                            temperature=options.temperatures[0],
                        )

                pbar.update(1)
        finally:
            pbar.close()
            self.last_speech_timestamp = 0.0
