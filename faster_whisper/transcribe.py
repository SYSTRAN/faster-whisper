import collections
import os
import string
import zlib

from typing import BinaryIO, List, Optional, Tuple, Union

import ctranslate2
import dtw
import numpy as np
import tokenizers

# from scipy.signal import medfilt as median_filter
from scipy.ndimage import \
	median_filter  # faster owing to https://github.com/openai/whisper/commit/f0083e7eb20d032390e42f6f6039947fa8669c93


from faster_whisper.audio import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor

def dim(a):
    if not type(a) == list:
        if type(a) is not np.ndarray or (type(a) == np.ndarray and a.shape == np.empty(1)[0].shape):
            return []

    if len(a) == 0:
        return []

    return [len(a)] + dim(a[0])

class Segment(collections.namedtuple("Segment", ("offset", "start", "end", "text", "tokens", "attention_weights", "token_scores"))):
    pass

class AudioInfo(
    collections.namedtuple("AudioInfo", ("language", "language_probability"))
):
    pass


class TranscriptionOptions(
    collections.namedtuple(
        "TranscriptionOptions",
        (
            "language",
            "task",
            "beam_size",
            "best_of",
            "patience",
            "length_penalty",
            "log_prob_threshold",
            "no_speech_threshold",
            "compression_ratio_threshold",
            "condition_on_previous_text",
            "temperatures",
            "initial_prompt",
            "prefix",
            "suppress_blank",
            "suppress_tokens",
            "without_timestamps",
            "max_initial_timestamp",
            "lucid_threshold",
        ),
    )
):
    pass

class WordTimestampOptions(
    collections.namedtuple(
        "WordTimestampOptions",
        (
            "audio",
            "language",
            "segments",
            "remove_punctuation_from_words",
            "compute_word_confidence",
            "include_punctuation_in_confidence",
            "refine_whisper_precision",
            "refine_whisper_precision_nframes",
            "word_alignement_most_top_layers",
            "trust_whisper_timestamps",
            "min_word_duration",
            "alignment_heads",
            "transcription_options"
        ),
    )
):
    pass


class WhisperModel:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        device_index: int = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
    ):
        """Initializes the Whisper model.

        Args:
          model_path: Path to the converted model.
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
        """
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
            self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
        else:
            self.tokenizer = tokenizers.Tokenizer.from_pretrained(
                "openai/whisper-tiny" + ("" if self.model.is_multilingual else ".en")
            )

        self.feature_extractor = FeatureExtractor()
        tokenizer_filepath = os.path.join(os.path.dirname(__file__), 'tokenizer.json')
        self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_filepath)

        self.sot_id = self.tokenizer.token_to_id("<|startoftranscript|>")
        self.eot_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.timestamp_begin_id = self.tokenizer.token_to_id("<|notimestamps|>") + 1
        self.input_stride = 2
        self.time_precision = 0.02
        self.max_length = 448

        self.audio_samples_per_token = self.feature_extractor.hop_length * 2  # 320
        self.audio_time_per_token = self.audio_samples_per_token / self.feature_extractor.sampling_rate  # 0.02
        self.segment_duration = self.feature_extractor.nb_max_frames * self.feature_extractor.hop_length / self.feature_extractor.sampling_rate  # 30.0 (sec)
        self._punctuation = "".join(c for c in string.punctuation if c not in ["-", "'"]) + "。，！？：”、…"
    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task="transcribe",
        beam_size=5,
        best_of=5,
        patience=1,
        length_penalty=1,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        language_threshold: float = 0.6,
        language_detection_segments: int = 1,
        lucid_threshold=0.3,

        # Additional options for word alignment
        remove_punctuation_from_words=False,
        compute_word_confidence=True,
        include_punctuation_in_confidence=False,
        refine_whisper_precision=0.5,
        min_word_duration=0.04,
        word_alignement_most_top_layers=6,
        trust_whisper_timestamps=True,    
    ):
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
          temperature: Temperature for sampling. It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            `compression_ratio_threshold` or `logprob_threshold`.
          compression_ratio_threshold: If the gzip compression ratio is above this value,
            treat as failed.
          log_prob_threshold: If the average log probability over sampled tokens is
            below this value, treat as failed.
          no_speech_threshold: If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `logprob_threshold`,
            consider the segment as silent.
          condition_on_previous_text: If True, the previous output of the model is provided
            as a prompt for the next window; disabling may make the text inconsistent across
            windows, but the model becomes less prone to getting stuck in a failure loop,
            such as repetition looping or timestamps going out of sync.
          initial_prompt: Optional text to provide as a prompt for the first window.
          prefix: Optional text to provide as a prefix for the first window.
          suppress_blank: Suppress blank outputs at the beginning of the sampling.
          suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
            of symbols as defined in the model config.json file.
          without_timestamps: Only sample text tokens.
          max_initial_timestamp: The initial timestamp cannot be later than this.

          remove_punctuation_from_words: bool
            If False, words will be glued with the next punctuation mark (if any).
            If True, there will be no punctuation mark in the `words[:]["text"]` list.
            It only affects these strings; This has no influence on the computation of the word confidence, whatever the value of `include_punctuation_in_confidence` is.
          compute_word_confidence: bool
            Whether to compute word confidence.
            If True, a finer confidence for each segment will be computed as well.

          include_punctuation_in_confidence: bool
            Whether to include proba of punctuation in the computation of the (previous) word confidence.
          refine_whisper_precision: float
            How much can we refine Whisper segment positions, in seconds. Must be a multiple of 0.02.
          min_word_duration: float
            Minimum duration of a word, in seconds. If a word is shorter than this, timestamps will be adjusted.

        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of AudioInfo
        """
        # Check input options
        assert refine_whisper_precision >= 0 and refine_whisper_precision / self.audio_time_per_token == round(
            refine_whisper_precision / self.audio_time_per_token), f"refine_whisper_precision must be a positive multiple of {self.audio_time_per_token}"
        refine_whisper_precision_nframes = round(refine_whisper_precision / self.audio_time_per_token)
        assert min_word_duration >= 0, f"min_word_duration must be a positive number"
        assert word_alignement_most_top_layers is None or word_alignement_most_top_layers > 0, f"word_alignement_most_top_layers must be a strictly positive number"

        if not isinstance(audio, np.ndarray):
            audio = decode_audio(
                audio, sampling_rate=self.feature_extractor.sampling_rate
            )
        features = self.feature_extractor(audio)

        import time
        stt = time.time()

        if language is None:
            num_frames = features.shape[-1]
            if language_detection_segments is None or language_detection_segments < 1:
                language_detection_segments = 1
            offset = 0
            languages = []
            while offset < num_frames and offset < self.feature_extractor.nb_max_frames * language_detection_segments:
                segment = self.get_segment(features, offset)
                input = self.get_input(segment)
                results = self.model.detect_language(input)
                language_token, language_probability = results[0][0]
                if language_threshold is not None and language_probability > language_threshold:
                    language = language_token[2:-2]
                    break
                else:
                    languages.append(language_token[2:-2])
                    offset += segment.shape[-1]
            else:
                # If no language detected for all segments, the majority vote of the highest projected
                # languages for all segments is used to determine the language.
                language = max(set(languages), key=languages.count)
        else:
            if self.tokenizer.token_to_id("<|%s|>" % language) is None:
                raise ValueError("%s is not a valid language code" % language)
            language_probability = 1

        transcription_options = TranscriptionOptions(
            task=task,
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            temperatures=(
                temperature if isinstance(temperature, (list, tuple)) else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=suppress_tokens,
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            lucid_threshold=lucid_threshold,
        )

        segments = self.generate_segments(features, language, transcription_options)
        segments = list(segments)

        audio_info = AudioInfo(
            language=language,
            language_probability=language_probability,
        )

        # return segments, audio_info

        # get the timestamps for each word
        wordtimestamp_options = WordTimestampOptions(
            audio=audio,
            language=language,
            segments=segments,
            remove_punctuation_from_words=remove_punctuation_from_words,
            compute_word_confidence=compute_word_confidence,
            include_punctuation_in_confidence=include_punctuation_in_confidence,
            refine_whisper_precision=refine_whisper_precision,
            refine_whisper_precision_nframes=refine_whisper_precision_nframes,
            word_alignement_most_top_layers=word_alignement_most_top_layers,
            trust_whisper_timestamps=trust_whisper_timestamps,
            min_word_duration=min_word_duration,
            alignment_heads=None,
            transcription_options=transcription_options,
        )

        import time
        stt = time.time()
        transcription = self.get_attention_timestamps(wordtimestamp_options)
        print("attention_timestamps", time.time() - stt)

        return transcription, audio_info

    def generate_segments(self, features, options):
        tokenized_segments = self.generate_tokenized_segments(features, options)

        for offset, start, end, tokens, token_scores, attention in tokenized_segments:
            text = self.decode_text_tokens(tokens)
            if not text.strip():
                continue

            yield Segment(
                offset=offset,
                start=start,
                end=end,
                text=text,
                tokens=tokens,
                attention_weights=attention,
                token_scores=token_scores,
            )

    def generate_tokenized_segments(self, features, options):
        num_frames = features.shape[-1]
        offset = 0
        all_tokens = []
        prompt_reset_since = 0
        prompt = None

        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = self.encode_text(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)

        while offset < num_frames:
            time_offset = offset * self.feature_extractor.time_per_frame
            segment = self.get_segment(features, offset)
            segment_duration = segment.shape[-1] * self.feature_extractor.time_per_frame

            # Lucid Whisper
            if ((offset + self.feature_extractor.nb_max_frames) / num_frames < 1.0) or (
                    offset == 0):  # first chunk, ergo no context or next chunk will be fully within num_frames ergo should be fine
                previous_tokens = all_tokens[prompt_reset_since:]
                prompt = self.get_prompt(language, previous_tokens, task=options.task,
                                         without_timestamps=options.without_timestamps,prefix=options.prefix,)
            else:  # next chunk will not be fully within num_frames i.e. last chunk, calculate lucid_score
                lucid_score = (num_frames - offset) / self.feature_extractor.nb_max_frames
                if lucid_score < options.lucid_threshold and prompt is not None:  # Lucid Score below threshold, erasing context!
                    prompt = self.get_prompt(language, [], task=options.task,
                                             without_timestamps=options.without_timestamps,prefix=options.prefix,)
                else:  # Lucid Score above threshold, keeping context!
                    previous_tokens = all_tokens[prompt_reset_since:]
                    prompt = self.get_prompt(language, previous_tokens, task=options.task,
                                             without_timestamps=options.without_timestamps, prefix=options.prefix,)

            import time
            stt = time.time()
            result, avg_log_prob, temperature = self.generate_with_fallback(segment, prompt, options)
            print("generate_time", time.time() - stt)
            if (
                result.no_speech_prob > options.no_speech_threshold
                and avg_log_prob < options.log_prob_threshold
            ):
                offset += segment.shape[-1]
                continue

            tokens = result.sequences_ids[0]
            token_scores = result.token_scores

            attention = np.expand_dims(np.expand_dims(result.attention, axis=0), axis=0)

            consecutive_timestamps = [
                i
                for i in range(len(tokens))
                if i > 0
                and tokens[i] >= self.timestamp_begin_id
                and tokens[i - 1] >= self.timestamp_begin_id
            ]

            if len(consecutive_timestamps) > 0:
                ended_with_single_timestamp = (
                    len(tokens) >= 2
                    and tokens[-2] < self.timestamp_begin_id
                    and tokens[-1] >= self.timestamp_begin_id
                )

                if ended_with_single_timestamp:
                    consecutive_timestamps.append(len(tokens))

                last_slice = 0
                for i, current_slice in enumerate(consecutive_timestamps):
                    sliced_tokens = tokens[last_slice:current_slice]
                    sliced_token_scores = token_scores[last_slice:current_slice]
                    sliced_attention = attention[:, :, last_slice:current_slice, :]
                    start_timestamp_position = (
                        sliced_tokens[0] - self.timestamp_begin_id
                    )
                    end_timestamp_position = sliced_tokens[-1] - self.timestamp_begin_id
                    start_time = (
                        time_offset + start_timestamp_position * self.time_precision
                    )
                    end_time = (
                        time_offset + end_timestamp_position * self.time_precision
                    )

                    yield offset, start_time, end_time, sliced_tokens, sliced_token_scores, sliced_attention
                    last_slice = current_slice

                if ended_with_single_timestamp:
                    # single timestamp at the end means no speech after the last timestamp.
                    offset += segment.shape[-1]
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_position = (
                        tokens[last_slice - 1] - self.timestamp_begin_id
                    )
                    offset += last_timestamp_position * self.input_stride
                all_tokens.extend(tokens[: last_slice + 1])

            else:
                duration = segment_duration
                timestamps = [
                    token for token in tokens if token >= self.timestamp_begin_id
                ]
                if len(timestamps) > 0 and timestamps[-1] != self.timestamp_begin_id:
                    last_timestamp_position = timestamps[-1] - self.timestamp_begin_id
                    duration = last_timestamp_position * self.time_precision
                yield offset, time_offset, time_offset + duration, tokens, token_scores, attention

                offset += segment.shape[-1]
                all_tokens.extend(tokens)

            if not options.condition_on_previous_text or temperature > 0.5:
                prompt_reset_since = len(all_tokens)

    def encode_text(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode_text_tokens(self, tokens):
        text_tokens = [token for token in tokens if token < self.eot_id]
        return self.tokenizer.decode(text_tokens)

    def decode_text_tokens_with_timestamps(self, tokens):
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        outputs = [[]]
        for token in tokens:
            if token >= self.timestamp_begin_id:
                timestamp = f"<|{(token - self.timestamp_begin_id) * 0.02:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [s if isinstance(s, str) else self.decode_text_tokens(s) for s in outputs]
        return "".join(outputs)

    def generate_with_fallback(self, segment, prompt, options):
        features = self.get_input(segment)
        result = None
        avg_log_prob = None
        final_temperature = None
        max_length = min(self.max_length, 2 * (self.max_length - len(prompt)))

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
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

            final_temperature = temperature
            result = self.model.generate(
                features,
                [prompt],
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                return_scores=True,
                return_attention=True,
                return_no_speech_prob=True,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                max_initial_timestamp_index=max_initial_timestamp_index,
                **kwargs,
            )[0]

            tokens = result.sequences_ids[0]

            # Recover the average log prob from the returned score.
            seq_len = len(tokens)
            cum_log_prob = result.scores[0] * (seq_len**options.length_penalty)
            avg_log_prob = cum_log_prob / (seq_len + 1)

            text = self.decode_text_tokens(tokens).strip()
            compression_ratio = get_compression_ratio(text)

            needs_fallback = False

            if (
                options.compression_ratio_threshold is not None
                and compression_ratio > options.compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive

            if (
                options.log_prob_threshold is not None
                and avg_log_prob < options.log_prob_threshold
            ):
                needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return result, avg_log_prob, final_temperature

    def get_prompt(
        self,
        language,
        previous_tokens,
        task="transcribe",
        without_timestamps=False,
        prefix=None,
    ):
        prompt = []

        if previous_tokens:
            prompt.append(self.tokenizer.token_to_id("<|startofprev|>"))
            prompt.extend(previous_tokens[-(self.max_length // 2 - 1) :])

        prompt.append(self.tokenizer.token_to_id("<|startoftranscript|>"))

        prompt.extend(
            [
                self.tokenizer.token_to_id("<|%s|>" % language),
                self.tokenizer.token_to_id("<|%s|>" % task),
            ]
        )

        if without_timestamps:
            prompt.append(self.tokenizer.token_to_id("<|notimestamps|>"))

        if prefix:
            prefix_tokens = self.encode_text(" " + prefix.strip())
            if len(prefix_tokens) >= self.max_length // 2:
                prefix_tokens = prefix_tokens[: self.max_length // 2 - 1]
            prompt.extend(prefix_tokens)

        return prompt

    def get_segment(self, features, offset=0):
        if offset > 0:
            features = features[:, offset:]

        num_frames = features.shape[-1]
        required_num_frames = self.feature_extractor.nb_max_frames

        if num_frames > required_num_frames:
            features = features[:, :required_num_frames]
        elif num_frames < required_num_frames:
            pad_widths = [(0, 0), (0, required_num_frames - num_frames)]
            features = np.pad(features, pad_widths)

        features = np.ascontiguousarray(features)
        return features

    def get_input(self, segment):
        segment = np.expand_dims(segment, 0)
        segment = ctranslate2.StorageView.from_array(segment)
        return segment

    def get_attention_timestamps(self, options):
        refine_whisper_precision_sec = options.refine_whisper_precision_nframes * self.audio_time_per_token

        # When not relying on Whisper timestamps
        current_tokens = []
        token_to_idx_segment = []

        words = []
        previous_end = 0
        audio_duration = options.audio.shape[-1] / self.feature_extractor.sampling_rate
        use_space = self.should_use_space(options.language)

        word_alignement_most_top_layers = float(
            "inf") if options.word_alignement_most_top_layers is None else options.word_alignement_most_top_layers

        transcription = {
            "text": "",
            "segments": [],
        }

        for i_segment, segment in enumerate(options.segments):

            # create dictionary of relevant segment features for the output
            new_segment = {
                "id": i_segment,
                "offset": segment.offset,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }

            start = end = tokens = None
            if options.trust_whisper_timestamps:
                start = segment.start
                end = segment.end
                if end < start:
                    # Whisper is wrong on the prediction of segment end
                    end = min(audio_duration, start + self.segment_duration)

                start_margin_min = start - refine_whisper_precision_sec
                start_margin_max = start + refine_whisper_precision_sec
                if start >= audio_duration - options.min_word_duration or (
                        previous_end >= start_margin_min and previous_end <= start_margin_max):
                    # Make start as accurate as possible (as the decoding will start with timestamp <|0|>)
                    start = previous_end
                else:
                    # Fallback
                    start = start_margin_min

                if start > audio_duration - options.min_word_duration:
                    continue

                end_margin_min = end - refine_whisper_precision_sec
                end_margin_max = end + refine_whisper_precision_sec
                if i_segment < len(options.segments) - 1:
                    # Try to enforce:
                    #   end + min_word_duration <= next start + refine_whisper_precision_sec
                    end_margin_max2 = options.segments[
                                          i_segment + 1].start + refine_whisper_precision_sec - options.min_word_duration
                    if end_margin_max2 >= end_margin_min:
                        end_margin_max = min(end_margin_max2, end_margin_max)
                end = min(audio_duration, end_margin_max)

                if end < start + options.min_word_duration:
                    end = min(audio_duration, start + options.min_word_duration)
                    if end <= start:
                        continue

                tokens = segment.tokens

            else:
                seek = segment.offset
                new_tokens = segment.tokens
                # Add timestamps that will be needed after
                if new_tokens[0] < self.timestamp_begin_id:
                    relative_start = segment.start - (seek * self.feature_extractor.hop_length / self.feature_extractor.sampling_rate)
                    start_token = round(
                        relative_start * self.feature_extractor.sampling_rate / self.audio_samples_per_token) + self.timestamp_begin_id
                    new_tokens = [start_token] + new_tokens
                if new_tokens[-1] < self.timestamp_begin_id:
                    relative_end = segment.end - (seek * self.feature_extractor.hop_length / self.feature_extractor.sampling_rate)
                    end_token = round(relative_end * self.feature_extractor.sampling_rate / self.audio_samples_per_token) + self.timestamp_begin_id
                    new_tokens = new_tokens + [end_token]

                current_tokens.extend(new_tokens)
                token_to_idx_segment.extend([i_segment] * len(new_tokens))

                next_seek = options.segments[i_segment + 1].offset if i_segment < len(options.segments) - 1 else None
                if seek != next_seek:
                    start = float(seek * self.feature_extractor.hop_length / self.feature_extractor.sampling_rate)
                    assert start < audio_duration, f"Got start {start} which is outside of audio duration {audio_duration}"
                    end = min(start + self.segment_duration, audio_duration)
                    tokens = current_tokens

            if start is None:
                continue

            start_sample = min(round(start * self.feature_extractor.sampling_rate), options.audio.shape[-1])
            end_sample = min(round(end * self.feature_extractor.sampling_rate), options.audio.shape[-1])

            sub_audio = self.audio_minimum_padding(options.audio[start_sample:end_sample])

            mfcc = self.feature_extractor(sub_audio)
            mfcc = self.feature_extractor.pad_or_trim(mfcc, self.feature_extractor.nb_max_frames)
            mfcc = np.expand_dims(mfcc, axis=0)

            # get the log prob for each token
            logprobs = np.array(segment.token_scores)

            # get the attention weights
            attention_weights = segment.attention_weights

            if options.word_alignement_most_top_layers:
                attention_weights = attention_weights[-options.word_alignement_most_top_layers:, :, :, :]

            ws = self.perform_word_alignment(
                tokens,
                attention_weights,
                use_space=use_space,
                remove_punctuation_from_words=options.remove_punctuation_from_words,
                refine_whisper_precision_nframes=options.refine_whisper_precision_nframes,
                mfcc=mfcc,
            )

            segment_logprobs = []
            i_token = 1
            for word in ws:
                offset = segment.offset * self.feature_extractor.time_per_frame
                word["start"] += offset
                word["end"] += offset
                word.update({"idx_segment": i_segment})
                if options.trust_whisper_timestamps:
                    word.update({"idx_segment": i_segment})
                else:
                    assert i_token < len(tokens)
                    assert word["tokens_indices"][0] == tokens[i_token]
                    word.update({"idx_segment": token_to_idx_segment[i_token]})
                    i_token += len(word["tokens"])
                    while i_token < len(tokens) and tokens[i_token] >= self.timestamp_begin_id:
                        i_token += 1
                if options.compute_word_confidence:
                    tok = word["tokens"]
                    tok_indices = word["tokens_indices"]
                    i_end = i_token + len(tok)
                    if options.include_punctuation_in_confidence:
                        while len(tok) > 1 and tok[-1][
                            -1] in self._punctuation:  # Note: look at the last character of token, to take into account "...", "!!", etc.
                            tok = tok[:-1]
                            tok_indices = tok_indices[:-1]
                    word_logprobs = [logprobs[step] for step in
                                     range(i_token, i_token + len(tok_indices))]
                    i_token = i_end
                    word.update({"confidence": round_confidence(np.exp(np.mean(word_logprobs)))})
                    segment_logprobs.append(word_logprobs)

                words.append(word)

            if len(segment_logprobs):
                segment_logprobs = np.concatenate(segment_logprobs)
                new_segment.update({"confidence": round_confidence((np.exp(np.mean(segment_logprobs))))})

            if len(ws):
                previous_end = ws[-1]["end"]

            if not options.trust_whisper_timestamps:
                current_tokens = []
                token_to_idx_segment = []

            # add the new segment to the output of segments
            transcription["segments"].append(new_segment)
            transcription["text"] += segment.text

        # Refine word positions
        self.ensure_increasing_positions(words, min_duration=options.min_word_duration if options.trust_whisper_timestamps else 0)

        word_segments = transcription["segments"]
        for word in words:
            word.pop("tokens")
            word.pop("tokens_indices")
            idx_segment = word.pop("idx_segment")
            segment = word_segments[idx_segment]
            if "words" in segment:
                segment["words"].append(word)
            else:
                segment["words"] = [word]
                if options.refine_whisper_precision:
                    segment["start"] = word["start"]
            if options.refine_whisper_precision:
                segment["end"] = word["end"]

        return transcription

    def audio_minimum_padding(self, audio):
        if audio.shape[-1] <= 200:
            return self.feature_extractor.pad_or_trim(audio, 201)
        return audio

    def find_start_padding(self, mfcc):
        """ Return start of padding given the mfcc, or None if there is no padding """
        last_mfcc = mfcc[0, :, -1]
        if np.min(last_mfcc) == np.max(last_mfcc) == 0:
            candidate_index = mfcc.shape[-1] - 2
            while candidate_index > 0:
                candidate = mfcc[0, :, candidate_index]
                if not np.array_equal(candidate, last_mfcc):
                    return candidate_index + 1
                candidate_index -= 1
            return 0  # WTF!?

    def should_use_space(self, language):
        return language not in ["zh", "ja", "th", "lo", "my"]

    def perform_word_alignment(
            self,
            tokens,
            attention_weights,
            use_space=True,
            mfcc=None,
            refine_whisper_precision_nframes=0,
            remove_punctuation_from_words=False,
            include_punctuation_in_timing=False,  # Was True before 1.9
            unfinished_decoding=False,
            alignment_heads=None,
            medfilt_width=9,
            qk_scale=1.0,
    ):
        """
        Perform word alignment on the given tokens and attention weights.
        Returns a list of (word, start_time, end_time) tuples.
        tokens: list of tokens (integers)
        attention_weights: list of attention weights (torch tensors)
        tokenizer: tokenizer used to tokenize the text
        use_space: whether to use spaces to split the tokens into words (should be true for all languages except Japanese, Chinese, ...)
        mfcc: MFCC features (used to identify padded region, and for plotting)
        refine_whisper_precision_nframes: precision time
        remove_punctuation_from_words: whether to remove punctuation from words
        include_punctuation_in_timing: whether to include punctuation in the timing of (previous) words
        unfinished_decoding: whether the decoding is unfinished (e.g. because the model is stuck)
        alignment_heads: list of attention heads to use for alignment
        medfilt_width: width of the median filter used to smooth the attention weights
        qk_scale: scale factor applied to the attention weights
        """

        assert len(
            tokens) > 1, f"Got unexpected sequence of tokens of length {len(tokens)} {self.decode_text_tokens_with_timestamps(tokens)}"
        start_token = tokens[0] - self.timestamp_begin_id
        end_token = tokens[-1] - self.timestamp_begin_id

        # Check start / end tokens
        if start_token < 0:
            raise RuntimeError(f"Missing start token in: {self.self.decode_text_tokens_with_timestamps(tokens)}")
        if len(tokens) == 1 or end_token < 0:
            end_token = self.feature_extractor.nb_max_frames // 2
        if end_token == start_token and refine_whisper_precision_nframes == 0:
            return []

        # Put some margin around the segment
        if refine_whisper_precision_nframes > 0:
            start_token = max(start_token - refine_whisper_precision_nframes, 0)
            end_token = min(end_token + refine_whisper_precision_nframes, self.feature_extractor.nb_max_frames // 2)

        if end_token <= start_token:
            raise RuntimeError(
                f"Got segment with null or negative duration {self.decode_text_tokens_with_timestamps(tokens)}: {start_token} {end_token}")

        start_time = start_token * self.audio_time_per_token
        end_time = end_token * self.audio_time_per_token

        split_tokens = self.split_tokens_on_spaces if use_space else self.split_tokens_on_unicode
        words, word_tokens, word_tokens_indices = split_tokens(tokens,
                                                               remove_punctuation_from_words=remove_punctuation_from_words)

        # If the last token is a punctuation that comes after a word
        # group this final punctuation with the final timestamp
        # This is to avoid assigning the final punctuation to a big silence or a noise/music background coming after
        num_punctuations_per_tokens = [
            0 if len(w) == 1 or w[-1] not in self._punctuation else 1
            for w in word_tokens
        ]
        if include_punctuation_in_timing:
            num_punctuations_per_tokens[:-2] = [0] * (len(num_punctuations_per_tokens) - 2)

        for i, w in enumerate(attention_weights):
            assert w.shape[-2] == len(
                tokens), f"Attention weights have wrong shape: {w.shape[-2]} (expected {len(tokens)})."
        weights = attention_weights  # layers * heads * tokens * frames

        num_tokens = weights.shape[-2]
        num_frames = end_token - start_token
        if num_tokens > num_frames:
            return self.perform_word_alignment(
                tokens[:num_frames - 1] + [tokens[-1]],
                np.concatenate((weights[:, :, :num_frames - 1, :], weights[:, :, -1:, :]), dim=-2),
                use_space=use_space,
                refine_whisper_precision_nframes=refine_whisper_precision_nframes,
                medfilt_width=medfilt_width,
                qk_scale=qk_scale,
                alignment_heads=alignment_heads,
                mfcc=mfcc,
                remove_punctuation_from_words=remove_punctuation_from_words,
                unfinished_decoding=True,
            )

        assert end_token <= weights.shape[-1]
        assert len(tokens) == num_tokens

        weights = weights[:, :, :, start_token: end_token]

        if alignment_heads is None:
            weights = weights.reshape(-1, *weights.shape[-2:])
        else:
            weights = np.stack([weights[l][h] for l, h in alignment_heads.indices().T])

        weights = median_filter(weights, (1, 1, medfilt_width))
        weights = softmax(np.array(weights * qk_scale), -1)
        # std = np.std(weights, axis=-2, keepdims=True, ddof=0)
        # mean = np.mean(weights, axis=-2, keepdims=True)
        # weights = (weights - mean)/std
        weights = weights.mean(axis=0)  # average over layers and heads
        weights = weights / np.linalg.norm(weights, axis=-2, keepdims=True)
        weights = -weights.astype(np.float64)
        worse_weight = 0

        # Get the limit of audio duration
        max_duration = None
        if mfcc is not None:
            max_duration = self.find_start_padding(mfcc)
            if max_duration is not None:
                max_duration = max_duration // 2

        # Enforce the max duration
        if max_duration:
            if start_token >= max_duration:
                pass
            else:
                weights[:-1, max_duration:] = worse_weight

        # Encourage to start early
        weights[0, 0] = weights.min()
        weights[0, refine_whisper_precision_nframes * 2:] = worse_weight

        # Similar as "symmetric1" but without the possibility to have the same timestamp for two tokens
        step_pattern = dtw.stepPattern.StepPattern(dtw.stepPattern._c(
            1, 1, 1, -1,
            1, 0, 0, 1,
            2, 0, 1, -1,
            2, 0, 0, 1,
        ))
        alignment = dtw.dtw(weights, step_pattern=step_pattern)

        plot = False
        if plot:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker

            mpl.use('TkAgg')  # !IMPORTANT

            if mfcc is None:
                plt.figure(figsize=(16, 9), frameon=False)
            else:
                plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [3, 1]})
                plt.subplot(2, 1, 1, frameon=False)

            plt.imshow(-weights, aspect="auto")
            plt.plot(alignment.index2s, alignment.index1s, color="red")

            xticks = np.arange(0, weights.shape[1], 1 / self.audio_time_per_token)
            xticklabels = [round_timestamp(x) for x in xticks * self.audio_time_per_token + start_time]

            ylims = plt.gca().get_ylim()

            ax = plt.gca()
            ax.tick_params('both', length=0, width=0, which='minor', pad=6)

            ax.yaxis.set_ticks_position("left")
            ax.yaxis.set_label_position("left")
            ax.invert_yaxis()
            ax.set_ylim(ylims)

            major_ticks = [-0.5]
            minor_ticks = []
            current_y = 0

            for word, word_token in zip(words, word_tokens):
                minor_ticks.append(current_y + len(word_token) / 2 - 0.5)
                current_y += len(word_token)
                major_ticks.append(current_y - 0.5)

            words_with_subwords = ["|".join(s).strip() for (w, s) in zip(words, word_tokens)]

            ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.yaxis.set_minor_formatter(
                ticker.FixedFormatter(words_with_subwords))
            ax.set_yticks(major_ticks)
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            for y in major_ticks:
                plt.axhline(y, color="black", linestyle="dashed")

            plt.ylabel("Words")

            if mfcc is not None:
                plt.xticks(xticks)
                plt.setp(plt.gca().get_xticklabels(), visible=False)

                xticks *= 2

                plt.subplot(2, 1, 2, frameon=False)
                plt.imshow(mfcc[0, :, start_token * 2: end_token * 2], aspect="auto")
                plt.yticks([])
                plt.ylabel("MFCC")

            plt.xticks(xticks, xticklabels)
            plt.xlabel("Time (s)")

        jumps = np.diff(alignment.index1s)
        jumps = np.pad(jumps, (1, 0), constant_values=1)
        jumps = jumps.astype(bool)
        jumps = alignment.index2s[jumps]
        jump_times = jumps * self.audio_time_per_token
        jump_times = np.pad(jump_times, (0, 1),
                            constant_values=end_time - start_time)

        # display the word-level timestamps in a table
        word_boundaries = np.cumsum([len(t) for t in word_tokens])
        word_boundaries = np.pad(word_boundaries, (1, 0))
        begin_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:] - num_punctuations_per_tokens]

        # Ignore start / end tokens
        if not refine_whisper_precision_nframes:
            begin_times[1] = begin_times[0]
        if not refine_whisper_precision_nframes:
            end_times[-2] = end_times[-1]
        if unfinished_decoding:
            words = words[1:]
            word_tokens = word_tokens[1:]
            word_tokens_indices = word_tokens_indices[1:]
            begin_times = begin_times[1:]
            end_times = end_times[1:]
        else:
            words = words[1:-1]
            word_tokens = word_tokens[1:-1]
            word_tokens_indices = word_tokens_indices[1:-1]
            begin_times = begin_times[1:-1]
            end_times = end_times[1:-1]

        if plot:
            ymin = 1

            if mfcc is not None:
                for i, (w, begin, end) in enumerate(zip(words, begin_times, end_times)):
                    plt.text(begin * 2 / self.audio_time_per_token, mfcc.shape[-2] * 1.05, w, ha="left", va="bottom",
                             color="red")
                    for x in [begin, end, ]:
                        plt.axvline(x * 2 / self.audio_time_per_token,
                                    color="red", linestyle="dotted")

                plt.subplot(2, 1, 1)

            for i, (w, ws, begin, end) in enumerate(zip(words, word_tokens, begin_times, end_times)):
                ymax = ymin + len(ws)
                if mfcc is None:
                    plt.text(begin / self.audio_time_per_token, num_tokens - 0.5, w, ha="left", va="top", color="red")
                for x in [begin, end, ]:
                    plt.axvline(x / self.audio_time_per_token, color="red", linestyle="dotted",
                                ymin=1 - ymin / num_tokens,
                                ymax=0,  # 1-ymax/num_tokens,
                                )
                ymin = ymax

            plt.show()

        return [
            dict(
                text=word,
                start=round_timestamp(begin + start_time),
                end=round_timestamp(end + start_time),
                tokens=tokens,
                tokens_indices=tokens_indices,
            )
            for word, begin, end, tokens, tokens_indices in
            zip(words, begin_times, end_times, word_tokens, word_tokens_indices)
            if not word.startswith("<|")
        ]

    def split_tokens_on_unicode(self, tokens, remove_punctuation_from_words=False,
                                isolate_punctuations=False):
        words = []
        word_tokens = []
        word_tokens_indices = []
        current_tokens = []

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_text_tokens_with_timestamps(current_tokens)
            if "\ufffd" not in decoded:
                empty_tokens = [""] * (len(current_tokens) - 1)
                punctuation = not isolate_punctuations and (
                            decoded.strip() and decoded.strip() in self._punctuation)
                previous_special = len(word_tokens_indices) > 0 and (word_tokens_indices[-1][-1] >= self.eot_id)
                if punctuation and not previous_special:
                    if len(words) == 0:
                        words = [""]
                        word_tokens = [[]]
                    if not remove_punctuation_from_words:
                        words[-1] += decoded
                    word_tokens[-1].extend(empty_tokens + [decoded])
                    word_tokens_indices[-1].extend(current_tokens)
                else:
                    words.append(decoded)
                    word_tokens.append(empty_tokens + [decoded])
                    word_tokens_indices.append(current_tokens)
                current_tokens = []

        return words, word_tokens, word_tokens_indices

    def split_tokens_on_spaces(self, tokens, remove_punctuation_from_words=False):
        subwords, subword_tokens_list, subword_tokens_indices_list = self.split_tokens_on_unicode(tokens,
                                                                                                  remove_punctuation_from_words=remove_punctuation_from_words)
        words = []
        word_tokens = []
        word_tokens_indices = []

        for i, (subword, subword_tokens, subword_tokens_indices) in enumerate(
                zip(subwords, subword_tokens_list, subword_tokens_indices_list)):
            special = (subword_tokens_indices[0] >= self.eot_id)
            previous_special = (i > 0) and (subword_tokens_indices_list[i - 1][0] >= self.eot_id)
            with_space = subword.startswith(" ")
            punctuation = (subword.strip() and subword.strip()) in self._punctuation
            if special or (with_space and not punctuation) or previous_special:
                words.append(subword.strip())
                word_tokens.append(subword_tokens)
                word_tokens_indices.append(subword_tokens_indices)
            else:
                words[-1] = words[-1] + subword.strip()
                word_tokens[-1].extend(subword_tokens)
                word_tokens_indices[-1].extend(subword_tokens_indices)

        return words, word_tokens, word_tokens_indices

    def ensure_increasing_positions(self, segments, min_duration=0):
        """
        Ensure that "start" and "end" come in increasing order
        """
        has_modified_backward = False
        previous_end = 0
        for i, seg in enumerate(segments):
            if seg["start"] < previous_end:
                assert i > 0
                new_start = round_timestamp((previous_end + seg["start"]) / 2)
                if new_start < segments[i - 1]["start"] + min_duration:
                    new_start = previous_end
                else:
                    segments[i - 1]["end"] = new_start
                    has_modified_backward = True
                seg["start"] = new_start
            if seg["end"] <= seg["start"] + min_duration:
                seg["end"] = seg["start"] + min_duration
            previous_end = seg["end"]
        if has_modified_backward:
            return self.ensure_increasing_positions(segments, min_duration)

        previous_end = 0
        for seg in segments:
            seg["start"] = round_timestamp(seg["start"])
            seg["end"] = round_timestamp(seg["end"])
            assert seg[
                       "start"] >= previous_end, f"Got segment {seg} coming before the previous finishes ({previous_end})"
            assert seg["end"] > seg["start"], f"Got segment {seg} with end <= start"
            previous_end = seg["end"]

        return segments

def round_confidence(x):
    return round(x, 3)

def round_timestamp(x):
    return round(x, 2)

def get_compression_ratio(text):
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))

def softmax(x, axis=-1):
    # Subtracting the maximum value for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
