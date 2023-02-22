import collections
import zlib

import ctranslate2
import numpy as np
import tokenizers

from faster_whisper.audio import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor


class Segment(collections.namedtuple("Segment", ("start", "end", "text"))):
    pass


class AudioInfo(
    collections.namedtuple("AudioInfo", ("language", "language_probability"))
):
    pass


class TranscriptionOptions(
    collections.namedtuple(
        "TranscriptionOptions",
        (
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
            "without_timestamps",
        ),
    )
):
    pass


class WhisperModel:
    def __init__(
        self,
        model_path,
        device="auto",
        device_index=0,
        compute_type="default",
        cpu_threads=0,
        num_workers=1,
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

        self.feature_extractor = FeatureExtractor()
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(
            "openai/whisper-tiny" + ("" if self.model.is_multilingual else ".en")
        )
        self.eot_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.timestamp_begin_id = self.tokenizer.token_to_id("<|notimestamps|>") + 1
        self.input_stride = 2
        self.time_precision = 0.02
        self.max_length = 448

    def transcribe(
        self,
        input_file,
        language=None,
        task="transcribe",
        beam_size=5,
        best_of=5,
        patience=1,
        length_penalty=1,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        initial_prompt=None,
        without_timestamps=False,
    ):
        """Transcribes an input file.

        Arguments:
          input_file: Path to the input file or a file-like object.
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
          without_timestamps: Only sample text tokens.

        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of AudioInfo
        """
        audio = decode_audio(
            input_file, sampling_rate=self.feature_extractor.sampling_rate
        )
        features = self.feature_extractor(audio)

        if language is None:
            if not self.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                segment = self.get_segment(features)
                input = self.get_input(segment)
                results = self.model.detect_language(input)
                language_token, language_probability = results[0][0]
                language = language_token[2:-2]
        else:
            if self.tokenizer.token_to_id("<|%s|>" % language) is None:
                raise ValueError("%s is not a valid language code" % language)
            language_probability = 1

        options = TranscriptionOptions(
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
            without_timestamps=without_timestamps,
        )

        segments = self.generate_segments(features, language, options)

        audio_info = AudioInfo(
            language=language,
            language_probability=language_probability,
        )

        return segments, audio_info

    def generate_segments(self, features, language, options):
        tokenized_segments = self.generate_tokenized_segments(
            features, language, options
        )

        for start, end, tokens in tokenized_segments:
            text = self.decode_text_tokens(tokens)
            if not text.strip():
                continue

            yield Segment(
                start=start,
                end=end,
                text=text,
            )

    def generate_tokenized_segments(self, features, language, options):
        num_frames = features.shape[-1]
        offset = 0
        all_tokens = []
        prompt_reset_since = 0

        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = self.tokenizer.encode(
                initial_prompt, add_special_tokens=False
            )
            all_tokens.extend(initial_prompt_tokens.ids)

        while offset < num_frames:
            time_offset = offset * self.feature_extractor.time_per_frame
            segment = self.get_segment(features, offset)
            segment_duration = segment.shape[-1] * self.feature_extractor.time_per_frame

            previous_tokens = all_tokens[prompt_reset_since:]
            prompt = self.get_prompt(
                language,
                previous_tokens,
                task=options.task,
                without_timestamps=options.without_timestamps,
            )

            result, avg_log_prob, temperature = self.generate_with_fallback(
                segment, prompt, options
            )

            if (
                result.no_speech_prob > options.no_speech_threshold
                and avg_log_prob < options.log_prob_threshold
            ):
                offset += segment.shape[-1]
                continue

            tokens = result.sequences_ids[0]

            consecutive_timestamps = [
                i
                for i in range(len(tokens))
                if i > 0
                and tokens[i] >= self.timestamp_begin_id
                and tokens[i - 1] >= self.timestamp_begin_id
            ]

            if len(consecutive_timestamps) > 0:
                last_slice = 0
                for i, current_slice in enumerate(consecutive_timestamps):
                    sliced_tokens = tokens[last_slice:current_slice]
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

                    last_in_window = i + 1 == len(consecutive_timestamps)

                    # Include the last timestamp so that all tokens are included in a segment.
                    if last_in_window:
                        sliced_tokens.append(tokens[current_slice])

                    yield start_time, end_time, sliced_tokens
                    last_slice = current_slice

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

                yield time_offset, time_offset + duration, tokens

                offset += segment.shape[-1]
                all_tokens.extend(tokens)

            if not options.condition_on_previous_text or temperature > 0.5:
                prompt_reset_since = len(all_tokens)

    def decode_text_tokens(self, tokens):
        text_tokens = [token for token in tokens if token < self.eot_id]
        return self.tokenizer.decode(text_tokens)

    def generate_with_fallback(self, segment, prompt, options):
        features = self.get_input(segment)
        result = None
        avg_log_prob = None
        final_temperature = None
        max_length = min(self.max_length, 2 * (self.max_length - len(prompt)))

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
                max_length=max_length,
                return_scores=True,
                return_no_speech_prob=True,
                **kwargs,
            )[0]

            # Recover the average log prob from the returned score.
            seq_len = len(result.sequences_ids[0])
            cum_log_prob = result.scores[0] * (seq_len**options.length_penalty)
            avg_log_prob = cum_log_prob / (seq_len + 1)

            tokens = result.sequences_ids[0]
            text = self.decode_text_tokens(tokens)
            compression_ratio = get_compression_ratio(text)

            if (
                compression_ratio <= options.compression_ratio_threshold
                and avg_log_prob >= options.log_prob_threshold
            ):
                break

        return result, avg_log_prob, final_temperature

    def get_prompt(
        self,
        language,
        previous_tokens,
        task="transcribe",
        without_timestamps=False,
    ):
        prompt = []

        if previous_tokens:
            prompt.append(self.tokenizer.token_to_id("<|startofprev|>"))
            prompt.extend(previous_tokens[-(self.max_length // 2 - 1) :])

        prompt.append(self.tokenizer.token_to_id("<|startoftranscript|>"))

        if self.model.is_multilingual:
            prompt.extend(
                [
                    self.tokenizer.token_to_id("<|%s|>" % language),
                    self.tokenizer.token_to_id("<|%s|>" % task),
                ]
            )

        if without_timestamps:
            prompt.append(self.tokenizer.token_to_id("<|notimestamps|>"))

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


def get_compression_ratio(text):
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))
