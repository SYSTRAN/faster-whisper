import itertools
import json
import logging
import os
import zlib
import random
from collections import defaultdict, Counter

from inspect import signature
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union, TypedDict

import ctranslate2
import numpy as np
import tokenizers

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

import torch
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator


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

#Added additional parameters for multilingual videos and fixes below
class TranscriptionOptions(NamedTuple):
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
    log_prob_low_threshold: Optional[float] #New parameter
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
    multilingual: bool #New parameter
    output_language: str #New parameter


class TranscriptionInfo(NamedTuple):
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionOptions
    vad_options: VadOptions


class SingleSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech.

    start (float): Start time in seconds.
    end (float): End time in seconds.
    text (str): transcription of the segment.
    """

    start: float
    end: float
    text: str

class StreamSegment(NamedTuple):
    """
    A single segment (up to multiple sentences) of a speech.
    
    start (float): Start time in seconds.
    end (float): End time in seconds.
    text (str): transcription of the segment.
    """
    start: float
    end: float
    text: str


class TranscriptionResult(TypedDict):
    """
    A list of segments and word segments of a speech.
    """
    segments: List[SingleSegment]
    language: str


class BatchedInferencePipeline(Pipeline):

    """
    Huggingface Pipeline wrapper for WhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
            self,
            model,
            options : NamedTuple,
            tokenizer=None,
            device: Union[int, str, "torch.device"] = -1,
            framework = "pt",
            language : Optional[str] = None,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        features = torch.tensor(self.model.feature_extractor(audio, padding=True)[:,:self.model.feature_extractor.nb_max_frames])
        return {'inputs': features}

    def _forward(self, model_inputs):

        outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        return {'text': outputs}

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
            
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator
    
    def get_language_and_tokenizer(self, audio, task=None, language=None):

        language_probability = 1.0
        if self.tokenizer is None:
            if not language:
                language, language_probability = self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = Tokenizer(self.model.hf_tokenizer,
                                                                self.model.model.is_multilingual, task=task,
                                                                language=language)
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = Tokenizer(self.model.hf_tokenizer,
                                                                    self.model.model.is_multilingual, task=task,
                                                                    language=language)
        
        return language, language_probability, task
    
    def audio_split(self, audio, segments, sampling_rate):
        for seg in segments:
            f1 = int(seg['start'] * sampling_rate)
            f2 = int(seg['end'] * sampling_rate)
            yield {'inputs': audio[f1:f2]}

    def transcribe_stream(
        self, audio: Union[str, np.ndarray], vad_segments, batch_size=None, num_workers=0, language=None, task=None, log_progress = False, combined_progress=False
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = decode_audio(audio)

        language, language_probability, task = self.get_language_and_tokenizer(audio, task, language)
 
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        sampling_rate = self.model.feature_extractor.sampling_rate
        for idx, out in enumerate(self.__call__(self.audio_split(audio, vad_segments, sampling_rate), batch_size=batch_size, num_workers=num_workers)):
            if log_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                self.model.logger.info(f"Progress: {percent_complete:.2f}%...")

            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]

            yield StreamSegment(
                    text=text,
                    start=round(vad_segments[idx]['start'], 3),
                    end=round(vad_segments[idx]['end'], 3)
            ), {'language': language, 'language_probability':language_probability}

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

    def transcribe(
        self, audio: Union[str, np.ndarray], vad_segments, batch_size=None, num_workers=0, language=None, task=None, log_progress = False, combined_progress=False
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = decode_audio(audio)

        language, language_probability, task = self.get_language_and_tokenizer(audio, task, language)

        segments: List[SingleSegment] = [] 

        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        sampling_rate = self.model.feature_extractor.sampling_rate
        for idx, out in enumerate(self.__call__(self.data(audio, vad_segments, sampling_rate), batch_size=batch_size, num_workers=num_workers)):
            if log_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                self.model.logger.info(f"Progress: {percent_complete:.2f}%...")

            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]

            segments.append({
                'text':text,
                'start': round(vad_segments[idx]['start'], 3),
                'end': round(vad_segments[idx]['end'], 3)
            })

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        return {"segments": segments, "info": {'language': language, 'language_probability':language_probability}} 
    
    def detect_language(self, audio: np.ndarray):
        
        segment = torch.tensor(self.model.feature_extractor(audio, padding=True)[:,:self.model.feature_extractor.nb_max_frames])
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        self.model.logger.info(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio...")
        return language, language_probability
    
    def detect_language_multi_segment(self, audio: Union[str, BinaryIO, np.ndarray], params: dict):
        return self.model.detect_language_multi_segment(audio, params)


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
    ):
        """Initializes the Whisper model.

        Args:
          model_size_or_path: Size of the model to use (tiny, tiny.en, base, base.en,
            small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large), a path to a
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

        #set the random seed to make sure consistency across runs
        ctranslate2.set_random_seed(42)
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

        self.feat_kwargs = self._get_feature_kwargs(model_path)
        self.feature_extractor = FeatureExtractor(**self.feat_kwargs)
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

    def _get_feature_kwargs(self, model_path) -> dict:
        preprocessor_config_file = os.path.join(model_path, "preprocessor_config.json")
        config = {}
        if os.path.isfile(preprocessor_config_file):
            try:
                with open(preprocessor_config_file, "r", encoding="utf-8") as json_file:
                    config = json.load(json_file)
                valid_keys = signature(FeatureExtractor.__init__).parameters.keys()
                config = {k: v for k, v in config.items() if k in valid_keys}
            except json.JSONDecodeError as e:
                self.logger.warning(
                    "Could not load preprocessor_config.json: %s", str(e)
                )

        return config

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
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
        log_prob_low_threshold: Optional[float] = -2.0,
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
            of symbols as defined in the model config.json file.
          without_timestamps: Only sample text tokens.
          max_initial_timestamp: The initial timestamp cannot be later than this.
          word_timestamps: Extract word-level timestamps using the cross-attention pattern
            and dynamic time warping, and include the timestamps for each word in each segment.
          prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the next word
          append_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the previous word
          multilingual: If True, perform transcription on multilingual videos and return the transcript based
            on the 'output_language' flag.
          output_language: Valid only if multilingual is set to True. Specifies the string representing the output language. One of
            'en' (English) or 'hybrid' (code-switched transcription).
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

        #setting vad chunks if enabled
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

        encoder_output = None
        all_language_probs = None

        #setting output_language for multilingual videos
        if multilingual:
            if output_language is None:
                output_language = "en"
            elif output_language not in ["en","hybrid"]:
                raise ValueError("Output language needs to be one of 'en'/'hybrid'.")

        #detecting the language if not provided
        if language is None:
            if not self.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                segment = features[:, : self.feature_extractor.nb_max_frames]
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
            log_prob_low_threshold = log_prob_low_threshold,
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
            multilingual = multilingual,
            output_language = output_language,
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

    def generate_segments(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        encoder_output: Optional[ctranslate2.StorageView] = None,
    ) -> Iterable[Segment]:
        content_frames = features.shape[-1] - self.feature_extractor.nb_max_frames
        idx = 0
        seek = 0
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
        while seek < content_frames:
            time_offset = seek * self.feature_extractor.time_per_frame
            segment = features[:, seek : seek + self.feature_extractor.nb_max_frames]
            segment_size = min(
                self.feature_extractor.nb_max_frames, content_frames - seek
            )
            segment_duration = segment_size * self.feature_extractor.time_per_frame

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Processing segment at %s", format_timestamp(time_offset)
                )

            previous_tokens = all_tokens[prompt_reset_since:]
            
            if encoder_output is None:
                encoder_output = self.encode(segment)

            # Perform language detection at every segment to update task based on output language, 
            # if the language is english, task is transcribe, else the task is translate to english (default settings) or transcribe if 'output_language' is 'hybrid'
            if options.multilingual: 
                results = self.model.detect_language(encoder_output)
                language_token, language_probability = results[0][0]
                language = language_token[2:-2]
                if options.output_language == "en" and language != "en":
                    task = "translate"
                else:
                    task = "transcribe"
                
                #Update tokenizer based on task and language
                tokenizer = Tokenizer(
                    self.hf_tokenizer,
                    self.model.is_multilingual,
                    task=task,
                    language=language,
                    )
            #Update prompt based on task and language
            prompt = self.get_prompt(
                tokenizer,
                previous_tokens,
                without_timestamps=options.without_timestamps,
                prefix=options.prefix if seek == 0 else None,
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
                
                # Skip if the logprob is very low (below the threshold value), despite no_speech_prob being low (ex: Too ambiguous outputs for input music and noise)
                if (
                    avg_logprob < options.log_prob_low_threshold 
                ):
                    should_skip = True
                if should_skip:
                    self.logger.debug(
                        "No speech threshold is met (%f > %f)",
                        result.no_speech_prob,
                        options.no_speech_threshold,
                    )

                    # fast-forward to the next segment boundary
                    seek += segment_size
                    continue

            tokens = result.sequences_ids[0]

            previous_seek = seek
            current_segments = []

            single_timestamp_ending = (
                len(tokens) >= 2
                and tokens[-2] < tokenizer.timestamp_begin
                and tokens[-1] >= tokenizer.timestamp_begin
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
                    start_timestamp_position = (
                        sliced_tokens[0] - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                        sliced_tokens[-1] - tokenizer.timestamp_begin
                    )
                    start_time = (
                        time_offset + start_timestamp_position * self.time_precision
                    )
                    end_time = (
                        time_offset + end_timestamp_position * self.time_precision
                    )

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

            if options.word_timestamps:
                self.add_word_timestamps(
                    current_segments,
                    tokenizer,
                    encoder_output,
                    segment_size,
                    options.prepend_punctuations,
                    options.append_punctuations,
                    last_speech_timestamp=last_speech_timestamp,
                )

                word_end_timestamps = [
                    w["end"] for s in current_segments for w in s["words"]
                ]
                if len(word_end_timestamps) > 0:
                    last_speech_timestamp = word_end_timestamps[-1]
                if not single_timestamp_ending and len(word_end_timestamps) > 0:
                    seek_shift = round(
                        (word_end_timestamps[-1] - time_offset) * self.frames_per_second
                    )

                    if seek_shift > 0:
                        seek = previous_seek + seek_shift

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

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

        features = np.expand_dims(features, 0)
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
                max_length=self.max_length,
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
            ):
                needs_fallback = False  # silence

            if not needs_fallback:
                break
        else:
            # all failed, select the result with the highest average log probability
            decode_result = max(
                below_cr_threshold_results or all_results, key=lambda x: x[1]
            )

        return decode_result

    def get_prompt(
        self,
        tokenizer: Tokenizer,
        previous_tokens: List[int],
        without_timestamps: bool = False,
        prefix: Optional[str] = None,
    ) -> List[int]:
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
    ) -> None:
        if len(segments) == 0:
            return

        text_tokens_per_segment = [
            [token for token in segment["tokens"] if token < tokenizer.eot]
            for segment in segments
        ]

        text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
        alignment = self.find_alignment(
            tokenizer, text_tokens, encoder_output, num_frames
        )
        word_durations = np.array([word["end"] - word["start"] for word in alignment])
        word_durations = word_durations[word_durations.nonzero()]
        median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
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

        time_offset = (
            segments[0]["seek"]
            * self.feature_extractor.hop_length
            / self.feature_extractor.sampling_rate
        )

        word_index = 0

        for segment, text_tokens in zip(segments, text_tokens_per_segment):
            saved_tokens = 0
            words = []

            while word_index < len(alignment) and saved_tokens < len(text_tokens):
                timing = alignment[word_index]

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
                if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
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

                last_speech_timestamp = segment["end"]

            segment["words"] = words

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

        result = self.model.align(
            encoder_output,
            tokenizer.sot_sequence,
            [text_tokens],
            num_frames,
            median_filter_width=median_filter_width,
        )[0]

        text_token_probs = result.text_token_probs

        alignments = result.alignments
        text_indices = np.array([pair[0] for pair in alignments])
        time_indices = np.array([pair[1] for pair in alignments])

        words, word_tokens = tokenizer.split_to_word_tokens(
            text_tokens + [tokenizer.eot]
        )
        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        if len(word_boundaries) <= 1:
            return []

        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] / self.tokens_per_second
        start_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]
        word_probabilities = [
            np.mean(text_token_probs[i:j])
            for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
        ]

        return [
            dict(
                word=word, tokens=tokens, start=start, end=end, probability=probability
            )
            for word, tokens, start, end, probability in zip(
                words, word_tokens, start_times, end_times, word_probabilities
            )
        ]

    def encode_batch(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = get_ctranslate2_storage(features)
        return self.model.encode(features, to_cpu=to_cpu)

    def generate_segment_batched(self, features: np.ndarray, tokenizer: Tokenizer, options: TranscriptionOptions, encoder_output = None):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode_batch(features)

        result = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                beam_size=options.beam_size,
                patience=options.patience,
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
            )

        tokens_batch = [x.sequences_ids[0] for x in result]

        def decode_batch(tokens: List[List[int]]) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)
        return text

    def detect_language_multi_segment(self, audio: Union[str, BinaryIO, np.ndarray], params: dict):

        """
        Language detection function - detect language based on N highly-confident segments of a language in the audio

        """
        # The threshold is used to decide if the audio is silence or not.
        # The default value is 0.02 (2.0%) which means that if more than 2.0% of the audio is silent,
        # the audio is considered as silence.

        if params['multilingual']:
            logging.warning('lang_id is not supported for multilingual audios, detecting a single major language.')

        speech_percentage_threshold = params['speech_percentage_threshold']
        language_threshold = params['language_threshold']
        num_detection_segments = params['language_detection_segments']
        vad_filter_enabled = params['vad_filter']
        vad_params = dict(min_silence_duration_ms = params['vad_min_silence_duration']) #2500

        if vad_filter_enabled:
            vad_params = VadOptions(**vad_params)

        # decode audio if it is not decoded already
        sampling_rate = self.feature_extractor.sampling_rate
        if not isinstance(audio, np.ndarray):
            audio = decode_audio(audio, sampling_rate=sampling_rate)

        # calculate duration of audio as number of seconds
        # audio.shape[0] is the number of samples in the audio
        # sampling_rate is the number of samples per second
        # if we divide the number of samples by the number of samples per second, 
        # we get the duration in seconds
        duration = audio.shape[0] / sampling_rate

        #Check if vad is enabled, and collect voiced segments
        if vad_filter_enabled:
            # get chunks of audio that contain speech
            speech_chunks = get_speech_timestamps(audio, vad_params)
            # merge chunks of audio that contain speech into a single array
            audio = collect_chunks(audio, speech_chunks)

            # calculate new duration of audio without silence
            duration_vad = audio.shape[0] / sampling_rate

            logging.debug(f"Lang ID: VAD filter removed {duration - duration_vad} sec of audio")
            
            # if the audio after VAD is less than 2% of the original audio, consider it as silence
            if duration_vad/duration < speech_percentage_threshold: 
                return {'language_code': 'silence','language_confidence': 1.0} 

            # update duration to be the duration after VAD
            duration = duration_vad

        # if the duration of the audio is less than 1 second, consider it as silence
        if duration < 1.0:
            return {'language_code': 'silence','language_confidence': 1.0} 

        # number of feature frames in 30 seconds of audio is 3000
        nb_max_frames = self.feature_extractor.nb_max_frames

        # TODO: need to check if it fails for long audios and if we need to split the audio
        
        # extract features from audio with padding (default)
        features = self.feature_extractor(audio)

        # number of segments in the audio 
        num_segments = features.shape[-1] // nb_max_frames

        if num_detection_segments > num_segments:
            logging.warning(f'Lang ID: Can not have more number of segments than possible with the duration of file, setting {num_segments} segments.')
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
        # Note: we need to get sufficient number of confident predictions per language, not in total.

        for i in indices:
            segment_features = features[:, i*nb_max_frames : (i+1)*nb_max_frames]
            try:
                encoder_output = self.encode(segment_features)
                results = self.model.detect_language(encoder_output)[0]
                
            except ValueError as e: #or RuntimeError
                logging.error(f'Inference error:{e}' )

            # results is the list of classes (languages) and their probabilities in the decreasing order,
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
                
                # Add language and probability to the list of languages when it is confident on prediction
                confident_language_probabilities[language].append(language_probability)

                # return the language when sufficient number of confident segments is achieved
                if num_confident_segments_per_language[language] >= num_detection_segments:
                    # Considering the average probability of only confident segments
                    return {'language_code': language,'language_confidence': np.average(confident_language_probabilities[language])} 

        # if we are unable to get sufficient number of confident predictions,
        # return the most frequently detected language (if there is a tie, return the one with maximum average probability)
        counter = Counter(detected_languages)
        
        # Define the key function to select frequent language with attached probabilities
        def key_func(language):
            # Calculate the frequency of the language
            frequency = counter[language]

            # Calculate the average probability of the language
            prob_avg = sum(all_language_probabilities[language]) / len(all_language_probabilities[language])
            
            return (frequency, prob_avg)
        
        max_language = None

        if detected_languages: 
            
            # Use the key function to find the language with maximum frequency and probability
            max_language = max(detected_languages, key = key_func)
            max_probability = sum(all_language_probabilities[max_language]) / len(all_language_probabilities[max_language])

            # Do additional checks for silence for non-confident case
            # calculate RMS amplitude and DC offset
            dc_offset = np.mean(audio)
            audio_minus_dc_offset = audio - dc_offset
            is_silent = np.all(abs(audio) < 0.01) or np.sqrt(np.mean(audio_minus_dc_offset**2)) < 0.01

            if is_silent:
                return {'language_code': 'silence','language_confidence': 1.0} 

            if max_language is not None:
                return  {'language_code': max_language,'language_confidence': max_probability}
        
        # Language is not detected for any segment and none of prev conditions met
        return {'language_code': 'silence','language_confidence': 1.0} 


def load_model_batch(whisper_arch,
               device,
               device_index=0,
               compute_type="float16",
               asr_options=None,
               language : Optional[str] = None,
               model=None,
               task="transcribe",
               download_root=None,
               threads=4):
    '''Load a Whisper model for inference.
    Args:
        whisper_arch: str - The name of the Whisper model to load.
        device: str - The device to load the model on.
        compute_type: str - The compute type to use for the model.
        options: dict - A dictionary of options to use for the model.
        language: str - The language of the model. (use English for now)
        download_root: Optional[str] - The root directory to download the model to.
        threads: int - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    '''

    if whisper_arch.endswith(".en"):
        language = "en"

    model = WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         cpu_threads=threads)
    if language is not None:
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        model.logger.warning("No language specified, language will be first detected for each audio file (increases inference time).")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0, 
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,# False for timings
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "log_prob_low_threshold": -2.0,
        "multilingual": False,
        "output_language": 'en',
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)


    default_asr_options = TranscriptionOptions(**default_asr_options)

    return BatchedInferencePipeline(
        model=model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
    )


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
