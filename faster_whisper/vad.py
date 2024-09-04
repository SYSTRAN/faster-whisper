import bisect
import functools
import os

from abc import ABC
from collections.abc import Callable
from typing import List, NamedTuple, Optional, Union

import numpy as np
import torch

from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import Annotation, Segment, SlidingWindowFeature

from faster_whisper.utils import get_assets_path


# The code below is adapted from https://github.com/snakers4/silero-vad.
class VadOptions(NamedTuple):
    """VAD options.

    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
    """

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400


def get_speech_timestamps(
    audio: torch.Tensor,
    vad_options: Optional[VadOptions] = None,
    **kwargs,
) -> List[dict]:
    """This method is used for splitting long audios into speech chunks using silero VAD.

    Args:
      audio: One dimensional float array.
      vad_options: Options for VAD processing.
      kwargs: VAD options passed as keyword arguments for backward compatibility.

    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """
    if vad_options is None:
        vad_options = VadOptions(**kwargs)

    threshold = vad_options.threshold
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    window_size_samples = 512
    speech_pad_ms = vad_options.speech_pad_ms
    sampling_rate = 16000
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    model = get_vad_model()
    state, context = model.get_initial_states(batch_size=1)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = np.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob, state, context = model(chunk, state, context, sampling_rate)
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15

    # to save potential segment end (and tolerate some silence)
    temp_end = 0
    # to save potential segment limits in case of maximum segment size reached
    prev_end = next_start = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                # previously reached silence (< neg_thres) and is still not speech (< thres)
                if next_start < prev_end:
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech["end"] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            # condition to avoid cutting in very short silence
            if (window_size_samples * i) - temp_end > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    return speeches


def collect_chunks(audio: torch.Tensor, chunks: List[dict]) -> torch.Tensor:
    """Collects and concatenates audio chunks."""
    if not chunks:
        return torch.tensor([], dtype=torch.float32)

    return torch.cat([audio[chunk["start"] : chunk["end"]] for chunk in chunks])


class SpeechTimestampsMap:
    """Helper class to restore original speech timestamps."""

    def __init__(self, chunks: List[dict], sampling_rate: int, time_precision: int = 2):
        self.sampling_rate = sampling_rate
        self.time_precision = time_precision
        self.chunk_end_sample = []
        self.total_silence_before = []

        previous_end = 0
        silent_samples = 0

        for chunk in chunks:
            silent_samples += chunk["start"] - previous_end
            previous_end = chunk["end"]

            self.chunk_end_sample.append(chunk["end"] - silent_samples)
            self.total_silence_before.append(silent_samples / sampling_rate)

    def get_original_time(
        self,
        time: float,
        chunk_index: Optional[int] = None,
    ) -> float:
        if chunk_index is None:
            chunk_index = self.get_chunk_index(time)

        total_silence_before = self.total_silence_before[chunk_index]
        return round(total_silence_before + time, self.time_precision)

    def get_chunk_index(self, time: float) -> int:
        sample = int(time * self.sampling_rate)
        return min(
            bisect.bisect(self.chunk_end_sample, sample),
            len(self.chunk_end_sample) - 1,
        )


@functools.lru_cache
def get_vad_model():
    """Returns the VAD model instance."""
    path = os.path.join(get_assets_path(), "silero_vad.onnx")
    return SileroVADModel(path)


class SileroVADModel:
    def __init__(self, path):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4

        self.session = onnxruntime.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def get_initial_states(self, batch_size: int):
        state = np.zeros((2, batch_size, 128), dtype=np.float32)
        context = np.zeros((batch_size, 64), dtype=np.float32)
        return state, context

    def __call__(self, x, state, context, sr: int):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if len(x.shape) > 2:
            raise ValueError(
                f"Too many dimensions for input audio chunk {len(x.shape)}"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        x = np.concatenate([context, x], axis=1)

        ort_inputs = {
            "input": x,
            "state": state,
            "sr": np.array(sr, dtype="int64"),
        }

        out, state = self.session.run(None, ort_inputs)
        context = x[..., -64:]

        return out, state, context


# BSD 2-Clause License

# Copyright (c) 2024, Max Bain

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# The code below is copied from whisper-x (https://github.com/m-bain/whisperX)
# and adapted for faster_whisper.
class SegmentX:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker


class VoiceActivitySegmentation(VoiceActivityDetection, ABC):
    """Pipeline wrapper class for Voice Activity Segmentation based on VAD scores."""

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        device: Optional[Union[str, torch.device]] = None,
        fscore: bool = False,
        use_auth_token: Optional[str] = None,
        **inference_kwargs,
    ):
        """Initialize the pipeline with the model name and the optional device.

        Args:
            dict parameters of VoiceActivityDetection class from pyannote:
            segmentation (PipelineModel): Loaded model name.
            device (torch.device or None): Device to perform the segmentation.
            fscore (bool): Flag indicating whether to compute F-score during inference.
            use_auth_token (str or None): Optional authentication token for model access.
            inference_kwargs (dict):  Additional arguments from VoiceActivityDetection pipeline.
        """
        super().__init__(
            segmentation=segmentation,
            device=device,
            fscore=fscore,
            use_auth_token=use_auth_token,
            **inference_kwargs,
        )

    def apply(
        self, file: AudioFile, hook: Optional[Callable] = None
    ) -> SlidingWindowFeature:
        """Apply voice activity detection on the audio file.

        Args:
            file (AudioFile): Processed file.
            hook (callable): Hook called with signature: hook("step_name", step_artefact, file=file)

        Returns:
            segmentations (SlidingWindowFeature): Voice activity segmentation.
        """
        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model if needed
        # output shape is (num_chunks, num_frames, 1)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file)

        return segmentations


class BinarizeVadScores:
    """Binarize detection scores using hysteresis thresholding.

    Reference:
        Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
        RNN-based Voice Activity Detection", InterSpeech 2015.

        Modified by Max Bain to include WhisperX's min-cut operation
        https://arxiv.org/abs/2303.00747

    """

    def __init__(
        self,
        onset: float = 0.5,
        offset: Optional[float] = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
        max_duration: float = float("inf"),
    ):
        """Initializes the parameters for Binarizing the VAD scores.

        Args:
            onset (float, optional):
                Onset threshold. Defaults to 0.5.
            offset (float, optional):
                Offset threshold. Defaults to `onset`.
            min_duration_on (float, optional):
                Remove active regions shorter than that many seconds. Defaults to 0s.
            min_duration_off (float, optional):
                Fill inactive regions shorter than that many seconds. Defaults to 0s.
            pad_onset (float, optional):
                Extend active regions by moving their start time by that many seconds.
                Defaults to 0s.
            pad_offset (float, optional):
                Extend active regions by moving their end time by that many seconds.
                Defaults to 0s.
            max_duration (float):
                The maximum length of an active segment.
        """
        super().__init__()

        self.onset = onset
        self.offset = offset or onset

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

        self.max_duration = max_duration

    def __get_active_regions(self, scores: SlidingWindowFeature) -> Annotation:
        """Extract active regions from VAD scores.

        Args:
            scores (SlidingWindowFeature): Detection scores.

        Returns:
            active (Annotation): Active regions.
        """
        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]
        # annotation meant to store 'active' regions
        active = Annotation()
        for k, k_scores in enumerate(scores.data.T):
            label = k if scores.labels is None else scores.labels[k]

            # initial state
            start = timestamps[0]
            is_active = k_scores[0] > self.onset
            curr_scores = [k_scores[0]]
            curr_timestamps = [start]
            t = start
            # optionally add `strict=False` for python 3.10 or later
            for t, y in zip(timestamps[1:], k_scores[1:]):
                # currently active
                if is_active:
                    curr_duration = t - start
                    if curr_duration > self.max_duration:
                        search_after = len(curr_scores) // 2
                        # divide segment
                        min_score_div_idx = search_after + np.argmin(
                            curr_scores[search_after:]
                        )
                        min_score_t = curr_timestamps[min_score_div_idx]
                        region = Segment(
                            start - self.pad_onset, min_score_t + self.pad_offset
                        )
                        active[region, k] = label
                        start = curr_timestamps[min_score_div_idx]
                        curr_scores = curr_scores[min_score_div_idx + 1 :]
                        curr_timestamps = curr_timestamps[min_score_div_idx + 1 :]
                    # switching from active to inactive
                    elif y < self.offset:
                        region = Segment(start - self.pad_onset, t + self.pad_offset)
                        active[region, k] = label
                        start = t
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []
                    curr_scores.append(y)
                    curr_timestamps.append(t)
                # currently inactive
                else:
                    # switching from inactive to active
                    if y > self.onset:
                        start = t
                        is_active = True

            # if active at the end, add final region
            if is_active:
                region = Segment(start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label

        return active

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        """Binarize detection scores.

        Args:
            scores (SlidingWindowFeature): Detection scores.

        Returns:
            active (Annotation): Binarized scores.
        """
        active = self.__get_active_regions(scores)
        # because of padding, some active regions might be overlapping: merge them.
        # also: fill same speaker gaps shorter than min_duration_off
        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            if self.max_duration < float("inf"):
                raise NotImplementedError("This would break current max_duration param")
            active = active.support(collar=self.min_duration_off)

        # remove tracks shorter than min_duration_on
        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active


def merge_chunks(
    segments,
    chunk_length,
    onset: float = 0.5,
    offset: Optional[float] = None,
    edge_padding: float = 0.1,
):
    """
    Merge operation described in whisper-x paper
    """
    curr_end = 0
    merged_segments = []
    seg_idxs = []
    speaker_idxs = []

    assert chunk_length > 0
    binarize = BinarizeVadScores(max_duration=chunk_length, onset=onset, offset=offset)
    segments = binarize(segments)
    segments_list = []
    for speech_turn in segments.get_timeline():
        segments_list.append(
            SegmentX(
                max(0.0, speech_turn.start - edge_padding),
                speech_turn.end + edge_padding,
                "UNKNOWN",
            )
        )  # 100ms edge padding to account for edge errors

    if len(segments_list) == 0:
        print("No active speech found in audio")
        return []

    # Make sur the starting point is the start of the segment.
    curr_start = segments_list[0].start

    for idx, seg in enumerate(segments_list):
        # if any segment start timing is less than previous segment end timing,
        # reset the edge padding. Similarly for end timing.
        if idx > 0:
            if seg.start < segments_list[idx - 1].end:
                seg.start += edge_padding
        if idx < len(segments_list) - 1:
            if seg.end > segments_list[idx + 1].start:
                seg.end -= edge_padding

        if seg.end - curr_start > chunk_length and curr_end - curr_start > 0:
            merged_segments.append(
                {
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                }
            )
            curr_start = seg.start
            seg_idxs = []
            speaker_idxs = []
        curr_end = seg.end
        seg_idxs.append((seg.start, seg.end))
        speaker_idxs.append(seg.speaker)
    # add final
    merged_segments.append(
        {
            "start": curr_start,
            "end": curr_end,
            "segments": seg_idxs,
        }
    )
    return merged_segments
