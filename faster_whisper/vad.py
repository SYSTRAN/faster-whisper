import bisect
import functools
import os

from typing import List, NamedTuple, Optional

import numpy as np
import torch

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

    onset: float = 0.5
    offset: float = onset - 0.15
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 200


def get_vad_scores(
    audio: torch.Tensor, sampling_rate: int = 16000, window_size_samples: int = 512
):
    model = get_vad_model()

    padded_audio = np.pad(
        audio.numpy(), (0, window_size_samples - audio.shape[0] % window_size_samples)
    )
    scores = model(padded_audio.reshape(1, -1))

    starts = np.asarray(range(0, len(audio), window_size_samples))
    ends = starts + window_size_samples
    ends[-1] = len(audio)
    middle = (starts + ends) / 2
    timestamps = [
        {
            "start": s / sampling_rate,
            "end": e / sampling_rate,
            "middle": m / sampling_rate,
        }
        for s, e, m in zip(starts, ends, middle)
    ]

    return scores.squeeze(-1), timestamps


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
    encoder_path = os.path.join(get_assets_path(), "silero_encoder.onnx")
    decoder_path = os.path.join(get_assets_path(), "silero_decoder.onnx")
    return SileroVADModel(encoder_path, decoder_path)


class SileroVADModel:
    def __init__(self, encoder_path, decoder_path):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 0
        opts.intra_op_num_threads = 0
        opts.log_severity_level = 4

        self.encoder_session = onnxruntime.InferenceSession(
            encoder_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def __call__(
        self, audio: np.ndarray, num_samples: int = 512, context_size_samples: int = 64
    ):
        assert (
            audio.ndim == 2
        ), "Input should be a 2D tensor with size (batch_size, num_samples)"
        assert (
            audio.shape[1] % num_samples == 0
        ), "Input size should be a multiple of num_samples"

        batch_size = audio.shape[0]

        state = np.zeros((2, batch_size, 128), dtype="float32")
        context = np.zeros(
            (batch_size, context_size_samples),
            dtype="float32",
        )

        batched_audio = audio.reshape(batch_size, -1, num_samples)
        context = batched_audio[..., -context_size_samples:]
        context[:, -1] = 0
        context = np.roll(context, 1, 1)
        batched_audio = np.concatenate([context, batched_audio], 2)

        batched_audio = batched_audio.reshape(-1, num_samples + context_size_samples)

        encoder_output = self.encoder_session.run(None, {"input": batched_audio})[0]
        encoder_output = encoder_output.reshape(batch_size, -1, 128)

        decoder_outputs = []
        for window in np.split(encoder_output, encoder_output.shape[1], axis=1):
            out, state = self.decoder_session.run(
                None, {"input": window.squeeze(1), "state": state}
            )
            decoder_outputs.append(out)

        out = np.stack(decoder_outputs, axis=1).squeeze(-1)
        return out


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


def get_active_regions(
    scores: np.ndarray,
    timestamps: List[dict],
    vad_options: VadOptions = None,
):
    """Extract active regions from VAD scores.

    Args:
        scores List: Detection scores.
        timestamps List: timestamps for each score
        vad_options VadOptions: options for VAD

    Returns:
        segments List: Active regions.
    """
    if vad_options is None:
        vad_options = VadOptions()

    onset = vad_options.onset
    offset = vad_options.offset
    max_speech_duration_s = vad_options.max_speech_duration_s
    silence_threshold = vad_options.min_silence_duration_ms // 16
    speech_pad = vad_options.speech_pad_ms / 1000

    for k_scores in scores:
        # initial state
        start = timestamps[0]["start"]
        is_active = k_scores[0] > onset
        curr_scores = [k_scores[0]] if is_active else []
        curr_timestamps = [start] if is_active else []
        segments = []
        silence_counter = 0
        for timestamp, score in zip(timestamps[1:], k_scores[1:]):
            # currently active
            if is_active:
                current_duration = timestamp["end"] - start
                if current_duration > max_speech_duration_s:
                    search_after = len(curr_scores) // 2
                    # divide segment
                    min_score_div_idx = search_after + np.argmin(
                        curr_scores[search_after:]
                    )
                    min_score_t = curr_timestamps[min_score_div_idx]
                    segments.append(
                        {
                            "start": start - speech_pad,
                            "end": min_score_t["end"] + speech_pad,
                        }
                    )
                    start = curr_timestamps[min_score_div_idx]["start"]
                    curr_scores = curr_scores[min_score_div_idx + 1 :]
                    curr_timestamps = curr_timestamps[min_score_div_idx + 1 :]
                    if silence_counter != 0:
                        silence_counter = sum(s < offset for s in curr_scores)
                # switching from active to inactive
                elif score < offset:
                    silence_counter += 1
                    if silence_counter > silence_threshold:
                        segments.append(
                            {
                                "start": start - speech_pad,
                                "end": curr_timestamps[-silence_counter]["end"]
                                + speech_pad,
                            }
                        )
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []
                        silence_counter = 0
                        continue
                else:
                    silence_counter = 0

                curr_scores.append(score)
                curr_timestamps.append(timestamp)
            # currently inactive
            else:
                # switching from inactive to active
                if score > onset:
                    start = timestamp["start"]
                    is_active = True
                    silence_counter = 0

        # if active at the end, add final region
        if is_active:
            segments.append(
                {"start": start - speech_pad, "end": timestamp["end"] + speech_pad}
            )

    return segments


def merge_segments(segments_list, vad_options: VadOptions):
    curr_end = 0
    seg_idxs = []
    merged_segments = []
    edge_padding = vad_options.speech_pad_ms / 1000
    chunk_size = vad_options.max_speech_duration_s

    curr_start = segments_list[0]["start"]

    for idx, seg in enumerate(segments_list):
        # if any segment start timing is less than previous segment end timing,
        # reset the edge padding. Similarly for end timing.
        if idx > 0:
            if seg["start"] < segments_list[idx - 1]["end"]:
                seg["start"] += edge_padding
        if idx < len(segments_list) - 1:
            if seg["end"] > segments_list[idx + 1]["start"]:
                seg["end"] -= edge_padding

        if seg["end"] - curr_start > chunk_size and curr_end - curr_start > 0:
            merged_segments.append(
                {
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                }
            )
            curr_start = seg["start"]
            seg_idxs = []
        curr_end = seg["end"]
        seg_idxs.append((seg["start"], seg["end"]))
    # add final
    merged_segments.append(
        {
            "start": curr_start,
            "end": curr_end,
            "segments": seg_idxs,
        }
    )
    return merged_segments


def support_segments(segments, collar=0):
    """
    Merge segments separated by less than `collar` seconds.

    Args:
        segments (List[Dict]): List of segments with 'start' and 'end' keys.
        collar (float, optional): Maximum gap between segments to merge. Defaults to 0.

    Returns:
        List[Dict]: List of merged segments.
    """
    supported_segments = []
    current_segment = segments[0]
    current_segment["start"] = max(current_segment["start"], 0)

    for segment in segments[1:]:
        if segment["start"] - current_segment["end"] <= collar:
            current_segment["end"] = segment["end"]
        else:
            supported_segments.append(current_segment)
            current_segment = segment

    supported_segments.append(current_segment)
    return supported_segments
