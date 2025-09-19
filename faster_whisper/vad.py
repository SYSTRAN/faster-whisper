import bisect
from typing import Literal

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from faster_whisper.vad_predictor.silero_vad import SileroVADModel, \
    get_vad_model
from faster_whisper.vad_predictor.ten_vad import TenVadPredictor
from faster_whisper.vad_predictor.debug import save_expanded_speech_prob_as_wav, \
    plot_speech_segments_to_wav


# The code below is adapted from https://github.com/snakers4/silero-vad.
@dataclass
class VadOptions:
    """VAD options.

    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      neg_threshold: Silence threshold for determining the end of speech. If a probability is lower
        than neg_threshold, it is always considered silence. Values higher than neg_threshold
        are only considered speech if the previous sample was classified as speech; otherwise,
        they are treated as silence. This parameter helps refine the detection of speech
         transitions, ensuring smoother segment boundaries.
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
    neg_threshold: float = None
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400
    model: Literal["silero_vad", "ten_vad"] = "silero_vad"


def get_speech_timestamps(
    audio: np.ndarray,
    vad_options: Optional[VadOptions] = None,
    sampling_rate: int = 16000,
    **kwargs,
) -> List[dict]:
    """This method is used for splitting long audios into speech chunks using silero VAD.

    Args:
      audio: One dimensional float array.
      vad_options: Options for VAD processing.
      sampling_rate: Sampling rate of the audio.
      kwargs: VAD options passed as keyword arguments for backward compatibility.

    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """
    if vad_options is None:
        vad_options = VadOptions(**kwargs)

    threshold = vad_options.threshold
    neg_threshold = vad_options.neg_threshold
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    window_size_samples = 512 if vad_options.model == "silero_vad" else 256
    speech_pad_ms = vad_options.speech_pad_ms
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

    model = get_vad_model() if vad_options.model == "silero_vad" else TenVadPredictor()
    padded_audio = np.pad(
        audio, (0, window_size_samples - audio.shape[0] % window_size_samples)
    )

    speech_probs = model(padded_audio.reshape(1, -1), num_samples = window_size_samples).squeeze(0)
# should be: ndarray shape=(7121, 1) dtype = float32
# is: ndarray (14242,) dtype
    save_expanded_speech_prob_as_wav(speech_probs, audio, "/tmp/out.wav", window_size_samples, sampling_rate)

    triggered = False
    speeches = []
    current_speech = {}
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

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

    plot_speech_segments_to_wav(audio, speeches, "/tmp/speeches.wav", sampling_rate)
    return speeches


def collect_chunks(
    audio: np.ndarray, chunks: List[dict], sampling_rate: int = 16000
) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    """Collects audio chunks."""
    if not chunks:
        chunk_metadata = {
            "start_time": 0,
            "end_time": 0,
        }
        return [np.array([], dtype=np.float32)], [chunk_metadata]

    audio_chunks = []
    chunks_metadata = []
    for chunk in chunks:
        chunk_metadata = {
            "start_time": chunk["start"] / sampling_rate,
            "end_time": chunk["end"] / sampling_rate,
        }
        audio_chunks.append(audio[chunk["start"] : chunk["end"]])
        chunks_metadata.append(chunk_metadata)
    return audio_chunks, chunks_metadata


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



def merge_segments(segments_list, vad_options: VadOptions, sampling_rate: int = 16000):
    if not segments_list:
        return []

    curr_end = 0
    seg_idxs = []
    merged_segments = []
    edge_padding = vad_options.speech_pad_ms * sampling_rate // 1000
    chunk_length = vad_options.max_speech_duration_s * sampling_rate

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

        if seg["end"] - curr_start > chunk_length and curr_end - curr_start > 0:
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
