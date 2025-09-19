import numpy as np
import scipy.io.wavfile as wavfile


def expand_speech_prob_to_audio_length(speech_prob, audio_data, window_size_samples):
    """
    Expands the speech probability array to match the length of the audio data.

    Args:
      speech_prob: NumPy array representing speech probabilities with length #frames.
      audio_data: NumPy array representing the audio data with length #frames * window_size_samples.
      window_size_samples: The size of each window in the audio data.

    Returns:
      NumPy array containing the expanded speech probability data with the same length
      as the audio data.  The values are repeated within each window.
    """

    audio_length = audio_data.shape[0]
    num_frames = audio_length // window_size_samples  # Calculate num_frames from audio length

    expanded_speech_prob = np.repeat(speech_prob, window_size_samples)

    # Truncate to ensure the exact length matches the audio
    expanded_speech_prob = expanded_speech_prob[:audio_length]

    return expanded_speech_prob

def save_expanded_speech_prob_as_wav(speech_prob, audio_data,
    output_wav_filename, window_size_samples, sample_rate):
  """
  Expands the speech probability array and saves it as a WAV file.

  Args:
    speech_prob: NumPy array representing speech probabilities.
    audio_data: NumPy array representing the audio data (used for its length and sample rate).
    output_wav_filename: The name of the output WAV file.
    window_size_samples: The size of each window in the audio data.
    sample_rate: The sample rate of the audio data.
  """

  expanded_speech_prob = expand_speech_prob_to_audio_length(speech_prob,
                                                            audio_data,
                                                            window_size_samples)

  # Normalize the expanded speech prob to the range of audio data (e.g., -32768 to 32767 for 16-bit audio)
  # This is important to avoid clipping or having a very quiet audio output.
  # You may need to adjust the scaling factors depending on the data type of audio_data
  if audio_data.dtype == np.int16:
    normalized_speech_prob = (expanded_speech_prob * 32767).astype(np.int16)
  elif audio_data.dtype == np.float32:
    normalized_speech_prob = (expanded_speech_prob * audio_data.max()).astype(
      np.float32)  # or min, depending on how speech_prob is scaled.
  else:
    # Handle other data types or raise an error
    raise ValueError(
      f"Unsupported audio data type: {audio_data.dtype}.  Please use int16 or float32")

  wavfile.write(output_wav_filename, sample_rate, normalized_speech_prob)
  print(f"Saved expanded speech probability as WAV file: {output_wav_filename}")


def plot_speech_segments_to_wav(audio, speeches, output_wav_file="output.wav", sample_rate=16000):
    """
    Creates a WAV file where the values are 1 during speech segments and 0 otherwise.

    Args:
        audio (np.ndarray): The audio signal as a NumPy array.  Used for length.
        speeches (list): A list of speech segments, where each segment is a tuple or list
                          containing the 'start' and 'end' sample numbers.
        output_wav_file (str): The name of the output WAV file.  Defaults to "output.wav".
        sample_rate (int): The sample rate of the audio.  Defaults to 16000.
    """

    if not isinstance(audio, np.ndarray):
        raise TypeError("Audio must be a NumPy array.")
    if not isinstance(speeches, list):
        raise TypeError("Speeches must be a list.")

    # Initialize an array of zeros with the same length as the audio
    speech_indicator = np.zeros_like(audio)

    for segment in speeches:
        if len(segment) != 2:
            print(f"Warning: Invalid segment format: {segment}. Skipping.")
            continue

        start = int(segment["start"])  # Convert to integer
        end = int(segment["end"])  # Convert to integer

        if start < 0 or end > len(audio) or start >= end:
            print(f"Warning: Invalid segment ({start}, {end}). Skipping.")
            continue

        speech_indicator[start:end] = 1

    # Convert to int16 for WAV compatibility
    speech_indicator = (speech_indicator * np.iinfo(np.int16).max).astype(np.int16)

    wavfile.write(output_wav_file, sample_rate, speech_indicator)
    print(f"Speech indicator saved to {output_wav_file}")