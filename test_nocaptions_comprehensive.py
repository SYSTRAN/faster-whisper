#!/usr/bin/env python3
"""
Comprehensive test for the nocaptions fix in BatchedInferencePipeline.

This test ensures that <|nocaptions|> tokens are properly suppressed
in batched inference to prevent malformed special tokens in output.
"""

import os
import tempfile
import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline


def create_test_audio():
    """Create a simple test audio file."""
    # Create a simple sine wave audio (440 Hz for 2 seconds)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a simple sine wave + some silence
    frequency = 440.0
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Add some silence periods that might trigger nocaptions
    silence_samples = int(0.5 * sample_rate)  # 0.5 seconds of silence
    audio = np.concatenate([
        np.zeros(silence_samples),  # Initial silence
        audio,                      # Sine wave
        np.zeros(silence_samples),  # Middle silence
        audio,                      # Another sine wave
        np.zeros(silence_samples)   # Final silence
    ])
    
    return audio.astype(np.float32)


def test_nocaptions_suppression_in_real_transcription():
    """Test that nocaptions doesn't appear in actual transcription."""
    
    print("Testing nocaptions suppression in real transcription...")
    
    # Create test audio
    audio = create_test_audio()
    
    # Test with regular WhisperModel
    print("Testing regular WhisperModel...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    segments_regular, info_regular = model.transcribe(
        audio,
        language="en",
        vad_filter=True,
        word_timestamps=False
    )
    
    regular_texts = []
    for segment in segments_regular:
        regular_texts.append(segment.text)
        if "<|nocaptions|>" in segment.text:
            print(f"❌ REGULAR: Found nocaptions in segment: {segment.text}")
    
    print(f"Regular transcription produced {len(regular_texts)} segments")
    
    # Test with BatchedInferencePipeline
    print("Testing BatchedInferencePipeline...")
    batched_model = BatchedInferencePipeline(model=model)
    
    segments_batched, info_batched = batched_model.transcribe(
        audio,
        language="en",
        vad_filter=True,
        word_timestamps=False,
        batch_size=4
    )
    
    batched_texts = []
    nocaptions_found = False
    for segment in segments_batched:
        batched_texts.append(segment.text)
        if "<|nocaptions|>" in segment.text:
            print(f"❌ BATCHED: Found nocaptions in segment: {segment.text}")
            nocaptions_found = True
    
    print(f"Batched transcription produced {len(batched_texts)} segments")
    
    if not nocaptions_found:
        print("✅ SUCCESS: No nocaptions tokens found in batched transcription!")
        return True
    else:
        print("❌ FAILURE: nocaptions tokens still present in batched transcription")
        return False


def test_token_suppression():
    """Test that nocaptions tokens are properly suppressed."""
    
    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import get_suppressed_tokens
    
    print("Testing token suppression...")
    
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    tokenizer = Tokenizer(
        model.hf_tokenizer, 
        multilingual=True, 
        task="transcribe", 
        language="en"
    )
    
    # Test default suppress_tokens = [-1]
    default_suppress = [-1]
    processed_suppress = get_suppressed_tokens(tokenizer, default_suppress)
    
    # Define nocaptions component tokens: '<', '|', 'no', 'ca', 'ptions', '|', '>'
    nocaptions_tokens = [27, 91, 1771, 496, 9799, 91, 29]
    
    not_suppressed = [t for t in nocaptions_tokens if t not in processed_suppress]
    
    if len(not_suppressed) == 0:
        print("✅ SUCCESS: All nocaptions tokens are suppressed!")
        return True
    else:
        print(f"❌ FAILURE: These nocaptions tokens are NOT suppressed: {not_suppressed}")
        return False


def main():
    """Run all tests."""
    print("Running comprehensive nocaptions fix tests...\n")
    
    test1_passed = test_token_suppression()
    print()
    test2_passed = test_nocaptions_suppression_in_real_transcription()
    
    print(f"\n{'='*50}")
    print(f"Test Results:")
    print(f"Token suppression test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Real transcription test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Overall: {'PASSED' if test1_passed and test2_passed else 'FAILED'}")
    print(f"{'='*50}")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
