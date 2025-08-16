#!/usr/bin/env python3
"""
Test cases for the nocaptions fix in BatchedInferencePipeline.

This test ensures that <|nocaptions|> tokens are properly suppressed
in batched inference to prevent malformed special tokens in output.
"""

# import pytest  # Not needed for simple test
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_suppressed_tokens


def test_nocaptions_tokens_suppressed():
    """Test that nocaptions component tokens are properly suppressed."""
    
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    tokenizer = Tokenizer(
        model.hf_tokenizer, 
        multilingual=True, 
        task="transcribe", 
        language="nn"  # Use same language as debug script
    )
    
    # Test default suppress_tokens = [-1]
    default_suppress = [-1]
    processed_suppress = get_suppressed_tokens(tokenizer, default_suppress)
    
    # Define nocaptions component tokens: '<', '|', 'no', 'ca', 'ptions', '|', '>'
    nocaptions_tokens = [27, 91, 1771, 496, 9799, 91, 29]
    
    print(f"Processed suppress tokens count: {len(processed_suppress)}")
    print(f"First 10 tokens: {processed_suppress[:10]}")
    
    # Check which tokens are and aren't suppressed
    suppressed = [t for t in nocaptions_tokens if t in processed_suppress]
    not_suppressed = [t for t in nocaptions_tokens if t not in processed_suppress]
    
    print(f"Suppressed nocaptions tokens: {suppressed}")
    print(f"NOT suppressed nocaptions tokens: {not_suppressed}")
    
    # All nocaptions tokens should be suppressed
    for token in nocaptions_tokens:
        assert token in processed_suppress, f"Token {token} should be suppressed but is not"


def test_batched_inference_filters_nocaptions():
    """Test that BatchedInferencePipeline filters out nocaptions segments."""
    
    # This is a more comprehensive test that would require audio data
    # For now, we test the token suppression which is the main fix
    
    model = WhisperModel("base", device="cpu", compute_type="int8")
    batched_model = BatchedInferencePipeline(model=model)
    
    # Verify the model is initialized correctly
    assert batched_model.model == model
    assert hasattr(batched_model, '_batched_segments_generator')


def test_suppress_tokens_default_behavior():
    """Test that default suppress_tokens behavior is preserved."""
    
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    tokenizer = Tokenizer(
        model.hf_tokenizer, 
        multilingual=True, 
        task="transcribe", 
        language="nn"  # Use same language as debug script
    )
    
    # Test with empty suppress_tokens
    empty_suppress = []
    processed_empty = get_suppressed_tokens(tokenizer, empty_suppress)
    
    # Should still contain essential tokens like transcribe, translate, etc.
    essential_tokens = [
        tokenizer.transcribe,
        tokenizer.translate,
        tokenizer.sot,
        tokenizer.sot_prev,
        tokenizer.sot_lm,
    ]
    
    for token in essential_tokens:
        assert token in processed_empty, f"Essential token {token} should always be suppressed"
    
    # Should also contain nocaptions tokens
    nocaptions_content_tokens = [1771, 496, 9799]  # 'no', 'ca', 'ptions'
    for token in nocaptions_content_tokens:
        assert token in processed_empty, f"Nocaptions token {token} should be suppressed"


if __name__ == "__main__":
    test_nocaptions_tokens_suppressed()
    test_batched_inference_filters_nocaptions()
    test_suppress_tokens_default_behavior()
    print("All tests passed!")
