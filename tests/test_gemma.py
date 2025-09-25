#!/usr/bin/env python3
"""Test Gemma model directly"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import create_test_inference_system

# Test with the exact model name from Ollama
print("Testing Gemma3n:e4b model:")
print("-" * 40)

try:
    # Use exact model name
    inference_system = create_test_inference_system("gemma3n", model_name="gemma3n:e4b")

    test_cases = [
        ("I disagree with this idea", "work-polite"),
        ("Send the report", "work-direct"),
        ("Can't make it", "personal-polite"),
    ]

    for input_text, tone in test_cases:
        result = inference_system.predict(input_text, tone=tone)
        if result:
            print(f"\nInput: '{input_text}'")
            print(f"Tone: {tone}")
            print(f"Output: '{result.text}'")
            if not result.text.startswith("[Error"):
                print(f"Latency: {result.latency_ms:.1f}ms")
                print(f"Tokens: {result.metadata.get('prompt_tokens', 0)} prompt, {result.metadata.get('completion_tokens', 0)} completion")

except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()