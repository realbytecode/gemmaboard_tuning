#!/usr/bin/env python3
"""Test Ollama connection and list available models"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama

try:
    client = ollama.Client()
    response = client.list()

    print("Available Ollama models:")
    print(f"Response type: {type(response)}")

    if hasattr(response, 'models'):
        models = response.models
    elif isinstance(response, dict) and 'models' in response:
        models = response['models']
    else:
        models = response

    for model in models:
        if isinstance(model, dict):
            print(f"  - {model.get('name', 'Unknown')}")
            print(f"    Size: {model.get('size', 'N/A')}")
            print(f"    Modified: {model.get('modified_at', 'N/A')}")
        else:
            print(f"  - {model}")
        print()

except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    import traceback
    traceback.print_exc()