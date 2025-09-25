#!/usr/bin/env python3
"""
Simple wrapper script to run evaluation from project root.
This allows easy command-line access to the evaluation pipeline.
"""

import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation_pipeline import main

if __name__ == "__main__":
    main()