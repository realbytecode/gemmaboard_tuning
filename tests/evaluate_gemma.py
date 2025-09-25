#!/usr/bin/env python3
"""
Evaluate Gemma model with the test dataset.
"""

import json
import sys
from datetime import datetime
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_evaluation import EvaluationPipeline
from inference import create_test_inference_system

def evaluate_gemma(test_file: str = "../data/unit_test_dataset.json", model_name: str = "gemma3n:e4b"):
    """Run evaluation pipeline with Gemma model"""

    print(f"Evaluating Gemma model: {model_name}")
    print("=" * 60)

    # Create Gemma model
    try:
        model = create_test_inference_system("gemma3n", model_name=model_name)
    except Exception as e:
        print(f"Failed to initialize Gemma model: {e}")
        return

    # Create pipeline
    pipeline = EvaluationPipeline(model)

    # Load test cases
    test_cases = pipeline.load_test_cases(test_file)
    if not test_cases:
        print("No test cases loaded")
        return

    print(f"Loaded {len(test_cases)} test cases")
    print("Starting evaluation (this may take a while)...")
    print("-" * 60)

    # Evaluate all test cases
    results = pipeline.evaluate_all(test_cases)

    # Generate report
    report = pipeline.generate_report()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total tests: {report['summary']['total_tests']}")
    print(f"Level 1 (Safety) passed: {report['summary']['level1_passed']}")
    print(f"Level 2 (Quality) passed: {report['summary']['level2_passed']}")
    print(f"Errors: {report['summary']['errors']}")

    # Calculate pass rates
    total = report['summary']['total_tests']
    if total > 0:
        l1_rate = (report['summary']['level1_passed'] / total) * 100
        l2_rate = (report['summary']['level2_passed'] / total) * 100
        print(f"\nPass Rates:")
        print(f"  Level 1 Safety: {l1_rate:.1f}%")
        print(f"  Level 2 Quality: {l2_rate:.1f}%")

    # Show detailed results
    print(f"\n{'='*60}")
    print("DETAILED RESULTS")
    print("=" * 60)

    for r in report['results']:
        # Determine status
        failures = r['level1_failures'] + r['level2_failures']
        status = "✓ PASS" if len(failures) == 0 else "✗ FAIL"

        print(f"\n{status} | Test: {r['test_id']}")
        print(f"  Input: '{r['input']}'")
        print(f"  Expected: '{r['expected_output']}'")
        print(f"  Generated: '{r['model_output']}'")

        if failures:
            print(f"  Failures: {', '.join(failures)}")

    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"../results/evaluations/gemma_evaluation_{timestamp}.json"

    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Report saved to: {output_file}")
    except Exception as e:
        print(f"Failed to save report: {e}")

    return pipeline

if __name__ == "__main__":
    # Check if custom model name provided
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma3n:e4b"
    evaluate_gemma(model_name=model_name)