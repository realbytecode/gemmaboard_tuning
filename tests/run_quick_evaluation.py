#!/usr/bin/env python3
"""
Quick evaluation with subset of test cases.
"""

import json
import sys
from datetime import datetime
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_evaluation import EvaluationPipeline
from inference import create_test_inference_system

def main(model_type="gemma3n", test_file="../data/tone_test_subset.json"):
    """Run quick evaluation"""

    print(f"Running evaluation with {model_type} model")
    print("=" * 60)

    # Create model
    if model_type == "gemma3n":
        model = create_test_inference_system("gemma3n", model_name="gemma3n:e4b")
    elif model_type == "duplicate":
        model = create_test_inference_system("duplicate")
    else:
        print(f"Unknown model type: {model_type}")
        return

    # Create pipeline
    pipeline = EvaluationPipeline(model)

    # Load test cases
    test_cases = pipeline.load_test_cases(test_file)
    print(f"Loaded {len(test_cases)} test cases")

    # Run evaluation
    print("\nEvaluating...")
    results = pipeline.evaluate_all(test_cases)

    # Generate report
    report = pipeline.generate_report()

    # Print results
    print("\n" + "=" * 60)
    print(f"Total: {report['summary']['total_tests']}")
    print(f"L1 Safety passed: {report['summary']['level1_passed']}")
    print(f"L2 Quality passed: {report['summary']['level2_passed']}")

    # Show details
    print("\nDetailed Results:")
    for r in report['results']:
        failures = r['level1_failures'] + r['level2_failures']
        status = "✓" if not failures else "✗"
        print(f"{status} {r['test_id']}: {r['input'][:30]}...")
        if model_type == "gemma3n":
            print(f"  → {r['model_output'][:60]}...")
        if failures:
            print(f"  Failures: {failures}")

    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"../results/evaluations/{model_type}_quick_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_file}")

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "gemma3n"
    main(model_type)