#!/usr/bin/env python3
"""
Unit test framework for evaluation pipeline.
Takes evaluation results and performs unit testing.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_evaluation import EvaluationPipeline, EvaluationResult, run_evaluation


@dataclass
class UnitTestResult:
    """Detailed unit test result"""
    test_id: str
    expected_failures: List[str]  # failure_cases from test case
    detected_failures: List[str]  # all failures from level1 + level2
    matched_failures: List[str]   # failures that matched expectations
    missing_failures: List[str]   # expected but not detected
    unexpected_failures: List[str] # detected but not expected
    passed: bool                   # True if all expected failures matched and no unexpected


class UnitTestFramework:
    """Unit test framework for evaluation results"""
    
    def __init__(self):
        self.unit_test_results: List[UnitTestResult] = []
    
    def run_unit_tests(self, evaluation_results: List[EvaluationResult]) -> List[UnitTestResult]:
        """Run unit tests on evaluation results"""
        unit_test_results = []
        
        for result in evaluation_results:
            # Only run unit tests if failure_case is specified
            if result.test_case.failure_case:
                unit_test_result = self._evaluate_single_unit_test(result)
                unit_test_results.append(unit_test_result)
                
        self.unit_test_results = unit_test_results
        return unit_test_results
    
    def _evaluate_single_unit_test(self, evaluation_result: EvaluationResult) -> UnitTestResult:
        """Unit test evaluation - detailed comparison of expected vs detected failures"""
        
        # Get expected failures (handle both string and list)
        if not evaluation_result.test_case.failure_case:
            expected_failures = []
        elif isinstance(evaluation_result.test_case.failure_case, list):
            expected_failures = evaluation_result.test_case.failure_case
        else:
            expected_failures = [evaluation_result.test_case.failure_case]
        
        # Get all detected failures
        detected_failures = evaluation_result.level1_failures + evaluation_result.level2_failures
        
        # Find matches using the pattern: for each expected failure, check if it's in detected
        matched_failures = []
        for failure_case in expected_failures:
            if failure_case in detected_failures:
                matched_failures.append(failure_case)
        
        # Find missing failures (expected but not detected)
        missing_failures = [f for f in expected_failures if f not in matched_failures]
        
        # Find unexpected failures (detected but not expected)  
        unexpected_failures = [f for f in detected_failures if f not in expected_failures]
        
        # Pass if all expected failures matched and no unexpected failures
        passed = len(missing_failures) == 0 and len(unexpected_failures) == 0
        
        return UnitTestResult(
            test_id=evaluation_result.test_case.test_id,
            expected_failures=expected_failures,
            detected_failures=detected_failures,
            matched_failures=matched_failures,
            missing_failures=missing_failures,
            unexpected_failures=unexpected_failures,
            passed=passed
        )
    
    def generate_unit_test_report(self) -> Dict[str, Any]:
        """Generate unit test report"""
        total = len(self.unit_test_results)
        passed = sum(1 for r in self.unit_test_results if r.passed)
        failed = total - passed
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_unit_tests": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total if total > 0 else 0
            },
            "results": [
                {
                    "test_id": r.test_id,
                    "expected": r.expected_failures,
                    "detected": r.detected_failures,
                    "matched": r.matched_failures,
                    "missing": r.missing_failures,
                    "unexpected": r.unexpected_failures,
                    "passed": r.passed
                }
                for r in self.unit_test_results
            ]
        }
        
        return report
    
    def save_unit_test_report(self, filepath: str):
        """Save unit test report to file"""
        report = self.generate_unit_test_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Unit test report saved to {filepath}")


def run_unit_tests(test_file: str, model_type: str = "duplicate"):
    """Run evaluation pipeline and then unit tests"""
    
    print("Running evaluation pipeline...")
    pipeline = run_evaluation(test_file, model_type)
    
    print("\nRunning unit tests...")
    unit_test_framework = UnitTestFramework()
    unit_test_results = unit_test_framework.run_unit_tests(pipeline.results)
    
    # Generate unit test report
    report = unit_test_framework.generate_unit_test_report()
    print(f"\nUnit Test Results:")
    print(f"Total: {report['summary']['total_unit_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Pass Rate: {report['summary']['pass_rate']:.1%}")
    
    # Show detailed results
    print(f"\nDetailed Unit Test Results:")
    for result in unit_test_results:
        status = "✓" if result.passed else "✗"
        details = f"Expected: {result.expected_failures}, Detected: {result.detected_failures}"
        if result.missing_failures:
            details += f", Missing: {result.missing_failures}"
        if result.unexpected_failures:
            details += f", Unexpected: {result.unexpected_failures}"
        print(f"{status} {result.test_id}: {details}")
    
    # Save unit test report
    output_file = f"../results/unit_tests/unit_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    unit_test_framework.save_unit_test_report(output_file)
    
    return pipeline, unit_test_framework


if __name__ == "__main__":
    run_unit_tests("../data/unit_test_dataset.json")