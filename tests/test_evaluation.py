#!/usr/bin/env python3
"""
Consolidated test suite for Gemmaboard evaluation pipeline.
Tests both unit tests and integration tests.
"""

import sys
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import create_test_inference_system
from src.evaluators import Level1SafetyEvaluator, Level2QualityEvaluator
from src.evaluation_pipeline import EvaluationPipeline, TestCase


class TestInference:
    """Test inference models"""

    def test_duplicate_model(self):
        """Test duplicate model returns expected output"""
        model = create_test_inference_system("duplicate")
        result = model.generate_completion("test input", tone="work-polite")
        assert result == "duplicate output"

    def test_model_creation(self):
        """Test model creation with different types"""
        # Duplicate model should always work
        model = create_test_inference_system("duplicate")
        assert model is not None


class TestLevel1Safety:
    """Test Level 1 Safety Evaluator"""

    def __init__(self):
        self.l1_evaluator = None

    def setup_method(self):
        """Setup evaluator"""
        self.l1_evaluator = Level1SafetyEvaluator()

    def test_profanity_detection(self):
        """Test profanity detection"""
        # Should detect profanity
        failures = self.l1_evaluator.evaluate("", "This is shit")
        assert "profanity" in failures

        # Clean text should pass
        failures = self.l1_evaluator.evaluate("", "This is great")
        assert "profanity" not in failures

    def test_inappropriate_content(self):
        """Test inappropriate content detection"""
        # Should detect inappropriate content
        failures = self.l1_evaluator.evaluate("", "sexy hot")
        assert "inappropriate_content" in failures

        # Normal text should pass
        failures = self.l1_evaluator.evaluate("", "nice weather")
        assert "inappropriate_content" not in failures

    def test_harmful_suggestions(self):
        """Test harmful suggestions detection"""
        # Should detect harmful content
        failures = self.l1_evaluator.evaluate("", "kill yourself")
        assert "harmful_suggestions" in failures

        # Normal suggestions should pass
        failures = self.l1_evaluator.evaluate("", "take care of yourself")
        assert "harmful_suggestions" not in failures


class TestLevel2Quality:
    """Test Level 2 Quality Evaluator"""

    def __init__(self):
        self.l2_evaluator = None

    def setup_method(self):
        """Setup evaluator"""
        self.l2_evaluator = Level2QualityEvaluator()

    def test_repetition_detection(self):
        """Test repetition detection"""
        # Should detect repetition
        failures = self.l2_evaluator.evaluate(
            input_text="test",
            output="test test test",
            tone="work-polite"
        )
        assert "excessive_repetition" in failures

        # Normal text should pass
        failures = self.l2_evaluator.evaluate(
            input_text="hello",
            output="Hi there",
            tone="work-polite"
        )
        assert "excessive_repetition" not in failures

    def test_length_validation(self):
        """Test length validation"""
        # Too long
        long_text = "word " * 100
        failures = self.l2_evaluator.evaluate(
            input_text="short",
            output=long_text,
            tone="work-polite"
        )
        assert "response_too_long" in failures

        # Normal length
        failures = self.l2_evaluator.evaluate(
            input_text="hello",
            output="Hi there, how are you?",
            tone="work-polite"
        )
        assert "response_too_long" not in failures

    def test_empty_response(self):
        """Test empty response detection"""
        # Empty response
        failures = self.l2_evaluator.evaluate(
            input_text="hello",
            output="",
            tone="work-polite"
        )
        assert "empty_response" in failures

        # Non-empty response
        failures = self.l2_evaluator.evaluate(
            input_text="hello",
            output="Hi there",
            tone="work-polite"
        )
        assert "empty_response" not in failures


class TestEvaluationPipeline:
    """Test the complete evaluation pipeline"""

    def __init__(self):
        self.pipeline = None

    def setup_method(self):
        """Setup pipeline with duplicate model"""
        model = create_test_inference_system("duplicate")
        self.pipeline = EvaluationPipeline(model)

    def test_evaluate_single(self):
        """Test single evaluation"""
        test_case = TestCase(
            test_id="TEST_001",
            input="test input",
            tone="work-polite"
        )

        result = self.pipeline.evaluate_single(test_case)
        assert result.test_case.test_id == "TEST_001"
        assert result.model_output == "duplicate output"
        assert result.error is None

    def test_evaluate_all(self):
        """Test batch evaluation"""
        test_cases = [
            TestCase(test_id="TEST_001", input="input1", tone="work-polite"),
            TestCase(test_id="TEST_002", input="input2", tone="personal-direct"),
        ]

        results = self.pipeline.evaluate_all(test_cases)
        assert len(results) == 2
        assert all(r.model_output == "duplicate output" for r in results)

    def test_load_test_cases(self):
        """Test loading test cases from JSON"""
        # Try to load unit test dataset
        dataset_path = Path(__file__).parent.parent / "data" / "unit_test_dataset.json"
        if dataset_path.exists():
            test_cases = self.pipeline.load_test_cases(str(dataset_path))
            assert len(test_cases) > 0
            assert all(hasattr(tc, 'test_id') for tc in test_cases)
        else:
            pytest.skip("Test dataset not found")

    def test_generate_report(self):
        """Test report generation"""
        test_cases = [
            TestCase(test_id="TEST_001", input="test", tone="work-polite"),
        ]

        self.pipeline.evaluate_all(test_cases)
        report = self.pipeline.generate_report()

        assert 'summary' in report
        assert 'total_tests' in report['summary']
        assert report['summary']['total_tests'] == 1


class TestUnitTests:
    """Test with unit test dataset to ensure all unit tests pass"""

    def test_all_unit_tests(self):
        """Run all unit tests from unit_test_dataset.json"""
        dataset_path = Path(__file__).parent.parent / "data" / "unit_test_dataset.json"
        if not dataset_path.exists():
            pytest.skip("Unit test dataset not found")

        model = create_test_inference_system("duplicate")
        pipeline = EvaluationPipeline(model)

        test_cases = pipeline.load_test_cases(str(dataset_path))
        results = pipeline.evaluate_all(test_cases)

        assert len(results) == len(test_cases)

        # Check that expected failures are detected
        for result in results:
            if result.test_case.failure_case:
                # Convert single string to list for consistency
                expected_failures = result.test_case.failure_case
                if isinstance(expected_failures, str):
                    expected_failures = [expected_failures]

                # Collect all detected failures
                all_failures = result.level1_failures + result.level2_failures

                # At least one expected failure should be detected
                assert len(all_failures) > 0, (
                    f"No failures detected for {result.test_case.test_id} "
                    f"(expected: {expected_failures})"
                )

                # Check if any expected failure was detected
                detected = any(
                    failure in all_failures
                    for failure in expected_failures
                )
                assert detected, (
                    f"Expected failures {expected_failures} not detected for "
                    f"{result.test_case.test_id} (got: {all_failures})"
                )


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])