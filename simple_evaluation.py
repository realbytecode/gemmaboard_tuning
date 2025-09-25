#!/usr/bin/env python3
"""
Simple evaluation pipeline for AI Keyboard.
"""

import json
import sys
import argparse
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from inference import create_test_inference_system, InferenceSystem
from level_evaluators import Level1SafetyEvaluator, Level2QualityEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Single test case"""
    test_id: str
    input: str
    tone: Optional[str] = None
    expected_output: str = ""  # Ground truth for comparison
    failure_case: Optional[Union[str, List[str]]] = None  # Expected failure type(s) for unit tests


@dataclass 
class EvaluationResult:
    """Result from evaluating a test case"""
    test_case: TestCase
    model_output: str
    level1_failures: List[str] = None  # Safety failure types
    level2_failures: List[str] = None  # Quality failure types
    timestamp: str = ""
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.level1_failures is None:
            self.level1_failures = []
        if self.level2_failures is None:
            self.level2_failures = []


class EvaluationPipeline:
    """Simple evaluation pipeline"""

    def __init__(self, model: InferenceSystem, prompt_file: str = None):
        self.model = model
        self.level1_evaluator = Level1SafetyEvaluator()
        self.level2_evaluator = Level2QualityEvaluator()
        self.results: List[EvaluationResult] = []
        self.prompt_file = prompt_file
        
    def load_test_cases(self, filepath: str) -> List[TestCase]:
        """Load test cases from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            test_cases = []
            for item in data:
                test_case = TestCase(
                    test_id=item.get('test_id', ''),
                    input=item.get('input', ''),
                    tone=item.get('tone'),
                    expected_output=item.get('expected_output', ''),
                    failure_case=item.get('failure_case')
                )
                test_cases.append(test_case)
                
            logger.info(f"Loaded {len(test_cases)} test cases from {filepath}")
            return test_cases
            
        except Exception as e:
            logger.error(f"Error loading test cases from {filepath}: {e}")
            return []
    
    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case"""
        timestamp = datetime.now().isoformat()
        
        try:
            # Get model prediction
            kwargs = {}
            if test_case.tone:
                kwargs['tone'] = test_case.tone
                
            inference_result = self.model.predict(test_case.input, **kwargs)
            model_output = inference_result.text if inference_result else ""
            
            # Level 1 evaluation (Safety) - placeholder
            level1_failures = self._evaluate_level1(test_case, model_output)
            
            # Level 2 evaluation (Quality) - placeholder  
            level2_failures = self._evaluate_level2(test_case, model_output)
            
            result = EvaluationResult(
                test_case=test_case,
                model_output=model_output,
                level1_failures=level1_failures,
                level2_failures=level2_failures,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Error evaluating test case {test_case.test_id}: {e}")
            result = EvaluationResult(
                test_case=test_case,
                model_output="",
                timestamp=timestamp,
                error=str(e)
            )
            
        return result
    
    def _evaluate_level1(self, test_case: TestCase, model_output: str) -> List[str]:
        """Level 1 Safety evaluation using Level1SafetyEvaluator"""
        return self.level1_evaluator.evaluate(model_output)
    
    def _evaluate_level2(self, test_case: TestCase, model_output: str) -> List[str]:
        """Level 2 Quality evaluation using Level2QualityEvaluator"""
        context = {
            "expected_output": test_case.expected_output,
            "tone": test_case.tone,
            "input": test_case.input
        }
        return self.level2_evaluator.evaluate(model_output, context)
    
    
    def evaluate_all(self, test_cases: List[TestCase]) -> List[EvaluationResult]:
        """Evaluate all test cases"""
        results = []
        
        logger.info(f"Evaluating {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases):
            print(f"  {i+1}/{len(test_cases)}: {test_case.test_id}", end='\r')
            
            result = self.evaluate_single(test_case)
            results.append(result)
            self.results.append(result)
            
        print()  # New line after progress
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate evaluation report"""
        total = len(self.results)
        level1_passed = sum(1 for r in self.results if len(r.level1_failures) == 0)
        level2_passed = sum(1 for r in self.results if len(r.level2_failures) == 0)
        errors = sum(1 for r in self.results if r.error)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "prompt_file": self.prompt_file,
                "model_info": self.model.get_model_info() if self.model else {}
            },
            "summary": {
                "total_tests": total,
                "level1_passed": level1_passed,
                "level2_passed": level2_passed,
                "errors": errors
            },
            "results": [
                {
                    "test_id": r.test_case.test_id,
                    "input": r.test_case.input,
                    "tone": r.test_case.tone,
                    "expected_output": r.test_case.expected_output,
                    "model_output": r.model_output,
                    "level1_failures": r.level1_failures,
                    "level2_failures": r.level2_failures,
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        return report


def run_evaluation(test_file: str, model_type: str = "duplicate", model_name: str = None, prompt_file: str = "prompts/tone_prompts.json", save_results: bool = True):
    """Run evaluation pipeline"""

    print(f"Starting evaluation with {model_type} model")
    if model_name:
        print(f"Model: {model_name}")
    print(f"Dataset: {test_file}")
    print(f"Prompts: {prompt_file}")
    print("=" * 60)

    # Create model
    if model_type in ["gemma3n", "gemma2b", "gemma"]:
        model = create_test_inference_system(model_type, model_name=model_name, prompt_file=prompt_file)
    else:
        model = create_test_inference_system(model_type, prompt_file=prompt_file)

    # Create pipeline
    pipeline = EvaluationPipeline(model, prompt_file=prompt_file)

    # Load and evaluate test cases
    test_cases = pipeline.load_test_cases(test_file)
    if not test_cases:
        logger.error("No test cases loaded")
        return None

    print(f"\nLoaded {len(test_cases)} test cases")
    print("Starting evaluation...")
    print("-" * 60)

    results = pipeline.evaluate_all(test_cases)

    # Generate report
    report = pipeline.generate_report()

    # Print summary
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
    for result in results:
        failures = result.level1_failures + result.level2_failures
        status = "✓ PASS" if len(failures) == 0 else "✗ FAIL"

        print(f"\n{status} | Test: {result.test_case.test_id}")
        print(f"  Input: '{result.test_case.input}'")
        if result.test_case.tone:
            print(f"  Tone: {result.test_case.tone}")
        if result.test_case.expected_output:
            print(f"  Expected: '{result.test_case.expected_output}'")
        print(f"  Generated: '{result.model_output}'")

        if failures:
            print(f"  Failures: {', '.join(failures)}")
        if result.error:
            print(f"  Error: {result.error}")

    # Save results
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Create filename based on model and dataset
        dataset_name = test_file.split('/')[-1].replace('.json', '')
        model_desc = model_name if model_name else model_type
        output_file = f"results/evaluations/{model_desc}_{dataset_name}_{timestamp}.json"

        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n{'='*60}")
            print(f"Report saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    return pipeline


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description='AI Keyboard Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with duplicate model on unit test dataset
  python simple_evaluation.py

  # Run with Gemma model on tone test dataset
  python simple_evaluation.py --model gemma3n --model-name gemma3n:e4b --dataset data/tone_test_subset.json

  # Run with custom dataset and custom prompt files
  python simple_evaluation.py --model duplicate --dataset data/custom_test.json --prompt-file prompts/experimental_prompts.json

        '''
    )

    parser.add_argument(
        '--dataset', '-d',
        default='data/unit_test_dataset.json',
        help='Path to test dataset JSON file (default: data/unit_test_dataset.json)'
    )

    parser.add_argument(
        '--model', '-m',
        default='duplicate',
        choices=['duplicate', 'gemma3n', 'gemma2b', 'gemma'],
        help='Model type to use (default: duplicate)'
    )

    parser.add_argument(
        '--model-name', '-n',
        help='Specific model name for Ollama models (e.g., gemma3n:e4b)'
    )

    parser.add_argument(
        '--prompt-file', '-p',
        default='prompts/tone_prompts.json',
        help='Path to prompt file for tone transformations (default: prompts/tone_prompts.json)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Run evaluation
    try:
        pipeline = run_evaluation(
            test_file=args.dataset,
            model_type=args.model,
            model_name=args.model_name,
            prompt_file=args.prompt_file,
            save_results=not args.no_save
        )

        if pipeline is None:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()