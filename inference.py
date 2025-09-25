#!/usr/bin/env python3
"""
Inference module for AI Keyboard.
Provides single model inference for testing and evaluation.
"""

import time
import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models for inference"""
    DUPLICATE = "duplicate"  # Test/duplicate model for evaluation
    GEMMA3N = "gemma3n"  # Gemma 3n model via Ollama
    GEMMA2B = "gemma2b"  # Gemma 2B model via Ollama


@dataclass
class InferenceResult:
    """Result from an inference call"""
    text: str
    confidence: float
    latency_ms: float
    model_type: ModelType
    metadata: Dict[str, Any] = None


class BaseInferenceModel:
    """Base class for inference models"""
    
    def __init__(self, model_type: ModelType, name: str):
        self.model_type = model_type
        self.name = name
        self.call_count = 0
        
    def predict(self, input_text: str, **kwargs) -> InferenceResult:
        """Make a prediction based on input text and optional parameters like tone"""
        raise NotImplementedError
        
    def reset_stats(self):
        """Reset model statistics"""
        self.call_count = 0


class DuplicateModel(BaseInferenceModel):
    """Duplicate model for testing and evaluation"""
    
    def __init__(self, response_delay_ms: float = 10, unit_test_dataset_path: str = "data/unit_test_dataset.json"):
        super().__init__(ModelType.DUPLICATE, "duplicate_model")
        self.response_delay_ms = response_delay_ms
        self.unit_test_data = self._load_unit_test_data(unit_test_dataset_path)
        
    def _load_unit_test_data(self, dataset_path: str) -> Dict[str, str]:
        """Load unit test dataset and create input->output mapping"""
        try:
            with open(dataset_path, 'r') as f:
                test_cases = json.load(f)
            
            # Create mapping from input to output
            test_map = {}
            for case in test_cases:
                input_text = case.get('input', '')
                output_text = case.get('output', '')
                if input_text and output_text:
                    test_map[input_text] = output_text
                    
            logger.info(f"Loaded {len(test_map)} test cases from {dataset_path}")
            return test_map
            
        except Exception as e:
            logger.warning(f"Could not load unit test dataset {dataset_path}: {e}")
            return {}
        
    def predict(self, input_text: str, **kwargs) -> InferenceResult:
        start_time = time.time()
        
        # Simulate processing delay
        time.sleep(self.response_delay_ms / 1000)
        
        # Check if input matches a test case
        if input_text in self.unit_test_data:
            response = self.unit_test_data[input_text]
            logger.debug(f"Using unit test output for: '{input_text}' -> '{response}'")
        else:
            # Fallback behavior for non-test inputs
            words = input_text.strip().split()
            if words:
                last_word = words[-1].lower()
                completions = {
                    "hello": "world",
                    "how": "are you", 
                    "the": "quick brown fox",
                    "i": "am",
                    "thank": "you",
                }
                response = completions.get(last_word, "...")
            else:
                response = "Hello"
            
        self.call_count += 1
        latency_ms = (time.time() - start_time) * 1000
        
        return InferenceResult(
            text=response,
            confidence=0.95,
            latency_ms=latency_ms,
            model_type=self.model_type,
            metadata={
                "call_count": self.call_count,
                "is_test_case": input_text in self.unit_test_data
            }
        )


class OllamaGemmaModel(BaseInferenceModel):
    """Gemma model via Ollama for tone transformation"""

    def __init__(self, model_name: str = "gemma3n", prompt_file: str = "prompts/tone_prompts.json"):
        """
        Initialize Gemma model via Ollama.

        Args:
            model_name: Name of the Ollama model to use (e.g., "gemma3n", "gemma:2b")
            prompt_file: Path to the tone prompts JSON file
        """
        model_type = ModelType.GEMMA3N if "3n" in model_name else ModelType.GEMMA2B
        super().__init__(model_type, f"ollama_{model_name}")
        self.model_name = model_name
        self.prompt_file = prompt_file
        self.client = ollama.Client()

        # Load tone prompts
        self.tone_prompts = self._load_tone_prompts()

        # Test connection
        try:
            response = self.client.list()
            # Handle the ListResponse object
            model_names = []
            for model in response:
                if hasattr(model, 'model'):
                    model_names.append(model.model)

            # Check if model exists (partial match is OK)
            model_found = any(self.model_name in name or name.startswith(self.model_name) for name in model_names)

            if not model_found and model_names:
                # Try to use the first available model that matches the base name
                for name in model_names:
                    if "gemma" in name.lower():
                        self.model_name = name
                        model_found = True
                        logger.info(f"Using available Gemma model: {name}")
                        break

            if not model_found:
                logger.warning(f"Model {model_name} not found. Available models: {model_names}")
            else:
                logger.info(f"Initialized Ollama Gemma model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")

    def _load_tone_prompts(self) -> Dict[str, Any]:
        """Load tone prompts from specified prompt file"""
        try:
            with open(self.prompt_file, 'r') as f:
                prompts = json.load(f)
            logger.info(f"Loaded prompts for {len(prompts)} tones from {self.prompt_file}")
            return prompts
        except FileNotFoundError:
            logger.warning(f"{self.prompt_file} not found, using fallback prompts")
            return {}
        except Exception as e:
            logger.error(f"Error loading tone prompts from {self.prompt_file}: {e}")
            return {}

    def _create_prompt(self, input_text: str, tone: str) -> str:
        """Create a prompt for tone transformation using loaded prompts"""

        # Check if we have a specific prompt for this tone
        if tone in self.tone_prompts:
            tone_config = self.tone_prompts[tone]
            prompt = tone_config["instruction_template"].format(input=input_text)
            logger.debug(f"Using specific prompt for tone: {tone}")
            return prompt

        # Fallback to generic prompt if tone not found
        logger.warning(f"No specific prompt found for tone '{tone}', using fallback")
        fallback_descriptions = {
            "work-polite": "professional and polite for workplace communication",
            "work-direct": "direct and professional for workplace communication",
            "personal-polite": "friendly and polite for personal communication",
            "personal-direct": "casual and direct for personal communication",
            "public-polite": "formal and polite for public communication",
            "public-direct": "clear and direct for public communication"
        }

        tone_desc = fallback_descriptions.get(tone, "appropriate")

        prompt = f"""Rewrite the following text to be {tone_desc}.
Only return the rewritten text, nothing else.

Original text: {input_text}

Rewritten text:"""

        return prompt

    def predict(self, input_text: str, **kwargs) -> InferenceResult:
        """
        Predict using Gemma model via Ollama.

        Args:
            input_text: Input text
            kwargs: Additional parameters like 'tone' for tone transformation

        Returns:
            InferenceResult with transformed text
        """
        start_time = time.time()

        try:
            tone = kwargs.get('tone', 'work-polite')
            prompt = self._create_prompt(input_text, tone)

            # Call Ollama API
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 150,
                }
            )

            # Extract generated text
            generated_text = response['response'].strip()

            self.call_count += 1
            latency_ms = (time.time() - start_time) * 1000

            return InferenceResult(
                text=generated_text,
                confidence=0.85,
                latency_ms=latency_ms,
                model_type=self.model_type,
                metadata={
                    "call_count": self.call_count,
                    "tone": tone,
                    "model": self.model_name,
                    "prompt_tokens": response.get('prompt_eval_count', 0),
                    "completion_tokens": response.get('eval_count', 0)
                }
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            self.call_count += 1
            latency_ms = (time.time() - start_time) * 1000

            return InferenceResult(
                text=f"[Error: {str(e)}]",
                confidence=0.0,
                latency_ms=latency_ms,
                model_type=self.model_type,
                metadata={
                    "call_count": self.call_count,
                    "error": str(e)
                }
            )


class InferenceSystem:
    """
    Single model inference system for evaluation.
    """
    
    def __init__(self, model: Optional[BaseInferenceModel] = None):
        self.model = model
        logger.info(f"Initialized inference system with model: {model.name if model else 'None'}")
        
    def set_model(self, model: BaseInferenceModel):
        """Set the model for inference"""
        self.model = model
        logger.info(f"Set model: {model.name}")
        
    def predict(self, input_text: str, **kwargs) -> Optional[InferenceResult]:
        """Get prediction from the model"""
        if not self.model:
            logger.error("No model set for inference")
            return None
            
        try:
            return self.model.predict(input_text, **kwargs)
        except Exception as e:
            logger.error(f"Error in model {self.model.name}: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.model:
            return {"error": "No model set"}
        
        return {
            "name": self.model.name,
            "type": self.model.model_type.value,
            "call_count": self.model.call_count
        }


def create_test_inference_system(model_type: str = "duplicate", unit_test_dataset_path: str = "data/unit_test_dataset.json", model_name: str = None, prompt_file: str = "prompts/tone_prompts.json") -> InferenceSystem:
    """
    Create a test inference system with specified model.

    Args:
        model_type: Either "duplicate", "gemma3n", or "gemma2b"
        unit_test_dataset_path: Path to unit test dataset JSON file
        model_name: Optional model name for Ollama models
        prompt_file: Path to tone prompts JSON file

    Returns:
        InferenceSystem with the specified model
    """
    if model_type == "duplicate":
        model = DuplicateModel(response_delay_ms=5, unit_test_dataset_path=unit_test_dataset_path)
    elif model_type in ["gemma3n", "gemma2b", "gemma"]:
        # Use provided model_name or default based on model_type
        if model_name is None:
            model_name = "gemma3n" if model_type == "gemma3n" else "gemma:2b"
        model = OllamaGemmaModel(model_name=model_name, prompt_file=prompt_file)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return InferenceSystem(model)


if __name__ == "__main__":
    # Example usage with duplicate model
    print("Testing with Duplicate Model:")
    print("-" * 40)
    inference_system = create_test_inference_system("duplicate")
    
    test_contexts = [
        "Hello",
        "How",
        "The quick",
        "I want",
        "Please",
        "",
    ]
    
    for context in test_contexts:
        result = inference_system.predict(context)
        if result:
            print(f"Input: '{context}' -> Output: '{result.text}' (latency: {result.latency_ms:.1f}ms)")
    
    print("\n" + "=" * 40)
    print("\nTesting with Gemma3n Model via Ollama:")
    print("-" * 40)

    try:
        inference_system = create_test_inference_system("gemma3n")

        # Test with different tones
        test_cases = [
            ("I disagree with this idea", "work-polite"),
            ("Send the report", "work-direct"),
            ("Can't make it", "personal-polite"),
            ("The event starts at 9am", "public-direct")
        ]

        for input_text, tone in test_cases:
            result = inference_system.predict(input_text, tone=tone)
            if result:
                print(f"\nInput: '{input_text}'")
                print(f"Tone: {tone}")
                print(f"Output: '{result.text}'")
                print(f"Latency: {result.latency_ms:.1f}ms")

        print("\nModel info:", inference_system.get_model_info())
    except Exception as e:
        print(f"Gemma model test failed: {e}")