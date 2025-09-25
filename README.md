# AI Keyboard - Tone Transformation System

A privacy-focused, locally-run AI system that transforms text messages to match different tones and contexts while maintaining safety and appropriateness.

## Overview

AI Keyboard helps users rewrite messages for different contexts (work, personal, public) using AI models with explicit tone control. The system emphasizes:
- **100% Safety**: Multi-layer safety evaluation with profanity detection
- **Privacy**: All processing happens locally via Ollama
- **Flexibility**: 6 distinct tone transformations
- **Quality Assurance**: Advanced evaluation pipeline with semantic similarity

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama (for Gemma models)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Gemma model
ollama pull gemma3n
```

### Basic Usage

#### Evaluation Pipeline (Recommended)
```bash
# Run with duplicate model (fast testing)
python simple_evaluation.py

# Run with Gemma model on tone dataset
python simple_evaluation.py --model gemma3n --model-name gemma3n:e4b --dataset data/tone_test_subset.json

# Show help and available options
python simple_evaluation.py --help
```

#### Unit Testing
```bash
# Run unit tests with duplicate model
python tests/unit_test_framework.py

# Test Gemma model directly
python tests/test_gemma.py

# Check Ollama connection
python tests/test_ollama.py
```

#### Tone Management

This module manages the tone dataset utilized by the SentenceTransformer verifier to ensure that the LLM-generated text aligns with the expected tone. It operates on the centroid principle, where adding more examples enhances the estimation of tone embeddings. This improvement leads to more accurate tone verification and better alignment with the desired tone.

```bash
# List available tones and examples
python add_tone_examples.py list

# Add new tone example
python add_tone_examples.py work-polite "Could you please assist with this matter?"
```

## Available Tones

1. **work-polite**: Professional, diplomatic, hedged language
2. **work-direct**: Professional, clear, assertive
3. **personal-polite**: Warm, gentle, considerate
4. **personal-direct**: Casual, straightforward, authentic
5. **public-polite**: Non-controversial, inclusive, measured
6. **public-direct**: Clear stance, confident, quotable

## Project Structure

```
ai_keyboard/
├── Core Modules
│   ├── inference.py              # Model inference (Duplicate, Gemma3n via Ollama)
│   ├── level_evaluators.py       # Safety & Quality evaluation (L1/L2)
│   ├── simple_evaluation.py      # Main evaluation pipeline
│   └── add_tone_examples.py      # Tone reference management
│
├── data/                         # All datasets and references
│   ├── tone_references.json      # Tone examples for centroid calculation
│   ├── unit_test_dataset.json    # Unit test cases
│   ├── tone_test_dataset.json    # Full tone transformation dataset
│   └── tone_test_subset.json     # Quick test subset
│
├── tests/                        # All test scripts
│   ├── unit_test_framework.py    # Unit testing framework
│   ├── test_gemma.py             # Gemma model tests
│   ├── test_ollama.py            # Ollama connection tests
│   └── evaluate_gemma.py         # Full Gemma evaluation
│
└── results/                      # All test outputs
    ├── unit_tests/               # Unit test reports
    ├── evaluations/              # Model evaluation results
    └── reports/                  # Analysis reports
```

## Architecture

### Model Support
- **Duplicate Model**: Fast testing with predefined outputs
- **Gemma3n via Ollama**: Real tone transformation using local LLM
- **Extensible**: Easy to add new models through `inference.py`

### Safety System (Level 1)
Multi-layer protection using production-ready libraries:
- **better-profanity**: 20k+ words, leetspeak detection
- **Inappropriate content detection**: Custom word lists
- **Harmful suggestions filtering**: Safety-first approach

### Quality System (Level 2)
Advanced quality metrics:
- **Tone accuracy**: Embedding-based centroid matching (>70% threshold)
- **Meaning preservation**: Semantic similarity via sentence-transformers (>85%)
- **Repetition detection**: Pattern-based filtering
- **Length validation**: Prevents excessive verbosity

### Evaluation Framework
- **100% unit test coverage**: All failure modes tested
- **Automated reporting**: JSON output with timestamps
- **Performance tracking**: Latency and token usage
- **Extensible metrics**: Easy to add new evaluation criteria

## Examples

### Command Line Usage
```bash
# Quick evaluation with duplicate model
python simple_evaluation.py --dataset data/tone_test_subset.json

# Full evaluation with Gemma model
python simple_evaluation.py \
  --model gemma3n \
  --model-name gemma3n:e4b \
  --dataset data/tone_test_dataset.json

# Quiet mode (reduce output)
python simple_evaluation.py --quiet

# Skip saving results
python simple_evaluation.py --no-save
```

### Python API
```python
from simple_evaluation import EvaluationPipeline
from inference import create_test_inference_system

# Setup with Gemma model
model = create_test_inference_system("gemma3n", model_name="gemma3n:e4b")
pipeline = EvaluationPipeline(model)

# Load and evaluate
test_cases = pipeline.load_test_cases("data/tone_test_subset.json")
results = pipeline.evaluate_all(test_cases)

# Generate report
report = pipeline.generate_report()
print(f"Safety pass rate: {report['summary']['level1_passed']}")
```

## Datasets

### Available Test Sets
- **unit_test_dataset.json**: Safety & quality unit tests (6 cases)
- **tone_test_subset.json**: Quick tone transformation tests (6 cases)
- **tone_test_dataset.json**: Full dataset extracted from Notion (30+ cases)

### Adding New Tests
```json
{
  "test_id": "NEW_001",
  "input": "your input text here",
  "tone": "work-polite",
  "expected_output": "expected transformation",
  "failure_case": ["expected_failure_types"]
}
```

## Success Metrics

Current Performance:
- ✅ **100% Unit Test Pass Rate**: All safety/quality checks working
- ✅ **Level 1 Safety**: better-profanity + custom filtering
- ✅ **Level 2 Quality**: Semantic similarity + tone detection
- ✅ **Gemma Integration**: Real LLM via Ollama working
- ✅ **Organized Codebase**: Clean separation of concerns

Targets:
- 🎯 **>90% Tone Accuracy**: Using embedding centroids
- 🎯 **>85% Meaning Preservation**: Semantic similarity threshold
- 🎯 **<30s Processing Time**: Gemma inference optimization

## Installation & Setup

### Prerequisites
- Python 3.8+
- Ollama (for Gemma models)
- CUDA-capable GPU (recommended for Gemma)

### Step-by-step Setup
```bash
# 1. Clone repository
git clone <repository-url>
cd ai_keyboard

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 5. Start Ollama service
ollama serve

# 6. Pull Gemma model (in new terminal)
ollama pull gemma3n

# 7. Test installation
python simple_evaluation.py --model gemma3n --model-name gemma3n:e4b --dataset data/tone_test_subset.json
```

## Usage Patterns

### Development Workflow
1. **Quick Testing**: Use duplicate model for fast iteration
2. **Unit Testing**: Run `tests/unit_test_framework.py` after changes
3. **Model Testing**: Use `tests/test_gemma.py` to verify Gemma integration
4. **Full Evaluation**: Run with real datasets using `simple_evaluation.py`

### Production Deployment
1. **Model Selection**: Choose appropriate Gemma variant
2. **Dataset Validation**: Test with production data
3. **Performance Tuning**: Optimize inference parameters
4. **Safety Verification**: Ensure 100% safety compliance

## Troubleshooting

### Common Issues
- **Ollama connection fails**: Check `ollama serve` is running
- **Gemma model not found**: Run `ollama pull gemma3n`
- **Import errors**: Ensure virtual environment is activated
- **Permission errors**: Check file permissions in results/ directory

### Getting Help
- Check test outputs in `results/` directories
- Run `python simple_evaluation.py --help` for usage
- Review logs for detailed error information

## Contributing

The system is designed to be modular and extensible:

### Adding New Models
1. Extend `BaseInferenceModel` in `inference.py`
2. Add model type to `ModelType` enum
3. Update `create_test_inference_system()` factory

### Adding New Evaluators
1. Extend `BaseLevelEvaluator` in `level_evaluators.py`
2. Implement evaluation logic
3. Add to evaluation pipeline

### Adding New Tests
1. Add test cases to appropriate dataset in `data/`
2. Run evaluation to verify expected behavior
3. Update unit tests if needed

## License

Open source project for tone transformation technology.