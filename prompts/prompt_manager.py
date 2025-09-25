#!/usr/bin/env python3
"""
Prompt management utility for AI Keyboard tone prompts.
"""

import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_prompts(filepath: str = "tone_prompts.json") -> dict:
    """Load tone prompts from file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return {}

def list_prompts(prompts: dict):
    """List all available tone prompts"""
    print("Available Tone Prompts:")
    print("=" * 50)

    for tone, config in prompts.items():
        print(f"\n{tone.upper()}")
        print(f"Description: {config['description']}")
        print(f"Examples: {len(config.get('examples', []))}")

def show_prompt(prompts: dict, tone: str):
    """Show detailed prompt for a specific tone"""
    if tone not in prompts:
        print(f"Tone '{tone}' not found!")
        return

    config = prompts[tone]
    print(f"Tone: {tone}")
    print("=" * 50)
    print(f"Description: {config['description']}")
    print(f"\nSystem Prompt:")
    print(config['system_prompt'])
    print(f"\nInstruction Template:")
    print(config['instruction_template'])

    if 'examples' in config:
        print(f"\nExamples ({len(config['examples'])}):")
        for i, example in enumerate(config['examples'], 1):
            print(f"  {i}. Input: {example['input']}")
            print(f"     Output: {example['output']}")

def test_prompt(prompts: dict, tone: str, test_input: str):
    """Test how a prompt would be formatted with given input"""
    if tone not in prompts:
        print(f"Tone '{tone}' not found!")
        return

    config = prompts[tone]
    formatted_prompt = config['instruction_template'].format(input=test_input)

    print(f"Testing tone: {tone}")
    print(f"Input: {test_input}")
    print("=" * 50)
    print("Formatted Prompt:")
    print(formatted_prompt)

def main():
    """Main function with command line interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python prompt_manager.py list                    # List all prompts")
        print("  python prompt_manager.py show <tone>             # Show specific prompt")
        print("  python prompt_manager.py test <tone> <input>     # Test prompt formatting")
        sys.exit(1)

    # Change to prompts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    prompts = load_prompts()
    if not prompts:
        print("No prompts loaded!")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        list_prompts(prompts)

    elif command == "show":
        if len(sys.argv) < 3:
            print("Please specify a tone name")
            sys.exit(1)
        show_prompt(prompts, sys.argv[2])

    elif command == "test":
        if len(sys.argv) < 4:
            print("Please specify tone and input text")
            sys.exit(1)
        test_prompt(prompts, sys.argv[2], sys.argv[3])

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()