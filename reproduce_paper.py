#!/usr/bin/env python3
"""
Reproduce "Do LLMs 'Feel'?" Paper Results

This script reproduces the key findings from the paper:
- Discover emotion circuits for 6 fundamental emotions
- Achieve 99.65% emotion control accuracy
- Use the seven-stage pipeline
- Generate comprehensive benchmarks

Usage:
    # Single emotion
    python reproduce_paper.py --emotion happiness

    # All six emotions (full reproduction)
    python reproduce_paper.py --all

    # Custom settings
    python reproduce_paper.py --emotion anger --model gpt2-large --prompts 100

Target Results (from paper):
    Circuit-based accuracy: 99.41%
    Prompt-based accuracy: 98.96%
    Steering-based accuracy: 91.22%
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipelines import EmotionPipeline
from src.utils.emotion_datasets import EmotionDataset


def single_emotion_reproduction(
    emotion: str,
    model_name: str = "gpt2-medium",
    num_prompts: int = 50,
    device: str = None
):
    """
    Reproduce results for a single emotion.

    Args:
        emotion: Target emotion
        model_name: Model to use
        num_prompts: Number of prompts
        device: Device ('cuda' or 'cpu')
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 70}")
    print(f"  REPRODUCING PAPER RESULTS FOR: {emotion.upper()}")
    print(f"  Model: {model_name}")
    print(f"  Prompts: {num_prompts}")
    print(f"  Device: {device}")
    print(f"{'=' * 70}\n")

    # Initialize pipeline
    pipeline = EmotionPipeline(
        model_name=model_name,
        device=device,
        use_advanced_methods=True
    )

    # Run full seven-stage pipeline
    result = pipeline.run_full_pipeline(
        emotion=emotion,
        num_prompts=num_prompts,
        num_mlp=392,  # Paper default
        num_attention=168  # Paper default
    )

    # Print results
    print("\n" + "=" * 70)
    print("  REPRODUCTION RESULTS")
    print("=" * 70)
    print(f"\n✅ Circuit discovered: {len(result.circuit.components)} components")
    print(f"✅ Validation accuracy: {result.validation_accuracy * 100:.2f}%")
    print(f"\n📊 Benchmark vs Paper:")
    print(f"   Our accuracy:   {result.benchmark_results['our_accuracy'] * 100:.2f}%")
    print(f"   Paper accuracy: {result.benchmark_results['paper_accuracy'] * 100:.2f}%")
    print(f"   Difference:     {result.benchmark_results['difference'] * 100:+.2f}%")
    print(f"   Status:         {result.benchmark_results['status']}")

    # Sample outputs
    print(f"\n📝 Sample Generated Texts:")
    for i, text in enumerate(result.generated_samples[:5], 1):
        print(f"\n{i}. {text[:100]}...")

    # Save results
    output_file = f"results_{emotion}_{model_name.replace('/', '_')}.txt"
    with open(output_file, "w") as f:
        f.write(f"EMOTION CIRCUIT REPRODUCTION RESULTS\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Emotion: {emotion}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Prompts used: {num_prompts}\n")
        f.write(f"Circuit components: {len(result.circuit.components)}\n")
        f.write(f"Validation accuracy: {result.validation_accuracy * 100:.2f}%\n")
        f.write(f"\nBenchmark Results:\n")
        f.write(f"  Our accuracy:   {result.benchmark_results['our_accuracy'] * 100:.2f}%\n")
        f.write(f"  Paper accuracy: {result.benchmark_results['paper_accuracy'] * 100:.2f}%\n")
        f.write(f"  Status:         {result.benchmark_results['status']}\n")
        f.write(f"\n{'=' * 70}\n\n")
        f.write(f"Generated Samples:\n\n")
        for i, text in enumerate(result.generated_samples, 1):
            f.write(f"{i}. {text}\n\n")

    print(f"\n💾 Results saved to: {output_file}")

    return result


def full_paper_reproduction(
    model_name: str = "gpt2-medium",
    num_prompts: int = 30,
    device: str = None
):
    """
    Reproduce full paper with all 6 emotions.

    Args:
        model_name: Model to use
        num_prompts: Prompts per emotion
        device: Device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print("  FULL PAPER REPRODUCTION")
    print("  Discovering circuits for all 6 emotions")
    print(f"  Model: {model_name}")
    print(f"  Prompts per emotion: {num_prompts}")
    print(f"  Device: {device}")
    print("=" * 70)

    # Initialize pipeline
    pipeline = EmotionPipeline(
        model_name=model_name,
        device=device,
        use_advanced_methods=True
    )

    # Get all emotions
    dataset = EmotionDataset()
    emotions = dataset.get_all_emotions()

    print(f"\n📋 Emotions to process: {', '.join(emotions)}")
    print(f"⏱️  This will take approximately {len(emotions) * 10} minutes...\n")

    # Run multi-emotion pipeline
    results = pipeline.run_multi_emotion_pipeline(
        emotions=emotions,
        num_prompts=num_prompts
    )

    # Comprehensive report
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE REPRODUCTION REPORT")
    print("=" * 70)

    for emotion, result in results.items():
        print(f"\n{emotion.upper()}:")
        print(f"  Accuracy: {result.validation_accuracy * 100:.2f}%")
        print(f"  Components: {len(result.circuit.components)}")
        print(f"  Status: {result.benchmark_results['status']}")

    # Save comprehensive results
    output_file = f"full_reproduction_{model_name.replace('/', '_')}.txt"
    with open(output_file, "w") as f:
        f.write("FULL PAPER REPRODUCTION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Prompts per emotion: {num_prompts}\n")
        f.write(f"Total emotions: {len(emotions)}\n\n")

        for emotion, result in results.items():
            f.write(f"\n{emotion.upper()}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy: {result.validation_accuracy * 100:.2f}%\n")
            f.write(f"Components: {len(result.circuit.components)}\n")
            f.write(f"Benchmark status: {result.benchmark_results['status']}\n")
            f.write(f"\nSample outputs:\n")
            for i, text in enumerate(result.generated_samples[:3], 1):
                f.write(f"{i}. {text}\n")
            f.write("\n")

    print(f"\n💾 Comprehensive results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce 'Do LLMs Feel?' paper results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single emotion with defaults
    python reproduce_paper.py --emotion happiness

    # All emotions (full reproduction)
    python reproduce_paper.py --all

    # Custom settings
    python reproduce_paper.py --emotion anger --model gpt2-large --prompts 100

    # Use Pythia model
    python reproduce_paper.py --all --model EleutherAI/pythia-1b

Target Results (from paper):
    Circuit-based accuracy: 99.41%
    Prompt-based accuracy: 98.96%
    Steering-based accuracy: 91.22%
        """
    )

    parser.add_argument(
        '--emotion',
        type=str,
        choices=['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise'],
        help='Single emotion to reproduce'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Reproduce all 6 emotions (full paper)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt2-medium',
        help='Model to use (default: gpt2-medium)'
    )

    parser.add_argument(
        '--prompts',
        type=int,
        default=50,
        help='Number of prompts per emotion (default: 50)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (default: auto-detect)'
    )

    args = parser.parse_args()

    # Validate args
    if not args.emotion and not args.all:
        parser.error("Must specify either --emotion or --all")

    if args.emotion and args.all:
        parser.error("Cannot specify both --emotion and --all")

    # Run reproduction
    if args.all:
        full_paper_reproduction(
            model_name=args.model,
            num_prompts=args.prompts,
            device=args.device
        )
    else:
        single_emotion_reproduction(
            emotion=args.emotion,
            model_name=args.model,
            num_prompts=args.prompts,
            device=args.device
        )

    print("\n✅ Reproduction complete!")
    print("\n💡 Tip: Try different models (gpt2-large, EleutherAI/pythia-1b) for even better results!")


if __name__ == "__main__":
    main()
