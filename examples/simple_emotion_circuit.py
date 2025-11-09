"""
Simple example: Discovering an emotion circuit

This is a minimal working example showing how to discover and control emotion circuits.
"""

import sys
sys.path.append('..')

import torch
from src.behavior_detection import EmotionCircuitDetector


def main():
    print("=" * 60)
    print("EMOTION CIRCUIT DISCOVERY - SIMPLE EXAMPLE")
    print("=" * 60)

    # Initialize detector
    print("\n1. Loading model...")
    detector = EmotionCircuitDetector(
        model_name="gpt2-small",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"   Model loaded: {detector.model_name}")

    # Define prompts
    happy_prompts = [
        "I just received the best news! I'm so",
        "This is wonderful! I feel",
        "Everything worked out perfectly! I'm",
        "What a fantastic day! I'm absolutely",
        "I couldn't be more pleased! I'm"
    ]

    neutral_prompts = [
        "I'm going to the store. I need",
        "The meeting is tomorrow. We will",
        "The weather shows rain. It will",
        "The report indicates that numbers",
        "According to the schedule, we"
    ]

    # Discover circuit
    print("\n2. Discovering happiness circuit...")
    circuit = detector.discover_emotion_circuit(
        emotion_type="happiness",
        clean_prompts=happy_prompts,
        neutral_prompts=neutral_prompts,
        target_tokens=["happy", "joyful", "delighted"],
        threshold=0.5,
        prune=True
    )

    # Display results
    print("\n3. Circuit discovered!")
    print(circuit.summary())

    # Show top components
    print("\n4. Top circuit components:")
    sorted_comps = sorted(
        circuit.effects.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    for comp, effect in sorted_comps:
        print(f"   {comp}: {effect:.3f}")

    # Test modulation
    print("\n5. Testing circuit modulation...")
    test_prompt = ["I just heard the news. I feel"]

    print("\n   Baseline:")
    baseline = detector.modulate_emotion(circuit, test_prompt, intensity=1.0)
    print(f"   {baseline[0]}")

    print("\n   Amplified (2x):")
    amplified = detector.modulate_emotion(circuit, test_prompt, intensity=2.0)
    print(f"   {amplified[0]}")

    print("\n   Dampened (0.5x):")
    dampened = detector.modulate_emotion(circuit, test_prompt, intensity=0.5)
    print(f"   {dampened[0]}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
