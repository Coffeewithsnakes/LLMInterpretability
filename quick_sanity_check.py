#!/usr/bin/env python3
"""
Quick Sanity Check

This script does a minimal test to verify the hook fix works.
Runs in ~30 seconds and tests the specific bug that was fixed.

Usage:
    python quick_sanity_check.py
"""

import sys

print("=" * 70)
print("  Quick Sanity Check - Testing Hook Fix")
print("=" * 70)

print("\n[1/5] Testing imports...")
try:
    from src.behavior_detection import EmotionCircuitDetector
    import torch
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nRun: pip install -r requirements.txt")
    sys.exit(1)

print("\n[2/5] Loading model...")
try:
    detector = EmotionCircuitDetector(model_name="gpt2-small", device="cpu")
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    sys.exit(1)

print("\n[3/5] Creating minimal test circuit...")
try:
    # Create a fake circuit with a few components for testing
    from src.circuit_discovery import Circuit
    test_circuit = Circuit(
        name="test_circuit",
        components=[
            "blocks.0.attn.hook_z",
            "blocks.1.mlp.hook_post"
        ],
        behavior_type="test"
    )
    print(f"✅ Test circuit created with {len(test_circuit.components)} components")
except Exception as e:
    print(f"❌ Circuit creation failed: {e}")
    sys.exit(1)

print("\n[4/5] Testing emotion modulation (THE CRITICAL TEST)...")
try:
    test_prompts = ["I feel"]
    results = detector.modulate_emotion(test_circuit, test_prompts, intensity=1.0)
    print(f"✅ Modulation successful! Generated: '{results[0][:50]}...'")
except Exception as e:
    print(f"❌ Modulation failed: {e}")
    print("\nThis is the bug that should be fixed. Error details:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[5/5] Testing with different intensities...")
try:
    amplified = detector.modulate_emotion(test_circuit, test_prompts, intensity=2.0)
    dampened = detector.modulate_emotion(test_circuit, test_prompts, intensity=0.5)
    print(f"✅ All intensity levels work")
    print(f"   Normal (1.0): {results[0][:40]}...")
    print(f"   Amplified (2.0): {amplified[0][:40]}...")
    print(f"   Dampened (0.5): {dampened[0][:40]}...")
except Exception as e:
    print(f"❌ Intensity test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("  🎉 ALL CHECKS PASSED!")
print("=" * 70)
print("\n✅ The hook fix is working correctly!")
print("✅ Emotion modulation works with hooks context manager!")
print("✅ Generation with hooks works!")
print("\nYou can now run the full demo with: python run_interactive.py")
print("=" * 70)
