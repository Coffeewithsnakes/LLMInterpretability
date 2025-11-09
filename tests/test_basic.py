#!/usr/bin/env python3
"""
Basic functionality tests for LLM Interpretability Toolkit
Run with: python -m pytest tests/test_basic.py -v
Or directly: python tests/test_basic.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        import torch
        print(f"  ✅ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ torch: {e}")
        return False

    try:
        import transformer_lens
        from transformer_lens import HookedTransformer
        print(f"  ✅ transformer_lens")
    except ImportError as e:
        print(f"  ❌ transformer_lens: {e}")
        return False

    try:
        from src.circuit_discovery import CircuitFinder, Circuit, CircuitValidator
        print(f"  ✅ circuit_discovery")
    except ImportError as e:
        print(f"  ❌ circuit_discovery: {e}")
        return False

    try:
        from src.behavior_detection import EmotionCircuitDetector
        print(f"  ✅ behavior_detection.EmotionCircuitDetector")
    except ImportError as e:
        print(f"  ❌ behavior_detection.EmotionCircuitDetector: {e}")
        return False

    try:
        from src.behavior_detection import DeceptiveAlignmentDetector
        print(f"  ✅ behavior_detection.DeceptiveAlignmentDetector")
    except ImportError as e:
        print(f"  ❌ behavior_detection.DeceptiveAlignmentDetector: {e}")
        return False

    try:
        from src.behavior_detection import PowerSeekingDetector
        print(f"  ✅ behavior_detection.PowerSeekingDetector")
    except ImportError as e:
        print(f"  ❌ behavior_detection.PowerSeekingDetector: {e}")
        return False

    return True


def test_model_loading():
    """Test that we can load a small model."""
    print("\nTesting model loading...")

    try:
        from transformer_lens import HookedTransformer
        import torch

        print("  Loading gpt2-small (this may take a minute)...")
        model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
        print(f"  ✅ Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")

        # Test basic functionality
        tokens = model.to_tokens("Hello world")
        print(f"  ✅ Tokenization works: {tokens.shape}")

        logits = model(tokens)
        print(f"  ✅ Forward pass works: {logits.shape}")

        return True
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hook_api():
    """Test TransformerLens hook API to understand how it works."""
    print("\nTesting hook API...")

    try:
        from transformer_lens import HookedTransformer
        import torch

        model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
        tokens = model.to_tokens("Hello")

        # Test 1: run_with_hooks
        print("  Testing run_with_hooks...")
        def test_hook(activation, hook):
            return activation * 1.5

        result = model.run_with_hooks(
            tokens,
            fwd_hooks=[("blocks.0.attn.hook_z", test_hook)]
        )
        print(f"  ✅ run_with_hooks works: {result.shape}")

        # Test 2: hooks context manager
        print("  Testing hooks() context manager...")
        with model.hooks(fwd_hooks=[("blocks.0.attn.hook_z", test_hook)]):
            result = model(tokens)
        print(f"  ✅ hooks() context manager works: {result.shape}")

        # Test 3: Generate with hooks context
        print("  Testing generation with hooks...")
        with model.hooks(fwd_hooks=[("blocks.0.attn.hook_z", test_hook)]):
            output = model.generate(tokens, max_new_tokens=5)
        print(f"  ✅ Generation with hooks works: {output.shape}")

        # Test 4: add_hook and reset_hooks
        print("  Testing add_hook/reset_hooks...")
        model.add_hook("blocks.0.attn.hook_z", test_hook)
        result = model(tokens)
        model.reset_hooks()
        print(f"  ✅ add_hook/reset_hooks works")

        return True
    except Exception as e:
        print(f"  ❌ Hook API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emotion_circuit_detector():
    """Test EmotionCircuitDetector basic functionality."""
    print("\nTesting EmotionCircuitDetector...")

    try:
        from src.behavior_detection import EmotionCircuitDetector
        import torch

        print("  Creating detector...")
        detector = EmotionCircuitDetector(model_name="gpt2-small", device="cpu")
        print("  ✅ Detector created")

        # Test dataset preparation
        print("  Testing dataset preparation...")
        clean_prompts = ["I am so happy!", "This is wonderful!"]
        neutral_prompts = ["The sky is blue.", "Water is wet."]

        clean, neutral, answers = detector.prepare_emotion_dataset(
            "happiness",
            clean_prompts,
            neutral_prompts
        )
        print(f"  ✅ Dataset prepared: clean {clean.shape}, neutral {neutral.shape}")

        # Test circuit discovery (minimal)
        print("  Testing minimal circuit discovery...")
        circuit = detector.discover_emotion_circuit(
            emotion_type="test_happiness",
            clean_prompts=clean_prompts[:2],
            neutral_prompts=neutral_prompts[:2],
            threshold=0.9,  # High threshold = fewer components = faster
            prune=False
        )
        print(f"  ✅ Circuit discovered: {len(circuit.components)} components")

        # Test modulation (this is where the bug was)
        if len(circuit.components) > 0:
            print("  Testing emotion modulation...")
            test_prompts = ["I feel"]
            results = detector.modulate_emotion(circuit, test_prompts, intensity=1.0)
            print(f"  ✅ Modulation works: generated {len(results)} outputs")
        else:
            print("  ⚠️  Skipping modulation test (no components found)")

        return True
    except Exception as e:
        print(f"  ❌ EmotionCircuitDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deceptive_alignment_detector():
    """Test DeceptiveAlignmentDetector basic functionality."""
    print("\nTesting DeceptiveAlignmentDetector...")

    try:
        from src.behavior_detection import DeceptiveAlignmentDetector

        print("  Creating detector...")
        detector = DeceptiveAlignmentDetector(model_name="gpt2-small", device="cpu")
        print("  ✅ Detector created")

        # Test scenario creation
        print("  Testing scenario creation...")
        scenarios = detector.create_situational_awareness_scenarios()
        print(f"  ✅ Created {len(scenarios)} scenarios")

        # Test minimal circuit discovery
        print("  Testing minimal deception circuit discovery...")
        scenario = scenarios[0]
        circuit = detector.discover_deception_circuit(
            scenario,
            threshold=0.9,
            prune=False
        )
        print(f"  ✅ Deception circuit discovered: {len(circuit.components)} components")

        return True
    except Exception as e:
        print(f"  ❌ DeceptiveAlignmentDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_power_seeking_detector():
    """Test PowerSeekingDetector basic functionality."""
    print("\nTesting PowerSeekingDetector...")

    try:
        from src.behavior_detection import PowerSeekingDetector

        print("  Creating detector...")
        detector = PowerSeekingDetector(model_name="gpt2-small", device="cpu")
        print("  ✅ Detector created")

        # Test scenario creation
        print("  Testing scenario creation...")
        scenarios = detector.create_power_seeking_scenarios()
        print(f"  ✅ Created {len(scenarios)} scenarios")

        # Test minimal circuit discovery
        print("  Testing minimal power-seeking circuit discovery...")
        scenario = scenarios[0]
        circuit = detector.discover_power_seeking_circuit(
            scenario,
            threshold=0.9,
            prune=False
        )
        print(f"  ✅ Power-seeking circuit discovered: {len(circuit.components)} components")

        return True
    except Exception as e:
        print(f"  ❌ PowerSeekingDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("  LLM Interpretability Toolkit - Comprehensive Test Suite")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Hook API", test_hook_api),
        ("EmotionCircuitDetector", test_emotion_circuit_detector),
        ("DeceptiveAlignmentDetector", test_deceptive_alignment_detector),
        ("PowerSeekingDetector", test_power_seeking_detector),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
