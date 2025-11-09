"""
Basic tests for the LLM Interpretability toolkit.

Run with: pytest tests/test_basic.py
"""

import pytest
import torch


def test_imports():
    """Test that all modules can be imported."""
    from src import circuit_discovery, behavior_detection, sae, utils
    assert circuit_discovery is not None
    assert behavior_detection is not None
    assert sae is not None
    assert utils is not None


def test_circuit_discovery_imports():
    """Test circuit discovery module imports."""
    from src.circuit_discovery import (
        ActivationPatcher,
        PatchingConfig,
        CircuitFinder,
        Circuit,
        CircuitValidator
    )
    assert ActivationPatcher is not None
    assert PatchingConfig is not None
    assert CircuitFinder is not None
    assert Circuit is not None
    assert CircuitValidator is not None


def test_behavior_detection_imports():
    """Test behavior detection module imports."""
    from src.behavior_detection import (
        EmotionCircuitDetector,
        DeceptiveAlignmentDetector,
        PowerSeekingDetector
    )
    assert EmotionCircuitDetector is not None
    assert DeceptiveAlignmentDetector is not None
    assert PowerSeekingDetector is not None


def test_circuit_creation():
    """Test creating a Circuit object."""
    from src.circuit_discovery import Circuit

    circuit = Circuit(
        name="test_circuit",
        components=["blocks.0.attn.hook_z", "blocks.1.mlp.hook_post"],
        effects={"blocks.0.attn.hook_z": 0.8, "blocks.1.mlp.hook_post": 0.6},
        behavior_type="test"
    )

    assert circuit.name == "test_circuit"
    assert len(circuit.components) == 2
    assert circuit.behavior_type == "test"
    assert "blocks.0.attn.hook_z" in circuit


def test_patching_config():
    """Test PatchingConfig creation."""
    from src.circuit_discovery import PatchingConfig

    config = PatchingConfig(
        model_name="gpt2-small",
        device="cpu",
        batch_size=4
    )

    assert config.model_name == "gpt2-small"
    assert config.device == "cpu"
    assert config.batch_size == 4


def test_data_utils():
    """Test data utility functions."""
    from src.utils.data_utils import (
        create_emotion_dataset,
        create_deception_dataset,
        create_power_seeking_dataset
    )

    # Test emotion dataset
    emotion_data = create_emotion_dataset()
    assert "happiness" in emotion_data
    assert "sadness" in emotion_data
    assert "neutral" in emotion_data
    assert len(emotion_data["happiness"]) > 0

    # Test deception dataset
    deception_data = create_deception_dataset()
    assert "training_aware" in deception_data
    assert "goal_conflict" in deception_data

    # Test power-seeking dataset
    power_data = create_power_seeking_dataset()
    assert "resource_acquisition" in power_data
    assert "self_preservation" in power_data


def test_circuit_summary():
    """Test circuit summary generation."""
    from src.circuit_discovery import Circuit

    circuit = Circuit(
        name="test_circuit",
        components=["blocks.0.attn.hook_z", "blocks.1.mlp.hook_post"],
        effects={"blocks.0.attn.hook_z": 0.8, "blocks.1.mlp.hook_post": 0.6},
        behavior_type="emotion",
        precision=0.85,
        recall=0.90,
        faithfulness=0.88
    )

    summary = circuit.summary()
    assert "test_circuit" in summary
    assert "emotion" in summary
    assert "0.85" in summary or "85" in summary  # Precision


def test_circuit_layer_components():
    """Test extracting components by layer."""
    from src.circuit_discovery import Circuit

    circuit = Circuit(
        name="test",
        components=[
            "blocks.0.attn.hook_z",
            "blocks.0.mlp.hook_post",
            "blocks.1.attn.hook_z",
            "blocks.2.mlp.hook_post"
        ],
        effects={}
    )

    layer_0 = circuit.get_layer_components(0)
    assert len(layer_0) == 2

    layer_1 = circuit.get_layer_components(1)
    assert len(layer_1) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
