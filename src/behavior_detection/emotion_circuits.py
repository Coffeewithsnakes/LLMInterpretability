"""
Emotion Circuit Detection

Reproduces the methodology from "Do LLMs 'Feel'? Emotion Circuits Discovery and Control"
This serves as a template for discovering circuits for other behaviors.

Pipeline:
1. Analytical Decomposition: Identify neurons and attention heads for emotions
2. Causal Validation: Use activation patching to verify importance
3. Circuit Modulation: Control emotion expression via circuit manipulation
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import numpy as np
from tqdm import tqdm

from ..circuit_discovery import CircuitFinder, Circuit, CircuitValidator


@dataclass
class EmotionDataset:
    """Dataset for emotion circuit discovery."""

    emotion_type: str  # e.g., "happiness", "sadness", "anger"
    clean_prompts: List[str]  # Prompts that should express the emotion
    neutral_prompts: List[str]  # Neutral prompts (no emotion)
    target_tokens: List[str]  # Expected emotion-expressing tokens


class EmotionCircuitDetector:
    """
    Discovers and validates emotion circuits in language models.

    Based on the paper methodology:
    1. Identify neurons with high activation on emotion-expressing examples
    2. Identify attention heads that attend to emotion-related tokens
    3. Validate components using activation patching
    4. Assemble into a complete circuit
    5. Demonstrate control via circuit modulation
    """

    def __init__(
        self,
        model_name: str = "gpt2-small",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
            cache_dir=cache_dir
        )
        self.model.eval()

        # Initialize circuit finder
        self.circuit_finder = CircuitFinder(model_name, self.device, cache_dir)
        self.validator = CircuitValidator(self.model)

        # Cache for discovered circuits
        self.circuits: Dict[str, Circuit] = {}

    def prepare_emotion_dataset(
        self,
        emotion_type: str,
        clean_prompts: List[str],
        neutral_prompts: List[str],
        target_tokens: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for emotion circuit discovery.

        Args:
            emotion_type: Type of emotion (e.g., "happiness", "anger")
            clean_prompts: Prompts expressing the emotion
            neutral_prompts: Neutral prompts
            target_tokens: Expected emotion words

        Returns:
            Tuple of (clean_inputs, neutral_inputs, answer_tokens)
        """
        # Tokenize prompts
        clean_tokens = self.model.to_tokens(clean_prompts)
        neutral_tokens = self.model.to_tokens(neutral_prompts)

        # Pad to same sequence length (required for activation patching)
        max_len = max(clean_tokens.shape[1], neutral_tokens.shape[1])

        if clean_tokens.shape[1] < max_len:
            # Pad clean tokens
            padding = torch.zeros(
                (clean_tokens.shape[0], max_len - clean_tokens.shape[1]),
                dtype=clean_tokens.dtype,
                device=clean_tokens.device
            )
            clean_tokens = torch.cat([clean_tokens, padding], dim=1)

        if neutral_tokens.shape[1] < max_len:
            # Pad neutral tokens
            padding = torch.zeros(
                (neutral_tokens.shape[0], max_len - neutral_tokens.shape[1]),
                dtype=neutral_tokens.dtype,
                device=neutral_tokens.device
            )
            neutral_tokens = torch.cat([neutral_tokens, padding], dim=1)

        # Prepare answer tokens if provided
        if target_tokens:
            answer_token_ids = []
            for token_str in target_tokens:
                token_id = self.model.to_single_token(token_str)
                answer_token_ids.append(token_id)

            answer_tokens = torch.tensor(answer_token_ids, device=self.device)
        else:
            answer_tokens = None

        return clean_tokens, neutral_tokens, answer_tokens

    def discover_emotion_circuit(
        self,
        emotion_type: str,
        clean_prompts: List[str],
        neutral_prompts: List[str],
        target_tokens: Optional[List[str]] = None,
        threshold: float = 0.5,
        prune: bool = True
    ) -> Circuit:
        """
        Discover a complete circuit for a specific emotion.

        Args:
            emotion_type: Emotion to discover circuit for
            clean_prompts: Prompts expressing the emotion
            neutral_prompts: Neutral control prompts
            target_tokens: Expected emotion-expressing tokens
            threshold: Component inclusion threshold
            prune: Whether to prune circuit to minimal set

        Returns:
            Discovered emotion circuit
        """
        print(f"Discovering circuit for emotion: {emotion_type}")

        # Prepare data
        clean_inputs, neutral_inputs, answer_tokens = self.prepare_emotion_dataset(
            emotion_type, clean_prompts, neutral_prompts, target_tokens
        )

        # Define metric function for emotion
        def emotion_metric(logits):
            """Measure how much the model expresses the target emotion."""
            if answer_tokens is not None:
                # Logit difference between emotion and neutral tokens
                emotion_logits = logits[..., -1, answer_tokens[0]]
                return emotion_logits.mean()
            else:
                # Use probability distribution entropy as proxy
                # Lower entropy = more confident (emotional) output
                probs = torch.softmax(logits[..., -1, :], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                return -entropy.mean()  # Negative because we want low entropy

        # Discover initial circuit
        circuit = self.circuit_finder.find_circuit(
            clean_inputs=clean_inputs,
            corrupted_inputs=neutral_inputs,
            metric_fn=emotion_metric,
            threshold=threshold,
            answer_tokens=answer_tokens,
            circuit_name=f"{emotion_type}_circuit",
            behavior_type="emotion"
        )

        print(f"Initial circuit has {len(circuit.components)} components")

        # Prune to minimal circuit if requested
        if prune and len(circuit.components) > 0:
            print("Pruning circuit to minimal set...")
            circuit = self.circuit_finder.iterative_pruning(
                circuit=circuit,
                clean_inputs=clean_inputs,
                corrupted_inputs=neutral_inputs,
                metric_fn=emotion_metric,
                min_performance=0.85,
                answer_tokens=answer_tokens
            )
            print(f"Pruned circuit has {len(circuit.components)} components")

        # Store circuit
        self.circuits[emotion_type] = circuit

        return circuit

    def validate_emotion_circuit(
        self,
        circuit: Circuit,
        test_clean_prompts: List[str],
        test_neutral_prompts: List[str],
        target_tokens: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Validate an emotion circuit on held-out test data.

        Args:
            circuit: Circuit to validate
            test_clean_prompts: Test prompts expressing emotion
            test_neutral_prompts: Test neutral prompts
            target_tokens: Expected emotion tokens

        Returns:
            Dictionary of validation metrics
        """
        # Prepare test data
        test_clean, test_neutral, answer_tokens = self.prepare_emotion_dataset(
            circuit.behavior_type,
            test_clean_prompts,
            test_neutral_prompts,
            target_tokens
        )

        # Define metric
        def emotion_metric(logits):
            if answer_tokens is not None:
                emotion_logits = logits[..., -1, answer_tokens[0]]
                return emotion_logits.mean()
            else:
                probs = torch.softmax(logits[..., -1, :], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                return -entropy.mean()

        # Validate
        results = self.validator.validate_circuit(
            circuit=circuit,
            test_clean=test_clean,
            test_corrupted=test_neutral,
            metric_fn=emotion_metric
        )

        return results

    def modulate_emotion(
        self,
        circuit: Circuit,
        prompts: List[str],
        intensity: float = 1.0
    ) -> List[str]:
        """
        Control emotion expression by modulating circuit activations.

        This demonstrates the "control" aspect of circuit discovery:
        by amplifying or dampening circuit activations, we can control
        how strongly the model expresses the emotion.

        Args:
            circuit: Emotion circuit to modulate
            prompts: Input prompts
            intensity: Modulation intensity (>1 amplifies, <1 dampens, 0 removes)

        Returns:
            Generated text with modulated emotion
        """
        inputs = self.model.to_tokens(prompts)

        # Create modulation hook that multiplies activations by intensity
        def modulation_hook(activation, hook):
            """Multiply activation by intensity."""
            return activation * intensity

        # Create list of (component_name, hook_function) tuples for all circuit components
        hook_list = [(component, modulation_hook) for component in circuit.components]

        # Generate with modulated circuit using hooks context manager
        # This is the correct way to temporarily add hooks in TransformerLens
        with torch.no_grad():
            with self.model.hooks(fwd_hooks=hook_list):
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.8
                )

        # Decode outputs
        generated_texts = [
            self.model.to_string(output) for output in outputs
        ]

        return generated_texts

    def analyze_circuit_components(self, circuit: Circuit) -> Dict[str, any]:
        """
        Analyze the composition of a circuit.

        Returns statistics about what types of components are in the circuit.

        Args:
            circuit: Circuit to analyze

        Returns:
            Dictionary with component statistics
        """
        stats = {
            "total_components": len(circuit.components),
            "layers": set(),
            "attention_heads": [],
            "mlp_components": [],
            "residual_components": [],
            "attention_outputs": []
        }

        for component in circuit.components:
            # Extract layer
            if "blocks." in component:
                layer_str = component.split("blocks.")[1].split(".")[0]
                try:
                    stats["layers"].add(int(layer_str))
                except ValueError:
                    pass

            # Categorize component type
            if "attn.hook_z" in component or ".hook_z" in component:
                stats["attention_heads"].append(component)
            elif "mlp" in component:
                stats["mlp_components"].append(component)
            elif "resid" in component:
                stats["residual_components"].append(component)
            elif "attn_out" in component:
                stats["attention_outputs"].append(component)

        stats["layers"] = sorted(list(stats["layers"]))
        stats["num_layers"] = len(stats["layers"])
        stats["num_attention_heads"] = len(stats["attention_heads"])
        stats["num_mlp"] = len(stats["mlp_components"])

        return stats

    def compare_emotion_circuits(
        self,
        circuits: Dict[str, Circuit]
    ) -> Dict[str, any]:
        """
        Compare multiple emotion circuits to find commonalities and differences.

        Args:
            circuits: Dictionary mapping emotion names to circuits

        Returns:
            Comparison analysis
        """
        if len(circuits) < 2:
            return {"error": "Need at least 2 circuits to compare"}

        # Find shared components
        all_components = [set(c.components) for c in circuits.values()]
        shared = set.intersection(*all_components)
        unique_per_circuit = {
            name: set(circuit.components) - shared
            for name, circuit in circuits.items()
        }

        # Layer distribution
        layer_usage = {}
        for name, circuit in circuits.items():
            stats = self.analyze_circuit_components(circuit)
            layer_usage[name] = stats["layers"]

        return {
            "num_circuits": len(circuits),
            "shared_components": list(shared),
            "num_shared": len(shared),
            "unique_components": {k: list(v) for k, v in unique_per_circuit.items()},
            "layer_usage": layer_usage,
            "circuit_sizes": {name: len(c.components) for name, c in circuits.items()}
        }
