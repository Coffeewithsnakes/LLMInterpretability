"""
Circuit Finder

Discovers minimal computational circuits responsible for specific behaviors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable
import torch
import numpy as np
from transformer_lens import HookedTransformer
from .activation_patching import ActivationPatcher, PatchingConfig


@dataclass
class Circuit:
    """Represents a discovered circuit in the model."""

    name: str
    components: List[str]
    effects: Dict[str, float]
    precision: float = 0.0
    recall: float = 0.0
    faithfulness: float = 0.0

    # Metadata
    behavior_type: str = ""
    discovery_method: str = ""
    metadata: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.components)

    def __contains__(self, component: str) -> bool:
        return component in self.components

    def get_layer_components(self, layer: int) -> List[str]:
        """Get all components in a specific layer."""
        return [c for c in self.components if f".{layer}." in c or f".{layer}" in c]

    def get_attention_heads(self) -> List[Tuple[int, int]]:
        """Extract (layer, head) tuples for attention heads in circuit."""
        heads = []
        for component in self.components:
            if ".attn.hook_z" in component or ".hook_z" in component:
                # Parse layer and head from component name
                parts = component.split(".")
                for i, part in enumerate(parts):
                    if "attn" in part and i + 1 < len(parts):
                        try:
                            layer = int(parts[i - 1].replace("blocks", ""))
                            head = int(parts[-1].replace("hook_z", ""))
                            heads.append((layer, head))
                        except (ValueError, IndexError):
                            continue
        return heads

    def summary(self) -> str:
        """Get a human-readable summary of the circuit."""
        lines = [
            f"Circuit: {self.name}",
            f"Behavior: {self.behavior_type}",
            f"Components: {len(self.components)}",
            f"Precision: {self.precision:.2%}",
            f"Recall: {self.recall:.2%}",
            f"Faithfulness: {self.faithfulness:.2%}",
            f"Method: {self.discovery_method}",
        ]

        if self.get_attention_heads():
            lines.append(f"Attention heads: {len(self.get_attention_heads())}")

        return "\n".join(lines)


class CircuitFinder:
    """
    Discovers circuits in language models using activation patching.

    This class implements the circuit discovery pipeline:
    1. Identify candidate components via activation patching
    2. Prune to minimal sufficient circuit
    3. Validate circuit performance
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
            cache_dir=cache_dir
        )
        self.model.eval()

        # Initialize patcher
        config = PatchingConfig(model_name=model_name, device=self.device)
        self.patcher = ActivationPatcher(config)

    def find_circuit(
        self,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        metric_fn: Callable,
        threshold: float = 0.5,
        answer_tokens: Optional[torch.Tensor] = None,
        circuit_name: str = "discovered_circuit",
        behavior_type: str = "unknown"
    ) -> Circuit:
        """
        Discover a circuit for a specific behavior.

        Args:
            clean_inputs: Input examples that produce the target behavior
            corrupted_inputs: Corrupted inputs that don't produce the behavior
            metric_fn: Function to measure behavior strength
            threshold: Minimum effect size for component inclusion
            answer_tokens: Expected answer tokens (if using default metric)
            circuit_name: Name for the discovered circuit
            behavior_type: Type of behavior (e.g., "emotion", "deception")

        Returns:
            Discovered Circuit object
        """
        # Find important components via activation patching
        important_components = self.patcher.find_important_components(
            clean_inputs=clean_inputs,
            corrupted_inputs=corrupted_inputs,
            threshold=threshold,
            metric_fn=metric_fn,
            answer_tokens=answer_tokens
        )

        if not important_components:
            print("Warning: No important components found above threshold")
            return Circuit(
                name=circuit_name,
                components=[],
                effects={},
                behavior_type=behavior_type,
                discovery_method="activation_patching"
            )

        # Create circuit
        circuit = Circuit(
            name=circuit_name,
            components=list(important_components.keys()),
            effects=important_components,
            behavior_type=behavior_type,
            discovery_method="activation_patching"
        )

        return circuit

    def iterative_pruning(
        self,
        circuit: Circuit,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        metric_fn: Callable,
        min_performance: float = 0.9,
        answer_tokens: Optional[torch.Tensor] = None
    ) -> Circuit:
        """
        Prune circuit to minimal set while maintaining performance.

        Uses an iterative algorithm:
        1. Rank components by effect size
        2. Try removing least important component
        3. If performance stays above threshold, keep removal
        4. Repeat until performance drops

        Args:
            circuit: Initial circuit to prune
            clean_inputs: Clean examples
            corrupted_inputs: Corrupted examples
            metric_fn: Metric function
            min_performance: Minimum performance to maintain (as fraction of original)
            answer_tokens: Answer tokens for metric

        Returns:
            Pruned circuit
        """
        # Get baseline performance with full circuit
        baseline_perf = self._evaluate_circuit_performance(
            circuit, clean_inputs, corrupted_inputs, metric_fn, answer_tokens
        )

        # Sort components by effect (least important first)
        sorted_components = sorted(
            circuit.components,
            key=lambda c: abs(circuit.effects.get(c, 0))
        )

        pruned_components = circuit.components.copy()
        pruned_effects = circuit.effects.copy()

        # Try removing each component
        for component in sorted_components:
            # Try without this component
            test_components = [c for c in pruned_components if c != component]

            if not test_components:
                break  # Don't remove last component

            test_circuit = Circuit(
                name=f"{circuit.name}_pruned",
                components=test_components,
                effects={k: v for k, v in pruned_effects.items() if k in test_components},
                behavior_type=circuit.behavior_type,
                discovery_method="pruning"
            )

            # Evaluate performance
            perf = self._evaluate_circuit_performance(
                test_circuit, clean_inputs, corrupted_inputs, metric_fn, answer_tokens
            )

            # If performance is still good, keep the removal
            if perf >= min_performance * baseline_perf:
                pruned_components = test_components
                del pruned_effects[component]
                print(f"Removed {component}, performance: {perf:.3f}")
            else:
                print(f"Keeping {component}, performance would drop to {perf:.3f}")

        # Create pruned circuit
        pruned_circuit = Circuit(
            name=f"{circuit.name}_pruned",
            components=pruned_components,
            effects=pruned_effects,
            behavior_type=circuit.behavior_type,
            discovery_method=f"{circuit.discovery_method} + pruning"
        )

        return pruned_circuit

    def _evaluate_circuit_performance(
        self,
        circuit: Circuit,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        metric_fn: Callable,
        answer_tokens: Optional[torch.Tensor] = None
    ) -> float:
        """
        Evaluate how well a circuit recovers the target behavior.

        Returns:
            Performance score (higher is better)
        """
        if not circuit.components:
            return 0.0

        # Get activations
        clean_cache, corrupted_cache = self.patcher.get_clean_corrupted_activations(
            clean_inputs, corrupted_inputs
        )

        # Patch all circuit components
        def multi_patch_hook(activation, hook, component_name):
            if component_name in circuit.components:
                return clean_cache[component_name]
            return activation

        hooks = [
            (comp, lambda act, hook, c=comp: multi_patch_hook(act, hook, c))
            for comp in circuit.components
        ]

        with torch.no_grad():
            patched_output = self.model.run_with_hooks(
                corrupted_inputs,
                fwd_hooks=hooks
            )

        # Compute metric
        performance = metric_fn(patched_output)

        return performance.item() if torch.is_tensor(performance) else performance

    def find_circuits_for_dataset(
        self,
        dataset: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        metric_fn: Callable,
        threshold: float = 0.5,
        behavior_type: str = "unknown",
        average_across_examples: bool = True
    ) -> Circuit:
        """
        Discover circuit across multiple examples.

        Args:
            dataset: List of (clean_input, corrupted_input, answer_tokens) tuples
            metric_fn: Metric function
            threshold: Component inclusion threshold
            behavior_type: Type of behavior
            average_across_examples: Average effects across examples vs. union

        Returns:
            Circuit discovered across dataset
        """
        if average_across_examples:
            # Average patching effects across all examples
            all_effects = []

            for clean_inp, corr_inp, ans_tok in dataset:
                effects = self.patcher.compute_patching_effect(
                    clean_inp, corr_inp, metric_fn, ans_tok
                )
                all_effects.append(effects)

            # Average effects
            avg_effects = {}
            all_components = set()
            for effects in all_effects:
                all_components.update(effects.keys())

            for component in all_components:
                values = [e.get(component, 0.0) for e in all_effects]
                avg_effects[component] = np.mean(values)

            # Filter by threshold
            important = {
                k: v for k, v in avg_effects.items()
                if abs(v) >= threshold
            }

            circuit = Circuit(
                name=f"{behavior_type}_circuit",
                components=list(important.keys()),
                effects=important,
                behavior_type=behavior_type,
                discovery_method="averaged_patching"
            )

        else:
            # Take union of important components
            all_components = set()
            all_effects = {}

            for clean_inp, corr_inp, ans_tok in dataset:
                important = self.patcher.find_important_components(
                    clean_inp, corr_inp, threshold, metric_fn=metric_fn, answer_tokens=ans_tok
                )
                all_components.update(important.keys())
                all_effects.update(important)

            circuit = Circuit(
                name=f"{behavior_type}_circuit",
                components=list(all_components),
                effects=all_effects,
                behavior_type=behavior_type,
                discovery_method="union_patching"
            )

        return circuit
