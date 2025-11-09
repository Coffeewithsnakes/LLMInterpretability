"""
Circuit Validator

Validates discovered circuits and measures their faithfulness.
"""

from typing import Dict, List, Tuple, Callable, Optional
import torch
import numpy as np
from transformer_lens import HookedTransformer
from .circuit_finder import Circuit
from sklearn.metrics import precision_score, recall_score, f1_score


class CircuitValidator:
    """
    Validates circuits and measures their quality.

    Metrics:
    - Precision: What fraction of included components are actually important?
    - Recall: What fraction of important components are included?
    - Faithfulness: How well does the circuit reproduce the behavior?
    - Minimality: Is the circuit minimal (no redundant components)?
    """

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.model.eval()

    def validate_circuit(
        self,
        circuit: Circuit,
        test_clean: torch.Tensor,
        test_corrupted: torch.Tensor,
        metric_fn: Callable,
        ground_truth_components: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Comprehensively validate a circuit.

        Args:
            circuit: Circuit to validate
            test_clean: Test set clean inputs
            test_corrupted: Test set corrupted inputs
            metric_fn: Metric function
            ground_truth_components: Known important components (if available)

        Returns:
            Dictionary of validation metrics
        """
        results = {}

        # 1. Faithfulness: How well does circuit reproduce behavior?
        faithfulness = self.measure_faithfulness(
            circuit, test_clean, test_corrupted, metric_fn
        )
        results["faithfulness"] = faithfulness

        # 2. Completeness: How much behavior is captured?
        completeness = self.measure_completeness(
            circuit, test_clean, test_corrupted, metric_fn
        )
        results["completeness"] = completeness

        # 3. Minimality: Are there redundant components?
        minimality = self.measure_minimality(
            circuit, test_clean, test_corrupted, metric_fn
        )
        results["minimality"] = minimality

        # 4. If ground truth available, compute precision/recall
        if ground_truth_components:
            precision, recall, f1 = self.compare_to_ground_truth(
                circuit, ground_truth_components
            )
            results["precision"] = precision
            results["recall"] = recall
            results["f1"] = f1

        # Update circuit object
        circuit.faithfulness = faithfulness
        circuit.precision = results.get("precision", 0.0)
        circuit.recall = results.get("recall", 0.0)

        return results

    def measure_faithfulness(
        self,
        circuit: Circuit,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        metric_fn: Callable
    ) -> float:
        """
        Measure how faithfully the circuit reproduces the target behavior.

        Faithfulness = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)

        Returns:
            Faithfulness score in [0, 1] (higher is better)
        """
        with torch.no_grad():
            # Get baseline metrics
            clean_output = self.model(clean_inputs)
            clean_metric = metric_fn(clean_output)

            corrupted_output = self.model(corrupted_inputs)
            corrupted_metric = metric_fn(corrupted_output)

            # Get activations
            _, clean_cache = self.model.run_with_cache(clean_inputs)

            # Patch circuit components
            hooks = []
            for component in circuit.components:
                if component in clean_cache:
                    clean_act = clean_cache[component]

                    def make_hook(clean_activation):
                        def hook(activation, hook):
                            return clean_activation
                        return hook

                    hooks.append((component, make_hook(clean_act)))

            # Run with patched circuit
            patched_output = self.model.run_with_hooks(
                corrupted_inputs,
                fwd_hooks=hooks
            )
            patched_metric = metric_fn(patched_output)

        # Compute faithfulness
        metric_range = clean_metric - corrupted_metric
        if abs(metric_range) < 1e-6:
            return 0.0

        faithfulness = (patched_metric - corrupted_metric) / metric_range
        return float(torch.clamp(faithfulness, 0.0, 1.0))

    def measure_completeness(
        self,
        circuit: Circuit,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        metric_fn: Callable
    ) -> float:
        """
        Measure what fraction of the behavior the circuit captures.

        Similar to faithfulness but focuses on how complete the circuit is.

        Returns:
            Completeness score in [0, 1]
        """
        # For now, use faithfulness as a proxy for completeness
        # More sophisticated measures could look at coverage across different examples
        return self.measure_faithfulness(circuit, clean_inputs, corrupted_inputs, metric_fn)

    def measure_minimality(
        self,
        circuit: Circuit,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        metric_fn: Callable,
        performance_threshold: float = 0.95
    ) -> float:
        """
        Measure whether the circuit is minimal (no redundant components).

        For each component, check if removing it significantly hurts performance.
        Minimality = fraction of components that are non-redundant.

        Args:
            circuit: Circuit to evaluate
            clean_inputs: Clean inputs
            corrupted_inputs: Corrupted inputs
            metric_fn: Metric function
            performance_threshold: Threshold for "significant" performance drop

        Returns:
            Minimality score in [0, 1] (higher = more minimal)
        """
        if len(circuit.components) == 0:
            return 1.0

        # Get baseline performance with full circuit
        baseline = self.measure_faithfulness(
            circuit, clean_inputs, corrupted_inputs, metric_fn
        )

        non_redundant = 0

        # Try removing each component
        for component in circuit.components:
            # Create circuit without this component
            reduced_components = [c for c in circuit.components if c != component]

            reduced_circuit = Circuit(
                name=f"{circuit.name}_reduced",
                components=reduced_components,
                effects={k: v for k, v in circuit.effects.items() if k != component},
                behavior_type=circuit.behavior_type
            )

            # Measure performance
            reduced_perf = self.measure_faithfulness(
                reduced_circuit, clean_inputs, corrupted_inputs, metric_fn
            )

            # If removing this component hurts performance, it's non-redundant
            if reduced_perf < performance_threshold * baseline:
                non_redundant += 1

        minimality = non_redundant / len(circuit.components)
        return minimality

    def compare_to_ground_truth(
        self,
        circuit: Circuit,
        ground_truth: List[str]
    ) -> Tuple[float, float, float]:
        """
        Compare discovered circuit to ground truth components.

        Args:
            circuit: Discovered circuit
            ground_truth: List of ground truth component names

        Returns:
            Tuple of (precision, recall, f1)
        """
        discovered = set(circuit.components)
        truth = set(ground_truth)

        if len(discovered) == 0:
            return 0.0, 0.0, 0.0

        # True positives: in both discovered and ground truth
        tp = len(discovered & truth)

        # False positives: in discovered but not ground truth
        fp = len(discovered - truth)

        # False negatives: in ground truth but not discovered
        fn = len(truth - discovered)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def cross_validate_circuit(
        self,
        circuit: Circuit,
        dataset_splits: List[Tuple[torch.Tensor, torch.Tensor]],
        metric_fn: Callable
    ) -> Dict[str, float]:
        """
        Cross-validate circuit on multiple data splits.

        Args:
            circuit: Circuit to validate
            dataset_splits: List of (clean, corrupted) input pairs
            metric_fn: Metric function

        Returns:
            Dictionary with mean and std of metrics across splits
        """
        faithfulness_scores = []

        for clean_inputs, corrupted_inputs in dataset_splits:
            faith = self.measure_faithfulness(
                circuit, clean_inputs, corrupted_inputs, metric_fn
            )
            faithfulness_scores.append(faith)

        return {
            "mean_faithfulness": np.mean(faithfulness_scores),
            "std_faithfulness": np.std(faithfulness_scores),
            "min_faithfulness": np.min(faithfulness_scores),
            "max_faithfulness": np.max(faithfulness_scores)
        }
