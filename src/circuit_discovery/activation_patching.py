"""
Activation Patching Implementation

Implements causal intervention techniques to identify important model components.
Based on the methodology from TransformerLens and recent mechanistic interpretability research.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from tqdm import tqdm
import numpy as np


@dataclass
class PatchingConfig:
    """Configuration for activation patching experiments."""

    model_name: str = "gpt2-small"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Patching parameters
    patch_layers: Optional[List[int]] = None  # None means all layers
    patch_positions: Optional[List[int]] = None  # None means all positions

    # Component types to patch
    patch_residual: bool = True
    patch_attn_out: bool = True
    patch_mlp_out: bool = True
    patch_attn_heads: bool = True  # Patch individual attention heads

    # Metric configuration
    metric_fn: Optional[Callable] = None  # Custom metric function

    # Performance
    batch_size: int = 8
    use_cache: bool = True


class ActivationPatcher:
    """
    Implements activation patching for causal analysis of transformer models.

    Activation patching works by:
    1. Running the model on a "clean" input to get clean activations
    2. Running the model on a "corrupted" input to get corrupted activations
    3. Patching the corrupted run with clean activations at specific components
    4. Measuring how much this restores the clean behavior

    Components that restore behavior significantly are causally important.
    """

    def __init__(self, config: PatchingConfig):
        self.config = config
        self.model = HookedTransformer.from_pretrained(
            config.model_name,
            device=config.device
        )
        self.model.eval()

    def get_clean_corrupted_activations(
        self,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        cache_acts: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get activations for both clean and corrupted inputs.

        Args:
            clean_inputs: Input IDs for clean examples
            corrupted_inputs: Input IDs for corrupted examples
            cache_acts: Whether to cache all activations

        Returns:
            Tuple of (clean_cache, corrupted_cache) dictionaries
        """
        with torch.no_grad():
            # Get clean activations
            _, clean_cache = self.model.run_with_cache(clean_inputs)

            # Get corrupted activations
            _, corrupted_cache = self.model.run_with_cache(corrupted_inputs)

        return clean_cache, corrupted_cache

    def patch_activation(
        self,
        corrupted_inputs: torch.Tensor,
        clean_activation: torch.Tensor,
        hook_name: str,
        position: Optional[int] = None
    ) -> torch.Tensor:
        """
        Patch a specific activation and return model output.

        Args:
            corrupted_inputs: The corrupted input to patch
            clean_activation: The clean activation to patch in
            hook_name: Name of the activation to patch
            position: Optional position to patch (None = all positions)

        Returns:
            Model output after patching
        """
        def patch_hook(activation, hook):
            """Hook function that patches the activation."""
            if position is not None:
                # Patch specific position
                if position < activation.shape[1] and position < clean_activation.shape[1]:
                    activation[:, position, :] = clean_activation[:, position, :]
            else:
                # Patch all positions - handle different sequence lengths
                min_seq_len = min(activation.shape[1], clean_activation.shape[1])
                activation[:, :min_seq_len, :] = clean_activation[:, :min_seq_len, :]
            return activation

        # Run model with patching hook
        with torch.no_grad():
            patched_output = self.model.run_with_hooks(
                corrupted_inputs,
                fwd_hooks=[(hook_name, patch_hook)]
            )

        return patched_output

    def compute_patching_effect(
        self,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        metric_fn: Optional[Callable] = None,
        answer_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the effect of patching each component.

        Args:
            clean_inputs: Clean input examples
            corrupted_inputs: Corrupted input examples
            metric_fn: Function to compute metric (default: logit diff)
            answer_tokens: Expected answer tokens for metric calculation

        Returns:
            Dictionary mapping component names to patching effects
        """
        # Get clean and corrupted activations
        clean_cache, corrupted_cache = self.get_clean_corrupted_activations(
            clean_inputs, corrupted_inputs
        )

        # Get baseline metrics
        with torch.no_grad():
            clean_output = self.model(clean_inputs)
            corrupted_output = self.model(corrupted_inputs)

        # Default metric: logit difference
        if metric_fn is None:
            if answer_tokens is None:
                raise ValueError("Must provide answer_tokens if not using custom metric_fn")

            def default_metric(logits):
                # Compute difference between correct and incorrect answer logits
                correct_logits = logits[range(len(answer_tokens)), -1, answer_tokens[:, 0]]
                incorrect_logits = logits[range(len(answer_tokens)), -1, answer_tokens[:, 1]]
                return (correct_logits - incorrect_logits).mean()

            metric_fn = default_metric

        clean_metric = metric_fn(clean_output)
        corrupted_metric = metric_fn(corrupted_output)

        # Components to patch
        components_to_patch = self._get_components_to_patch()

        patching_effects = {}

        # Patch each component and measure effect
        for component_name in tqdm(components_to_patch, desc="Patching components"):
            if component_name not in clean_cache or component_name not in corrupted_cache:
                continue

            clean_act = clean_cache[component_name]

            # Patch this component
            patched_output = self.patch_activation(
                corrupted_inputs,
                clean_act,
                component_name
            )

            patched_metric = metric_fn(patched_output)

            # Effect is how much patching restored the clean behavior
            # Normalized by the total difference
            if abs(clean_metric - corrupted_metric) > 1e-6:
                effect = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)
            else:
                effect = 0.0

            # Handle both tensor and float returns from metric functions
            if isinstance(effect, torch.Tensor):
                patching_effects[component_name] = effect.item()
            else:
                patching_effects[component_name] = float(effect)

        return patching_effects

    def _get_components_to_patch(self) -> List[str]:
        """Get list of component names to patch based on config."""
        components = []

        layers = (self.config.patch_layers if self.config.patch_layers is not None
                 else range(self.model.cfg.n_layers))

        for layer in layers:
            if self.config.patch_residual:
                components.append(get_act_name("resid_post", layer))

            if self.config.patch_attn_out:
                components.append(get_act_name("attn_out", layer))

            if self.config.patch_mlp_out:
                components.append(get_act_name("mlp_out", layer))

            if self.config.patch_attn_heads:
                for head in range(self.model.cfg.n_heads):
                    components.append(get_act_name("z", layer, head))

        return components

    def find_important_components(
        self,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, float]:
        """
        Find components with patching effect above threshold.

        Args:
            clean_inputs: Clean input examples
            corrupted_inputs: Corrupted input examples
            threshold: Minimum effect size to be considered important
            **kwargs: Additional arguments for compute_patching_effect

        Returns:
            Dictionary of important components and their effects
        """
        effects = self.compute_patching_effect(
            clean_inputs, corrupted_inputs, **kwargs
        )

        # Filter by threshold
        important = {
            name: effect for name, effect in effects.items()
            if abs(effect) >= threshold
        }

        # Sort by absolute effect size
        important = dict(sorted(
            important.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))

        return important

    def attribution_patching(
        self,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        metric_fn: Callable,
        num_samples: int = 10
    ) -> Dict[str, float]:
        """
        Perform attribution patching (faster approximate version).

        Attribution patching uses gradients to approximate the effect of patching,
        which is much faster than full activation patching.

        Args:
            clean_inputs: Clean input examples
            corrupted_inputs: Corrupted input examples
            metric_fn: Metric function to optimize
            num_samples: Number of random samples for approximation

        Returns:
            Dictionary of component attributions
        """
        # Get activations
        clean_cache, corrupted_cache = self.get_clean_corrupted_activations(
            clean_inputs, corrupted_inputs
        )

        attributions = {}
        components = self._get_components_to_patch()

        for component_name in tqdm(components, desc="Computing attributions"):
            if component_name not in clean_cache:
                continue

            clean_act = clean_cache[component_name]
            corrupted_act = corrupted_cache[component_name]

            # Compute gradient of metric w.r.t. this activation
            corrupted_act_grad = corrupted_act.clone().requires_grad_(True)

            def temp_hook(activation, hook):
                return corrupted_act_grad

            # Forward pass with gradient
            output = self.model.run_with_hooks(
                corrupted_inputs,
                fwd_hooks=[(component_name, temp_hook)]
            )

            metric = metric_fn(output)
            metric.backward()

            # Attribution is gradient dot (clean - corrupted)
            if corrupted_act_grad.grad is not None:
                attribution = (
                    corrupted_act_grad.grad * (clean_act - corrupted_act)
                ).sum().item()

                attributions[component_name] = attribution

        return attributions
