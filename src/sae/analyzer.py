"""
SAE Analyzer

Analyzes models using Sparse Autoencoders to identify interpretable features.
Integrates with SAELens for pre-trained SAEs.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from transformer_lens import HookedTransformer

try:
    from sae_lens import SAE
    SAE_AVAILABLE = True
except ImportError:
    SAE_AVAILABLE = False
    print("Warning: sae_lens not installed. Some functionality will be limited.")


class SAEAnalyzer:
    """
    Analyzes models using Sparse Autoencoders.

    SAEs decompose neural network activations into sparse, interpretable features.
    This class provides utilities for:
    - Loading pre-trained SAEs
    - Analyzing feature activations
    - Identifying behavior-specific features
    - Visualizing feature importance
    """

    def __init__(
        self,
        model_name: str = "gpt2-small",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device
        )
        self.model.eval()

        # SAE cache
        self.saes: Dict[int, any] = {}  # layer -> SAE

    def load_pretrained_sae(
        self,
        layer: int,
        sae_id: Optional[str] = None
    ) -> any:
        """
        Load a pre-trained SAE for a specific layer.

        Args:
            layer: Layer number to load SAE for
            sae_id: SAE identifier (if None, uses default for model)

        Returns:
            Loaded SAE object
        """
        if not SAE_AVAILABLE:
            raise ImportError("sae_lens not installed. Install with: pip install sae-lens")

        # Load SAE from SAELens
        # This is a simplified version - actual implementation would use SAELens API
        # For now, create placeholder
        print(f"Loading SAE for layer {layer}")
        print("Note: Actual SAE loading requires proper SAELens configuration")

        # Cache the SAE
        # self.saes[layer] = loaded_sae

        return None

    def get_feature_activations(
        self,
        inputs: torch.Tensor,
        layer: int,
        sae: Optional[any] = None
    ) -> torch.Tensor:
        """
        Get SAE feature activations for inputs.

        Args:
            inputs: Input token IDs
            layer: Layer to extract features from
            sae: SAE to use (loads if None)

        Returns:
            Feature activations [batch, pos, n_features]
        """
        if sae is None:
            sae = self.saes.get(layer)
            if sae is None:
                raise ValueError(f"No SAE loaded for layer {layer}")

        # Get model activations
        with torch.no_grad():
            _, cache = self.model.run_with_cache(inputs)

        layer_act_name = f"blocks.{layer}.hook_resid_post"
        activations = cache[layer_act_name]

        # Run through SAE
        if hasattr(sae, 'encode'):
            features = sae.encode(activations)
        else:
            # Placeholder if SAE not properly loaded
            print("Warning: Using random features (SAE not properly loaded)")
            features = torch.randn(
                activations.shape[0],
                activations.shape[1],
                1024,  # typical SAE feature dimension
                device=self.device
            )

        return features

    def find_behavior_features(
        self,
        behavior_examples: List[str],
        neutral_examples: List[str],
        layer: int,
        threshold: float = 0.8,
        top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Find SAE features that activate strongly for a specific behavior.

        Args:
            behavior_examples: Examples exhibiting the behavior
            neutral_examples: Neutral control examples
            layer: Layer to analyze
            threshold: Activation threshold
            top_k: Number of top features to return

        Returns:
            List of (feature_idx, discriminative_score) tuples
        """
        # Tokenize
        behavior_tokens = self.model.to_tokens(behavior_examples)
        neutral_tokens = self.model.to_tokens(neutral_examples)

        # Get feature activations
        sae = self.saes.get(layer)
        if sae is None:
            print(f"Warning: No SAE loaded for layer {layer}")
            return []

        behavior_features = self.get_feature_activations(behavior_tokens, layer, sae)
        neutral_features = self.get_feature_activations(neutral_tokens, layer, sae)

        # Compute discriminative score for each feature
        # Score = mean activation on behavior - mean activation on neutral
        behavior_mean = behavior_features.mean(dim=(0, 1))  # [n_features]
        neutral_mean = neutral_features.mean(dim=(0, 1))

        discriminative_scores = behavior_mean - neutral_mean

        # Get top-k features
        top_k_values, top_k_indices = torch.topk(discriminative_scores, k=min(top_k, len(discriminative_scores)))

        results = [
            (idx.item(), score.item())
            for idx, score in zip(top_k_indices, top_k_values)
        ]

        return results

    def analyze_feature_importance(
        self,
        features: List[int],
        layer: int,
        test_examples: List[str]
    ) -> Dict[int, float]:
        """
        Analyze importance of specific features on test examples.

        Args:
            features: List of feature indices to analyze
            layer: Layer containing the features
            test_examples: Examples to test on

        Returns:
            Dictionary mapping feature indices to importance scores
        """
        test_tokens = self.model.to_tokens(test_examples)
        sae = self.saes.get(layer)

        if sae is None:
            print(f"Warning: No SAE loaded for layer {layer}")
            return {}

        feature_acts = self.get_feature_activations(test_tokens, layer, sae)

        importance = {}
        for feat_idx in features:
            # Mean activation across all positions and examples
            mean_act = feature_acts[..., feat_idx].mean().item()
            importance[feat_idx] = mean_act

        return importance

    def get_feature_dashboard_data(
        self,
        feature_idx: int,
        layer: int,
        max_examples: int = 10
    ) -> Dict[str, any]:
        """
        Get data for visualizing a specific feature.

        This would integrate with SAE-Vis for visualization.

        Args:
            feature_idx: Feature index
            layer: Layer number
            max_examples: Maximum examples to include

        Returns:
            Dashboard data dictionary
        """
        dashboard_data = {
            "feature_idx": feature_idx,
            "layer": layer,
            "max_activating_examples": [],
            "feature_description": f"Feature {feature_idx} in layer {layer}",
            "activation_stats": {}
        }

        # This would be populated with actual SAE-Vis integration
        print(f"Feature dashboard data for feature {feature_idx}, layer {layer}")
        print("Note: Full integration with SAE-Vis requires additional setup")

        return dashboard_data

    def compare_features_across_behaviors(
        self,
        behaviors: Dict[str, List[str]],
        layer: int,
        top_k: int = 10
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Compare which features activate for different behaviors.

        Args:
            behaviors: Dict mapping behavior names to example lists
            layer: Layer to analyze
            top_k: Number of top features per behavior

        Returns:
            Dict mapping behavior names to top features
        """
        results = {}

        # Get neutral examples (use first behavior's examples as baseline)
        baseline_behavior = list(behaviors.keys())[0]
        baseline_examples = behaviors[baseline_behavior]

        for behavior_name, examples in behaviors.items():
            if behavior_name == baseline_behavior:
                continue

            top_features = self.find_behavior_features(
                behavior_examples=examples,
                neutral_examples=baseline_examples,
                layer=layer,
                top_k=top_k
            )

            results[behavior_name] = top_features

        return results
