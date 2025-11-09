"""
Feature Extraction

Tools for extracting and analyzing features from SAEs.
"""

from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
from collections import defaultdict


class FeatureExtractor:
    """
    Extracts and analyzes features from sparse autoencoders.

    Provides utilities for:
    - Feature activation patterns
    - Feature co-occurrence
    - Feature steering
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def extract_top_activating_features(
        self,
        feature_activations: torch.Tensor,
        top_k: int = 10,
        per_position: bool = False
    ) -> List[Tuple[int, float]]:
        """
        Extract features with highest activations.

        Args:
            feature_activations: Tensor of shape [batch, pos, n_features]
            top_k: Number of top features
            per_position: Whether to find top features per position

        Returns:
            List of (feature_idx, activation) tuples
        """
        if per_position:
            # Analyze each position separately
            # Average across batch dimension
            pos_features = feature_activations.mean(dim=0)  # [pos, n_features]

            results = []
            for pos in range(pos_features.shape[0]):
                top_vals, top_idx = torch.topk(pos_features[pos], k=top_k)
                results.append([
                    (idx.item(), val.item())
                    for idx, val in zip(top_idx, top_vals)
                ])
            return results
        else:
            # Global top features
            mean_activations = feature_activations.mean(dim=(0, 1))  # [n_features]
            top_vals, top_idx = torch.topk(mean_activations, k=top_k)

            return [
                (idx.item(), val.item())
                for idx, val in zip(top_idx, top_vals)
            ]

    def compute_feature_cooccurrence(
        self,
        feature_activations: torch.Tensor,
        threshold: float = 0.1
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute which features co-occur (activate together).

        Args:
            feature_activations: Tensor of shape [batch, pos, n_features]
            threshold: Activation threshold for considering feature "active"

        Returns:
            Dictionary mapping (feature_i, feature_j) to co-occurrence score
        """
        # Binarize activations
        active = (feature_activations > threshold).float()

        # Flatten batch and position
        active_flat = active.reshape(-1, active.shape[-1])  # [batch*pos, n_features]

        # Compute co-occurrence matrix
        cooccur = torch.mm(active_flat.T, active_flat)  # [n_features, n_features]

        # Normalize by number of activations
        feature_counts = active_flat.sum(dim=0, keepdim=True).T
        cooccur_normalized = cooccur / (feature_counts + 1e-8)

        # Extract top co-occurrences
        cooccur_dict = {}
        n_features = cooccur.shape[0]

        for i in range(n_features):
            for j in range(i + 1, n_features):
                if cooccur_normalized[i, j] > 0.1:  # Only store significant co-occurrences
                    cooccur_dict[(i, j)] = cooccur_normalized[i, j].item()

        return cooccur_dict

    def create_feature_steering_vector(
        self,
        target_features: List[int],
        feature_dim: int,
        amplification: float = 2.0
    ) -> torch.Tensor:
        """
        Create a steering vector to amplify specific features.

        Args:
            target_features: List of feature indices to amplify
            feature_dim: Total number of features
            amplification: How much to amplify (1.0 = no change)

        Returns:
            Steering vector
        """
        steering_vector = torch.ones(feature_dim, device=self.device)

        for feat_idx in target_features:
            steering_vector[feat_idx] = amplification

        return steering_vector

    def analyze_feature_sparsity(
        self,
        feature_activations: torch.Tensor,
        threshold: float = 0.01
    ) -> Dict[str, float]:
        """
        Analyze sparsity of feature activations.

        Args:
            feature_activations: Feature activations
            threshold: Threshold for considering feature active

        Returns:
            Sparsity statistics
        """
        # Fraction of features active
        active = (feature_activations > threshold).float()
        active_fraction = active.mean().item()

        # L0 norm (number of active features per example)
        l0 = (feature_activations > threshold).sum(dim=-1).float().mean().item()

        # L1 norm
        l1 = feature_activations.abs().mean().item()

        # L2 norm
        l2 = torch.norm(feature_activations, dim=-1).mean().item()

        return {
            "active_fraction": active_fraction,
            "mean_l0": l0,
            "mean_l1": l1,
            "mean_l2": l2
        }

    def find_monosemantic_features(
        self,
        feature_activations: torch.Tensor,
        tokens: torch.Tensor,
        vocab_size: int,
        min_examples: int = 5
    ) -> Dict[int, List[int]]:
        """
        Attempt to find monosemantic features (features that activate for specific tokens).

        Args:
            feature_activations: Feature activations [batch, pos, n_features]
            tokens: Token IDs [batch, pos]
            vocab_size: Size of vocabulary
            min_examples: Minimum examples to consider pattern

        Returns:
            Dict mapping feature indices to associated token IDs
        """
        n_features = feature_activations.shape[-1]
        feature_token_map = defaultdict(lambda: defaultdict(int))

        # Flatten
        acts_flat = feature_activations.reshape(-1, n_features)
        tokens_flat = tokens.reshape(-1)

        # For each position, record which tokens co-occur with which features
        for pos_idx in range(len(tokens_flat)):
            token_id = tokens_flat[pos_idx].item()
            acts = acts_flat[pos_idx]

            # Find active features
            active_features = torch.where(acts > 0.1)[0]

            for feat_idx in active_features:
                feature_token_map[feat_idx.item()][token_id] += 1

        # Find features that consistently activate for specific tokens
        monosemantic = {}

        for feat_idx, token_counts in feature_token_map.items():
            # Get most common tokens for this feature
            sorted_tokens = sorted(
                token_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            if sorted_tokens and sorted_tokens[0][1] >= min_examples:
                # Check if this is actually monosemantic (one token dominates)
                total_count = sum(token_counts.values())
                top_token_ratio = sorted_tokens[0][1] / total_count

                if top_token_ratio > 0.5:  # Token appears in >50% of activations
                    monosemantic[feat_idx] = [t for t, c in sorted_tokens[:3]]

        return monosemantic
