"""
Sparse Autoencoder (SAE) Module

Tools for training and analyzing sparse autoencoders for interpretability.
"""

from .analyzer import SAEAnalyzer
from .feature_extraction import FeatureExtractor

__all__ = [
    "SAEAnalyzer",
    "FeatureExtractor",
]
