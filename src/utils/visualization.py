"""
Visualization utilities for circuits and interpretability analysis.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_circuit(
    circuit_components: List[str],
    effects: Dict[str, float],
    title: str = "Circuit Components",
    figsize: Tuple[int, int] = (12, 6),
    top_k: int = 20
):
    """
    Visualize circuit components and their effects.

    Args:
        circuit_components: List of component names
        effects: Dictionary mapping components to effect sizes
        title: Plot title
        figsize: Figure size
        top_k: Number of top components to show
    """
    # Sort by effect size
    sorted_components = sorted(
        [(comp, effects.get(comp, 0.0)) for comp in circuit_components],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    components, effect_values = zip(*sorted_components) if sorted_components else ([], [])

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(components))
    colors = ['red' if e < 0 else 'blue' for e in effect_values]

    ax.barh(y_pos, effect_values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(components, fontsize=8)
    ax.set_xlabel('Effect Size')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_patching_results(
    patching_results: Dict[str, float],
    title: str = "Activation Patching Results",
    figsize: Tuple[int, int] = (14, 8),
    threshold: float = 0.5
):
    """
    Visualize activation patching results as a heatmap.

    Args:
        patching_results: Dictionary mapping component names to effects
        title: Plot title
        figsize: Figure size
        threshold: Threshold for highlighting important components
    """
    # Parse component names to extract layer and type
    layer_data = {}

    for component, effect in patching_results.items():
        # Extract layer number
        if "blocks." in component:
            layer_str = component.split("blocks.")[1].split(".")[0]
            try:
                layer = int(layer_str)
            except ValueError:
                continue

            # Determine component type
            if "attn" in component:
                comp_type = "attention"
            elif "mlp" in component:
                comp_type = "mlp"
            elif "resid" in component:
                comp_type = "residual"
            else:
                comp_type = "other"

            if layer not in layer_data:
                layer_data[layer] = {}

            layer_data[layer][comp_type] = effect

    if not layer_data:
        print("No valid layer data to plot")
        return None

    # Create heatmap
    layers = sorted(layer_data.keys())
    comp_types = ["attention", "mlp", "residual"]

    matrix = np.zeros((len(comp_types), len(layers)))

    for i, comp_type in enumerate(comp_types):
        for j, layer in enumerate(layers):
            if comp_type in layer_data[layer]:
                matrix[i, j] = layer_data[layer][comp_type]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        xticklabels=layers,
        yticklabels=comp_types,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={'label': 'Effect Size'},
        ax=ax
    )

    ax.set_xlabel('Layer')
    ax.set_ylabel('Component Type')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_layer_distribution(
    circuits: Dict[str, List[str]],
    model_num_layers: int = 12,
    title: str = "Circuit Layer Distribution"
):
    """
    Plot distribution of circuit components across layers.

    Args:
        circuits: Dictionary mapping circuit names to component lists
        model_num_layers: Total number of layers in model
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for circuit_name, components in circuits.items():
        layer_counts = [0] * model_num_layers

        for component in components:
            if "blocks." in component:
                layer_str = component.split("blocks.")[1].split(".")[0]
                try:
                    layer = int(layer_str)
                    if 0 <= layer < model_num_layers:
                        layer_counts[layer] += 1
                except ValueError:
                    continue

        ax.plot(range(model_num_layers), layer_counts, marker='o', label=circuit_name)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Components')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
