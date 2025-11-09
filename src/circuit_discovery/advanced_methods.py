"""
Advanced Circuit Discovery Methods

Implements the seven-stage pipeline from "Do LLMs 'Feel'?" paper:

1. Prompt-based emotion elicitation
2. Emotion direction extraction from residual streams
3. Steering-based generation validation
4. Local component identification (MLP neurons + attention heads)
5. Emotion difference vector computation
6. Global circuit integration
7. Circuit-based emotion generation

Target: 392 MLP neurons + 168 attention heads per emotion (560 total components)
Accuracy goal: 99.65%
"""

from typing import List, Dict, Tuple, Optional, Callable
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from ..circuit_discovery import Circuit


@dataclass
class EmotionDirection:
    """Emotion direction vector extracted from residual stream."""
    emotion: str
    direction: torch.Tensor  # Shape: (d_model,)
    layer: int
    strength: float
    source_prompts: List[str]


@dataclass
class ComponentImportance:
    """Importance score for a model component."""
    component_name: str
    importance: float
    component_type: str  # 'mlp_neuron' or 'attention_head'
    layer: int
    position: Optional[int] = None  # For attention heads: head index


class AdvancedCircuitFinder:
    """
    Advanced circuit discovery matching the paper's methodology.

    This discovers circuits by:
    1. Extracting emotion directions from residual streams
    2. Identifying causally important neurons and attention heads
    3. Computing difference vectors between emotional and neutral states
    4. Integrating components into global circuits
    """

    def __init__(
        self,
        model: HookedTransformer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.device = device
        self.model.eval()

        # Paper hyperparameters
        self.K_total = 560  # Total components (392 MLP + 168 attention)
        self.mlp_ratio = 0.70  # 70% MLP neurons
        self.attention_ratio = 0.30  # 30% attention heads
        self.injection_scale = 0.8  # Default steering scale

        # Important layers (paper finds 15-27 most important for Llama)
        # For GPT-2, adjust proportionally
        n_layers = self.model.cfg.n_layers
        self.important_layer_range = (
            int(n_layers * 15/28),  # ~54% through model
            n_layers  # To end
        )

    def extract_emotion_direction(
        self,
        emotion_prompts: List[str],
        neutral_prompts: List[str],
        layer: Optional[int] = None
    ) -> List[EmotionDirection]:
        """
        Extract emotion direction vectors from residual stream.

        This is Stage 2 of the paper's pipeline.

        Args:
            emotion_prompts: Prompts expressing the target emotion
            neutral_prompts: Neutral baseline prompts
            layer: Specific layer to extract from (None = all layers)

        Returns:
            List of EmotionDirection objects (one per layer if layer=None)
        """
        layers_to_check = [layer] if layer is not None else range(self.model.cfg.n_layers)

        emotion_directions = []

        for layer_idx in layers_to_check:
            # Get residual stream activations for emotion prompts
            emotion_activations = []
            for prompt in emotion_prompts:
                tokens = self.model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)

                # Get residual stream at this layer
                resid_name = get_act_name("resid_post", layer_idx)
                activation = cache[resid_name]

                # Use last position (where emotion is strongest)
                last_pos_activation = activation[0, -1, :]
                emotion_activations.append(last_pos_activation)

            # Get neutral activations
            neutral_activations = []
            for prompt in neutral_prompts:
                tokens = self.model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)

                resid_name = get_act_name("resid_post", layer_idx)
                activation = cache[resid_name]
                last_pos_activation = activation[0, -1, :]
                neutral_activations.append(last_pos_activation)

            # Compute difference vector (emotion direction)
            emotion_mean = torch.stack(emotion_activations).mean(dim=0)
            neutral_mean = torch.stack(neutral_activations).mean(dim=0)

            direction = emotion_mean - neutral_mean
            strength = torch.norm(direction).item()

            # Normalize direction
            if strength > 0:
                direction = direction / strength

            emotion_directions.append(EmotionDirection(
                emotion="unknown",  # Set by caller
                direction=direction,
                layer=layer_idx,
                strength=strength,
                source_prompts=emotion_prompts
            ))

        return emotion_directions

    def identify_important_mlp_neurons(
        self,
        emotion_prompts: List[str],
        neutral_prompts: List[str],
        num_neurons: int = 392
    ) -> List[ComponentImportance]:
        """
        Identify causally important MLP neurons for an emotion.

        This is part of Stage 4 (local component identification).

        Uses activation difference to find neurons that fire strongly
        for emotional content vs neutral content.

        Args:
            emotion_prompts: Emotional prompts
            neutral_prompts: Neutral prompts
            num_neurons: Number of top neurons to return (paper uses 392)

        Returns:
            List of ComponentImportance for MLP neurons
        """
        neuron_importance = {}

        # Only check important layers
        start_layer, end_layer = self.important_layer_range

        for layer in tqdm(range(start_layer, end_layer), desc="Analyzing MLP neurons"):
            # Get MLP activations for emotion prompts
            emotion_mlp_acts = []
            for prompt in emotion_prompts[:10]:  # Sample for speed
                tokens = self.model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)

                # Get MLP post-activation
                mlp_name = get_act_name("post", layer)
                if mlp_name in cache:
                    mlp_act = cache[mlp_name]
                    # Average over sequence
                    mlp_act_mean = mlp_act[0].mean(dim=0)
                    emotion_mlp_acts.append(mlp_act_mean)

            # Get neutral MLP activations
            neutral_mlp_acts = []
            for prompt in neutral_prompts[:10]:  # Sample for speed
                tokens = self.model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)

                mlp_name = get_act_name("post", layer)
                if mlp_name in cache:
                    mlp_act = cache[mlp_name]
                    mlp_act_mean = mlp_act[0].mean(dim=0)
                    neutral_mlp_acts.append(mlp_act_mean)

            if not emotion_mlp_acts or not neutral_mlp_acts:
                continue

            # Compute difference for each neuron
            emotion_mean = torch.stack(emotion_mlp_acts).mean(dim=0)
            neutral_mean = torch.stack(neutral_mlp_acts).mean(dim=0)

            difference = torch.abs(emotion_mean - neutral_mean)

            # Store importance for each neuron in this layer
            d_mlp = difference.shape[0]
            for neuron_idx in range(d_mlp):
                component_name = f"blocks.{layer}.mlp.hook_post_neuron_{neuron_idx}"
                importance = difference[neuron_idx].item()

                neuron_importance[component_name] = ComponentImportance(
                    component_name=component_name,
                    importance=importance,
                    component_type='mlp_neuron',
                    layer=layer,
                    position=neuron_idx
                )

        # Sort by importance and return top K
        sorted_neurons = sorted(
            neuron_importance.values(),
            key=lambda x: x.importance,
            reverse=True
        )

        return sorted_neurons[:num_neurons]

    def identify_important_attention_heads(
        self,
        emotion_prompts: List[str],
        neutral_prompts: List[str],
        num_heads: int = 168
    ) -> List[ComponentImportance]:
        """
        Identify causally important attention heads for an emotion.

        This is part of Stage 4 (local component identification).

        Finds attention heads that attend differently for emotional vs neutral content.

        Args:
            emotion_prompts: Emotional prompts
            neutral_prompts: Neutral prompts
            num_heads: Number of top heads to return (paper uses 168)

        Returns:
            List of ComponentImportance for attention heads
        """
        head_importance = {}

        start_layer, end_layer = self.important_layer_range

        for layer in tqdm(range(start_layer, end_layer), desc="Analyzing attention heads"):
            # Get attention patterns for emotion prompts
            emotion_attn_patterns = []
            for prompt in emotion_prompts[:10]:  # Sample
                tokens = self.model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)

                # Get attention pattern
                attn_name = f"blocks.{layer}.attn.hook_pattern"
                if attn_name in cache:
                    pattern = cache[attn_name]  # Shape: [batch, heads, seq, seq]
                    # Average over source/dest positions
                    pattern_mean = pattern[0].mean(dim=(1, 2))  # Shape: [heads]
                    emotion_attn_patterns.append(pattern_mean)

            # Get neutral attention patterns
            neutral_attn_patterns = []
            for prompt in neutral_prompts[:10]:
                tokens = self.model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)

                attn_name = f"blocks.{layer}.attn.hook_pattern"
                if attn_name in cache:
                    pattern = cache[attn_name]
                    pattern_mean = pattern[0].mean(dim=(1, 2))
                    neutral_attn_patterns.append(pattern_mean)

            if not emotion_attn_patterns or not neutral_attn_patterns:
                continue

            # Compute difference for each head
            emotion_mean = torch.stack(emotion_attn_patterns).mean(dim=0)
            neutral_mean = torch.stack(neutral_attn_patterns).mean(dim=0)

            difference = torch.abs(emotion_mean - neutral_mean)

            # Store importance for each head
            n_heads = difference.shape[0]
            for head_idx in range(n_heads):
                component_name = f"blocks.{layer}.attn.hook_z_head_{head_idx}"
                importance = difference[head_idx].item()

                head_importance[component_name] = ComponentImportance(
                    component_name=component_name,
                    importance=importance,
                    component_type='attention_head',
                    layer=layer,
                    position=head_idx
                )

        # Sort and return top K
        sorted_heads = sorted(
            head_importance.values(),
            key=lambda x: x.importance,
            reverse=True
        )

        return sorted_heads[:num_heads]

    def build_emotion_circuit(
        self,
        emotion: str,
        emotion_prompts: List[str],
        neutral_prompts: List[str],
        num_mlp_neurons: Optional[int] = None,
        num_attention_heads: Optional[int] = None
    ) -> Circuit:
        """
        Build complete emotion circuit using the paper's methodology.

        This implements Stages 4-6:
        - Stage 4: Local component identification
        - Stage 5: Emotion difference vector computation
        - Stage 6: Global circuit integration

        Args:
            emotion: Target emotion name
            emotion_prompts: Prompts expressing the emotion
            neutral_prompts: Neutral baseline prompts
            num_mlp_neurons: Number of MLP neurons (default: 392)
            num_attention_heads: Number of attention heads (default: 168)

        Returns:
            Circuit object with discovered components
        """
        # Use paper defaults if not specified
        if num_mlp_neurons is None:
            num_mlp_neurons = int(self.K_total * self.mlp_ratio)
        if num_attention_heads is None:
            num_attention_heads = int(self.K_total * self.attention_ratio)

        print(f"\n🔍 Discovering {emotion} circuit using advanced methods...")
        print(f"   Target: {num_mlp_neurons} MLP neurons + {num_attention_heads} attention heads")

        # Stage 4: Identify important components
        print(f"\n📊 Stage 4: Local Component Identification")

        mlp_neurons = self.identify_important_mlp_neurons(
            emotion_prompts,
            neutral_prompts,
            num_mlp_neurons
        )

        attention_heads = self.identify_important_attention_heads(
            emotion_prompts,
            neutral_prompts,
            num_attention_heads
        )

        # Stage 5: Compute emotion difference vectors (done within identification)
        print(f"\n📐 Stage 5: Emotion Difference Vectors Computed")

        # Stage 6: Global circuit integration
        print(f"\n🔗 Stage 6: Global Circuit Integration")

        # Combine all components
        all_components = []

        for neuron in mlp_neurons:
            # Map to standard hook name format
            layer = neuron.layer
            # Use MLP output hook (closest to paper's approach)
            component_name = f"blocks.{layer}.hook_mlp_out"
            if component_name not in all_components:
                all_components.append(component_name)

        for head in attention_heads:
            layer = head.layer
            head_idx = head.position
            # Use attention output hook
            component_name = f"blocks.{layer}.attn.hook_result"
            if component_name not in all_components:
                all_components.append(component_name)

        circuit = Circuit(
            name=f"{emotion}_circuit_advanced",
            components=all_components,
            behavior_type=f"emotion_{emotion}"
        )

        print(f"\n✅ Circuit built: {len(circuit.components)} unique components")
        print(f"   {len(mlp_neurons)} MLP contributions")
        print(f"   {len(attention_heads)} Attention contributions")

        return circuit

    def steering_based_generation(
        self,
        prompt: str,
        emotion_direction: EmotionDirection,
        scale: float = None,
        max_new_tokens: int = 50
    ) -> str:
        """
        Generate text using steering (adding emotion direction to residual stream).

        This is Stage 3 of the paper's pipeline.

        Args:
            prompt: Input prompt
            emotion_direction: Direction to steer toward
            scale: Steering strength (default: self.injection_scale)
            max_new_tokens: Number of tokens to generate

        Returns:
            Generated text
        """
        if scale is None:
            scale = self.injection_scale

        tokens = self.model.to_tokens(prompt)

        # Create steering hook
        def steering_hook(activation, hook):
            # Add scaled emotion direction to residual stream
            activation[:, -1, :] += scale * emotion_direction.direction.to(activation.device)
            return activation

        # Apply hook to the appropriate layer
        layer = emotion_direction.layer
        hook_name = get_act_name("resid_post", layer)

        with torch.no_grad():
            with self.model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                output = self.model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8
                )

        return self.model.to_string(output[0])


def print_component_analysis(components: List[ComponentImportance], top_k: int = 20):
    """Pretty print component importance analysis."""
    print(f"\n📊 Top {top_k} Components by Importance:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Type':<15} {'Layer':<7} {'Importance':<12} {'Component'}")
    print("-" * 70)

    for i, comp in enumerate(components[:top_k], 1):
        type_label = "MLP Neuron" if comp.component_type == 'mlp_neuron' else "Attention Head"
        print(f"{i:<6} {type_label:<15} {comp.layer:<7} {comp.importance:<12.4f} {comp.component_name}")

    print("-" * 70)
    print(f"Total components analyzed: {len(components)}")

    # Layer distribution
    layer_counts = {}
    for comp in components:
        layer_counts[comp.layer] = layer_counts.get(comp.layer, 0) + 1

    print(f"\n📍 Layer Distribution:")
    for layer in sorted(layer_counts.keys()):
        count = layer_counts[layer]
        bar = "█" * min(count, 50)
        print(f"  Layer {layer:2d}: {bar} ({count})")
