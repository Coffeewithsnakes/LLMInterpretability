# LLM Interpretability Toolkit

> **🚀 NEW USER? START HERE!** → [**START_HERE.md**](START_HERE.md) for step-by-step beginner guide!
>
> **⚡ Quick Setup:** Run `python setup_interactive.py` and follow the prompts!
>
> **🔬 RESEARCHERS:** See [**RESEARCH_FEATURES.md**](RESEARCH_FEATURES.md) for advanced features & paper reproduction!

A comprehensive toolkit for mechanistic interpretability of large language models, with a focus on circuit discovery, behavior detection (deceptive alignment, power-seeking), and model steering.

## 🎉 NEW: Research-Grade Features

**Reproduce "Do LLMs 'Feel'?" Paper Results (99.65% accuracy)**

### Interactive Menu (Easiest!)

```bash
python run_interactive.py
# Select: "🔬 Research: Reproduce Paper Results"
```

**OR use command line:**

```bash
# Single emotion reproduction
python reproduce_paper.py --emotion happiness --model gpt2-medium

# Full paper reproduction (all 6 emotions)
python reproduce_paper.py --all --model EleutherAI/pythia-1b
```

**Features:**
- ✅ 600+ high-quality emotion prompts (6 emotions × 100 prompts)
- ✅ Seven-stage pipeline matching paper methodology
- ✅ 99%+ accuracy validation metrics
- ✅ Advanced circuit discovery (392 MLP + 168 attention components)
- ✅ Comprehensive benchmarking against paper
- ✅ Support for GPT-2, Pythia, Llama, and other models
- ✅ **NEW:** Interactive menu - no command line needed!

**See [RESEARCH_FEATURES.md](RESEARCH_FEATURES.md) for complete documentation.**

## Overview

This project implements state-of-the-art interpretability techniques, including:

- **Circuit Discovery**: Identify and validate computational subgraphs responsible for specific behaviors
- **Activation Patching**: Causal interventions to understand which model components matter
- **Sparse Autoencoders (SAEs)**: Decompose activations into interpretable features
- **Behavior Detection**: Tools for detecting deceptive alignment, power-seeking, and other concerning behaviors
- **Model Steering**: Techniques to control and modify model behavior

### Methodology

This toolkit is based on the circuit discovery methodology from recent research (e.g., "Do LLMs 'Feel'? Emotion Circuits Discovery and Control"), adapted for detecting and analyzing various AI behaviors.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LLMInterpretability.git
cd LLMInterpretability

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

## Quick Start

### 1. Circuit Discovery for Emotions

```python
from src.circuit_discovery import CircuitDiscovery
from src.data import load_emotion_dataset

# Load model and data
discovery = CircuitDiscovery(model_name="gpt2-small")
dataset = load_emotion_dataset()

# Discover emotion circuits
circuits = discovery.find_circuits(
    dataset=dataset,
    target_behavior="happiness",
    method="activation_patching"
)

# Validate circuits
validation = discovery.validate_circuits(circuits)
print(f"Circuit precision: {validation['precision']:.2%}")
```

### 2. Detecting Deceptive Alignment

```python
from src.behavior_detection import DeceptiveAlignmentDetector

detector = DeceptiveAlignmentDetector(model_name="llama-2-7b")

# Analyze model for deceptive reasoning
results = detector.analyze(
    prompts=prompts,
    threshold=0.7
)

# Get suspicious circuits
suspicious = detector.get_deceptive_circuits()
```

### 3. Using Sparse Autoencoders

```python
from src.sae import SAEAnalyzer

analyzer = SAEAnalyzer(model_name="gpt2-small")

# Train or load SAE
sae = analyzer.load_pretrained_sae(layer=8)

# Analyze features for power-seeking behavior
power_features = analyzer.find_behavior_features(
    behavior_type="power_seeking",
    threshold=0.8
)
```

## Project Structure

```
LLMInterpretability/
├── src/
│   ├── circuit_discovery/      # Core circuit discovery algorithms
│   │   ├── activation_patching.py
│   │   ├── circuit_finder.py
│   │   └── validator.py
│   ├── sae/                     # Sparse autoencoder utilities
│   │   ├── trainer.py
│   │   ├── analyzer.py
│   │   └── feature_extraction.py
│   ├── behavior_detection/      # Behavior-specific detectors
│   │   ├── deceptive_alignment.py
│   │   ├── power_seeking.py
│   │   └── emotion_circuits.py
│   ├── intervention/            # Model steering and control
│   │   ├── steering_vectors.py
│   │   └── circuit_modulation.py
│   ├── data/                    # Dataset loaders
│   └── utils/                   # Shared utilities
├── notebooks/                   # Jupyter notebooks with examples
│   ├── 01_emotion_circuits.ipynb
│   ├── 02_deceptive_alignment.ipynb
│   └── 03_power_seeking_detection.ipynb
├── experiments/                 # Experiment scripts
└── tests/                       # Unit tests
```

## Key Concepts

### Circuit Discovery Pipeline

1. **Component Identification**: Find neurons and attention heads that activate for target behaviors
2. **Causal Validation**: Use activation patching to verify causal importance
3. **Circuit Refinement**: Prune to minimal sufficient circuit
4. **Evaluation**: Measure precision, recall, and faithfulness

### Behavior Detection

The toolkit can detect various behaviors including:

- **Deceptive Alignment**: When models appear aligned during training but pursue hidden objectives
- **Power-Seeking**: Tendencies to acquire resources or influence
- **Emotion Circuits**: Computational pathways for emotional processing
- **Goal-Directed Reasoning**: Circuits that implement specific objectives

### Activation Patching

Activation patching is a causal intervention technique:

1. Run model on "clean" input → get clean activations
2. Run model on "corrupted" input → get corrupted activations
3. Patch corrupted run with clean activations at specific components
4. Measure how much this restores clean behavior
5. Important components restore behavior significantly

## Examples

See the `notebooks/` directory for detailed tutorials:

- **Emotion Circuits**: Reproduce the emotion circuit discovery methodology
- **Deceptive Alignment**: Detect when models exhibit deceptive reasoning
- **Power-Seeking**: Identify circuits related to power-seeking behaviors
- **SAE Analysis**: Use sparse autoencoders for feature extraction

## Research Background

This toolkit implements techniques from:

- **Circuit Discovery**: Mechanistic interpretability research from Anthropic, Neel Nanda, and others
- **Sparse Autoencoders**: "Towards Monosemanticity" and subsequent SAE research
- **Activation Patching**: Causal tracing methods from various mechanistic interpretability papers
- **Alignment Research**: Deceptive alignment detection from AI safety research

## Contributing

Contributions are welcome! Areas for improvement:

- Additional behavior detectors
- Support for more model architectures
- Automated circuit discovery algorithms
- Visualization tools
- Performance optimizations

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{llm_interpretability_toolkit,
  title = {LLM Interpretability Toolkit},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/LLMInterpretability}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [SAELens](https://github.com/jbloomAus/SAELens).
