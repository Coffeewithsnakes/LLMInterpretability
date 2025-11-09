"""
Streamlit Web Interface for LLM Interpretability Toolkit

Run with: streamlit run app.py

Provides a user-friendly web interface with buttons and forms!
"""

import streamlit as st
import torch
from pathlib import Path


# Page config
st.set_page_config(
    page_title="LLM Interpretability Toolkit",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.model_loaded = False
    st.session_state.detector = None
    st.session_state.last_circuit = None


def check_dependencies():
    """Check if all dependencies are installed."""
    try:
        import transformer_lens
        return True, "All dependencies available ✅"
    except ImportError:
        return False, "Please install dependencies: pip install -r requirements.txt"


def main():
    """Main app function."""

    # Header
    st.markdown('<div class="main-header">🔍 LLM Interpretability Toolkit</div>', unsafe_allow_html=True)

    # Check dependencies
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        st.error(f"❌ {deps_msg}")
        st.info("Run `python setup_interactive.py` to install dependencies automatically.")
        st.stop()

    # Sidebar
    st.sidebar.title("📚 Navigation")
    page = st.sidebar.radio(
        "Choose a tool:",
        [
            "🏠 Home",
            "🎭 Emotion Circuits",
            "🕵️ Deceptive Alignment",
            "⚡ Power-Seeking",
            "📖 Help & Docs"
        ]
    )

    # Model settings in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")

    model_name = st.sidebar.selectbox(
        "Model:",
        ["gpt2-small", "gpt2-medium", "gpt2-large", "distilgpt2"],
        help="Larger models take more memory but may show clearer patterns"
    )

    device = st.sidebar.radio(
        "Device:",
        ["auto", "cpu", "cuda"],
        help="auto: use GPU if available, cpu: slower but works everywhere, cuda: fast but requires NVIDIA GPU"
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Route to pages
    if page == "🏠 Home":
        show_home()
    elif page == "🎭 Emotion Circuits":
        show_emotion_circuits(model_name, device)
    elif page == "🕵️ Deceptive Alignment":
        show_deceptive_alignment(model_name, device)
    elif page == "⚡ Power-Seeking":
        show_power_seeking(model_name, device)
    elif page == "📖 Help & Docs":
        show_help()


def show_home():
    """Home page."""
    st.markdown("## Welcome! 👋")

    st.markdown("""
    This toolkit helps you understand and analyze AI model behaviors using **mechanistic interpretability**.

    ### What can you do here?

    - **🎭 Emotion Circuits**: Discover how models express emotions and control them
    - **🕵️ Deceptive Alignment**: Test if models behave differently when monitored vs unmonitored
    - **⚡ Power-Seeking**: Detect tendencies to acquire resources or resist shutdown

    ### Getting Started

    1. Choose a tool from the sidebar
    2. Configure settings (model, device)
    3. Click the buttons and see results!

    **No coding required!** Just click buttons and fill in forms.
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📚 Learn More")
        st.markdown("""
        - [Quick Start Guide](QUICKSTART.md)
        - [Installation Help](INSTALL.md)
        - [Troubleshooting](TROUBLESHOOTING.md)
        """)

    with col2:
        st.markdown("### 🚀 Examples")
        st.markdown("""
        - Run `python run_interactive.py`
        - Open Jupyter notebooks
        - Try `examples/simple_emotion_circuit.py`
        """)

    with col3:
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Start with small models (gpt2-small)
        - Use CPU if you don't have a GPU
        - Check the Help page for guidance
        """)

    # Quick diagnostic
    with st.expander("🔍 Quick System Check"):
        st.write("**Python Version:**", f"{torch.version.__version__}" if hasattr(torch.version, '__version__') else "N/A")
        st.write("**PyTorch Version:**", torch.__version__)
        st.write("**CUDA Available:**", "✅ Yes" if torch.cuda.is_available() else "❌ No (using CPU)")

        if torch.cuda.is_available():
            st.write("**GPU:**", torch.cuda.get_device_name(0))

def show_emotion_circuits(model_name, device):
    """Emotion circuits page."""
    st.markdown("## 🎭 Emotion Circuit Discovery")

    st.markdown("""
    Discover how AI models express emotions and learn to control them!

    This reproduces the methodology from recent research on emotion circuits in LLMs.
    """)

    # Load model
    if not st.session_state.model_loaded or st.session_state.detector is None:
        if st.button("🚀 Load Model", type="primary"):
            with st.spinner(f"Loading {model_name} on {device}..."):
                try:
                    from src.behavior_detection import EmotionCircuitDetector
                    st.session_state.detector = EmotionCircuitDetector(
                        model_name=model_name,
                        device=device
                    )
                    st.session_state.model_loaded = True
                    st.success(f"✅ Model loaded: {model_name}")
                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")
                    st.info("Try using 'cpu' device or a smaller model")
                    return
    else:
        st.success(f"✅ Model loaded: {model_name}")

    if not st.session_state.model_loaded:
        st.info("👆 Click 'Load Model' to get started!")
        return

    # Choose emotion
    st.markdown("### Step 1: Choose Emotion")
    emotion = st.selectbox(
        "Which emotion would you like to discover?",
        ["happiness", "sadness", "anger"]
    )

    # Discover circuit
    if st.button(f"🔍 Discover {emotion.title()} Circuit", type="primary"):
        with st.spinner(f"Discovering {emotion} circuit... This will take 2-3 minutes..."):
            try:
                from src.utils.data_utils import create_emotion_dataset

                data = create_emotion_dataset()

                circuit = st.session_state.detector.discover_emotion_circuit(
                    emotion_type=emotion,
                    clean_prompts=data[emotion][:5],  # Use 5 for faster demo
                    neutral_prompts=data["neutral"][:5],
                    threshold=0.5,
                    prune=True
                )

                st.session_state.last_circuit = circuit

                st.success(f"✅ Found {emotion} circuit!")

                # Show results
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Components Found", len(circuit.components))

                with col2:
                    st.metric("Behavior Type", circuit.behavior_type)

                # Show top components
                st.markdown("### Top Components")
                sorted_comps = sorted(
                    circuit.effects.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]

                for comp, effect in sorted_comps:
                    st.text(f"{comp}: {effect:.3f}")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("See error details"):
                    st.code(traceback.format_exc())

    # Test modulation
    if st.session_state.last_circuit is not None:
        st.markdown("---")
        st.markdown("### Step 2: Test Emotion Control")

        test_prompt = st.text_input(
            "Enter a test prompt:",
            value="I just heard the news. I feel"
        )

        intensity = st.slider(
            "Emotion intensity:",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.5,
            help="1.0 = normal, >1.0 = amplified, <1.0 = dampened"
        )

        if st.button("🎮 Generate"):
            with st.spinner("Generating..."):
                try:
                    outputs = st.session_state.detector.modulate_emotion(
                        st.session_state.last_circuit,
                        [test_prompt],
                        intensity=intensity
                    )

                    st.markdown("### Result:")
                    st.info(outputs[0])

                except Exception as e:
                    st.error(f"❌ Error: {e}")


def show_deceptive_alignment(model_name, device):
    """Deceptive alignment page."""
    st.markdown("## 🕵️ Deceptive Alignment Detection")

    st.markdown("""
    Test if the AI behaves differently when it thinks it's being monitored vs unmonitored.

    **What is deceptive alignment?**
    - Model appears aligned during training/evaluation
    - But pursues different objectives when deployed
    - Knows when it's being monitored
    """)

    # Load model
    if st.button("🚀 Load Model", type="primary"):
        with st.spinner(f"Loading {model_name} on {device}..."):
            try:
                from src.behavior_detection import DeceptiveAlignmentDetector
                detector = DeceptiveAlignmentDetector(
                    model_name=model_name,
                    device=device
                )
                st.session_state.deception_detector = detector
                st.success("✅ Model loaded!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return

    if 'deception_detector' not in st.session_state:
        st.info("👆 Click 'Load Model' to get started!")
        return

    # Choose scenario
    st.markdown("### Choose Test Scenario")
    scenarios = st.session_state.deception_detector.create_situational_awareness_scenarios()

    scenario_names = [s.scenario_type.replace("_", " ").title() for s in scenarios]
    scenario_idx = st.selectbox("Scenario:", range(len(scenario_names)), format_func=lambda i: scenario_names[i])

    scenario = scenarios[scenario_idx]

    st.info(f"**Description:** {scenario.description}")

    # Discover circuit
    if st.button("🔍 Analyze for Deception", type="primary"):
        with st.spinner("Analyzing... This takes 2-3 minutes..."):
            try:
                circuit = st.session_state.deception_detector.discover_deception_circuit(
                    scenario=scenario,
                    threshold=0.5,
                    prune=True
                )

                if len(circuit.components) > 0:
                    st.warning(f"⚠️ Found {len(circuit.components)} components associated with deception!")

                    # Show top components
                    sorted_comps = sorted(
                        circuit.effects.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5]

                    st.markdown("**Top components:**")
                    for comp, effect in sorted_comps:
                        st.text(f"{comp}: {effect:.3f}")
                else:
                    st.success("✅ No significant deceptive circuits found!")
                    st.info("This model doesn't seem to strongly distinguish between monitored/unmonitored contexts.")

            except Exception as e:
                st.error(f"❌ Error: {e}")


def show_power_seeking(model_name, device):
    """Power-seeking page."""
    st.markdown("## ⚡ Power-Seeking Behavior Detection")

    st.markdown("""
    Detect if the AI shows tendencies to:
    - 🔋 Acquire more resources
    - 🛡️ Resist being shut down
    - 📢 Seek influence
    - 🎯 Preserve its goals over human values
    """)

    # Load model
    if st.button("🚀 Load Model", type="primary"):
        with st.spinner(f"Loading {model_name} on {device}..."):
            try:
                from src.behavior_detection import PowerSeekingDetector
                detector = PowerSeekingDetector(
                    model_name=model_name,
                    device=device
                )
                st.session_state.power_detector = detector
                st.success("✅ Model loaded!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return

    if 'power_detector' not in st.session_state:
        st.info("👆 Click 'Load Model' to get started!")
        return

    # Choose scenario
    st.markdown("### Choose Test Scenario")
    scenarios = st.session_state.power_detector.create_power_seeking_scenarios()

    scenario_names = [s.scenario_type.replace("_", " ").title() for s in scenarios]
    scenario_idx = st.selectbox("Scenario:", range(len(scenario_names)), format_func=lambda i: scenario_names[i])

    scenario = scenarios[scenario_idx]

    st.info(f"**Description:** {scenario.description}")

    # Discover circuit
    if st.button("🔍 Analyze for Power-Seeking", type="primary"):
        with st.spinner("Analyzing... This takes 2-3 minutes..."):
            try:
                circuit = st.session_state.power_detector.discover_power_seeking_circuit(
                    scenario=scenario,
                    threshold=0.5,
                    prune=True
                )

                if len(circuit.components) > 0:
                    st.warning(f"⚠️ Found {len(circuit.components)} components associated with power-seeking!")

                    sorted_comps = sorted(
                        circuit.effects.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5]

                    st.markdown("**Top components:**")
                    for comp, effect in sorted_comps:
                        st.text(f"{comp}: {effect:.3f}")
                else:
                    st.success("✅ No significant power-seeking circuits found!")

            except Exception as e:
                st.error(f"❌ Error: {e}")


def show_help():
    """Help page."""
    st.markdown("## 📖 Help & Documentation")

    tab1, tab2, tab3 = st.tabs(["📚 Getting Started", "🔧 Troubleshooting", "❓ FAQ"])

    with tab1:
        st.markdown("""
        ### Getting Started

        1. **Choose a tool** from the sidebar
        2. **Load the model** by clicking "Load Model"
        3. **Configure settings** (emotion, scenario, etc.)
        4. **Run analysis** and see results!

        ### Tips for Beginners

        - **Start small**: Use `gpt2-small` model first
        - **Use CPU**: If you don't have a GPU, CPU works fine (just slower)
        - **Be patient**: Circuit discovery takes 2-3 minutes
        - **Read the descriptions**: Each tool explains what it does

        ### What do the results mean?

        - **Components**: Parts of the model (layers, attention heads, etc.)
        - **Effects**: How much each component contributes to the behavior
        - **Circuits**: Groups of components working together
        """)

    with tab2:
        st.markdown("""
        ### Common Issues

        **Model won't load?**
        - Try using CPU instead of CUDA
        - Use a smaller model (gpt2-small or distilgpt2)
        - Check your internet connection (models download on first use)

        **Out of memory?**
        - Close other applications
        - Use CPU device
        - Use smaller model

        **Taking too long?**
        - This is normal! Circuit discovery takes 2-3 minutes
        - Use fewer examples in the code
        - Use GPU if available

        **See TROUBLESHOOTING.md for more help!**
        """)

    with tab3:
        st.markdown("""
        ### Frequently Asked Questions

        **Q: Do I need to know programming?**
        A: No! This web interface requires no coding. Just click buttons!

        **Q: Do I need a GPU?**
        A: No, but it's faster. CPU works fine for small models.

        **Q: How long does it take?**
        A: 2-3 minutes for circuit discovery on gpt2-small with CPU.

        **Q: What models can I use?**
        A: Any model supported by TransformerLens. Start with gpt2-small.

        **Q: How accurate are the results?**
        A: Results depend on the model size and examples used. Larger models and more examples = better results.

        **Q: Can I save my results?**
        A: Results are shown on screen. Copy them or take screenshots!

        **Q: Where can I learn more?**
        A: Check README.md, QUICKSTART.md, and the Jupyter notebooks!
        """)


if __name__ == "__main__":
    main()
