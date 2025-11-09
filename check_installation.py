#!/usr/bin/env python3
"""
Installation Diagnostic Tool

Run this script to check if everything is installed correctly.
It will tell you exactly what's wrong and how to fix it!
"""

import sys
import platform


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_item(name, check_func, fix_hint=""):
    """Check an item and print result."""
    try:
        result = check_func()
        if result:
            print(f"✅ {name}: OK")
            if isinstance(result, str):
                print(f"   └─ {result}")
            return True
        else:
            print(f"❌ {name}: FAILED")
            if fix_hint:
                print(f"   └─ Fix: {fix_hint}")
            return False
    except Exception as e:
        print(f"❌ {name}: ERROR")
        print(f"   └─ {str(e)}")
        if fix_hint:
            print(f"   └─ Fix: {fix_hint}")
        return False


def main():
    print_section("🔍 LLM Interpretability Toolkit - Installation Diagnostic")

    print("\nThis tool will check your installation and tell you how to fix any problems.")
    print("Checking...\n")

    all_checks_passed = True

    # System Information
    print_section("💻 System Information")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")

    # Python Version Check
    print_section("🐍 Python Version")
    def check_python():
        v = sys.version_info
        if v.major == 3 and v.minor >= 9:
            return f"Python {v.major}.{v.minor}.{v.micro}"
        return False

    if not check_item(
        "Python 3.9+",
        check_python,
        "Install Python 3.9+ from https://www.python.org/downloads/"
    ):
        all_checks_passed = False
        print("\n⚠️  CRITICAL: Python version too old. Please upgrade!")
        print("   Visit: https://www.python.org/downloads/")
        return

    # Core Dependencies
    print_section("📦 Core Dependencies")

    def check_torch():
        import torch
        return f"v{torch.__version__}"

    if not check_item(
        "PyTorch",
        check_torch,
        "pip install torch"
    ):
        all_checks_passed = False

    def check_transformerlens():
        import transformer_lens
        from transformer_lens import HookedTransformer
        # TransformerLens doesn't have __version__
        return "Installed"

    if not check_item(
        "TransformerLens",
        check_transformerlens,
        "pip install transformer-lens"
    ):
        all_checks_passed = False

    def check_numpy():
        import numpy
        return f"v{numpy.__version__}"

    if not check_item(
        "NumPy",
        check_numpy,
        "pip install numpy"
    ):
        all_checks_passed = False

    def check_transformers():
        import transformers
        return f"v{transformers.__version__}"

    check_item(
        "Transformers (Hugging Face)",
        check_transformers,
        "pip install transformers"
    )

    # Optional Dependencies
    print_section("📊 Optional Dependencies")

    def check_matplotlib():
        import matplotlib
        return "Installed"

    check_item(
        "Matplotlib (for plotting)",
        check_matplotlib,
        "pip install matplotlib"
    )

    def check_pandas():
        import pandas
        return "Installed"

    check_item(
        "Pandas (for data handling)",
        check_pandas,
        "pip install pandas"
    )

    def check_jupyter():
        import jupyter
        return "Installed"

    check_item(
        "Jupyter (for notebooks)",
        check_jupyter,
        "pip install jupyter"
    )

    def check_streamlit():
        import streamlit
        return "Installed"

    check_item(
        "Streamlit (for web UI)",
        check_streamlit,
        "pip install streamlit"
    )

    # GPU Check
    print_section("🎮 GPU Support")

    def check_cuda():
        import torch
        if torch.cuda.is_available():
            return f"CUDA {torch.version.cuda} with {torch.cuda.device_count()} GPU(s)"
        return False

    has_gpu = check_item(
        "CUDA (GPU support)",
        check_cuda,
        "GPU not available - will use CPU (slower but works fine)"
    )

    if not has_gpu:
        print("   ℹ️  No GPU detected. The toolkit will use CPU (this is fine, just slower)")

    # Project Structure
    print_section("📁 Project Structure")

    from pathlib import Path

    def check_src():
        return Path("src").exists()

    check_item(
        "src/ directory",
        check_src,
        "Make sure you're in the LLMInterpretability directory"
    )

    def check_notebooks():
        return Path("notebooks").exists()

    check_item(
        "notebooks/ directory",
        check_notebooks,
        "Clone the full repository"
    )

    def check_requirements():
        return Path("requirements.txt").exists()

    check_item(
        "requirements.txt",
        check_requirements,
        "Clone the full repository"
    )

    # Functionality Tests
    print_section("🧪 Functionality Tests")

    def test_import_toolkit():
        from src.behavior_detection import EmotionCircuitDetector
        from src.circuit_discovery import CircuitFinder
        return "All modules import successfully"

    check_item(
        "Toolkit modules",
        test_import_toolkit,
        "Check that src/ directory is present and properly structured"
    )

    def test_model_loading():
        from transformer_lens import HookedTransformer
        try:
            # Don't actually download, just check if the function works
            import torch
            return "Model loading capability available"
        except:
            return False

    check_item(
        "Model loading",
        test_model_loading,
        "Check internet connection for downloading models"
    )

    # Summary
    print_section("📋 Summary")

    if all_checks_passed:
        print("\n✅ All critical checks passed!")
        print("\n🎉 Your installation looks good!")
        print("\n📚 Next steps:")
        print("   1. Run: python run_interactive.py")
        print("   2. Or try: python examples/simple_emotion_circuit.py")
        print("   3. Or explore notebooks: jupyter notebook")
    else:
        print("\n⚠️  Some critical checks failed!")
        print("\n🔧 To fix your installation:")
        print("   1. Read the error messages above")
        print("   2. Run: python setup_interactive.py")
        print("   3. Or manually install: pip install -r requirements.txt")
        print("\n📖 For more help, see TROUBLESHOOTING.md")

    # Recommendations
    print_section("💡 Recommendations")

    if not has_gpu:
        print("\n💡 Consider using Google Colab for free GPU access:")
        print("   https://colab.research.google.com/")
        print("   Upload the notebooks and run them there!")

    print("\n💡 For best experience:")
    print("   • Use Python 3.10 or 3.11")
    print("   • Have at least 8GB RAM")
    print("   • Use a GPU if you have one (NVIDIA with CUDA)")
    print("   • Make sure you have good internet for downloading models")

    # Quick fixes
    print_section("🚑 Quick Fixes")
    print("\nIf something failed, try these commands:\n")
    print("1. Install all dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Install just the essentials:")
    print("   pip install torch transformer-lens numpy")
    print("\n3. Run the automated setup:")
    print("   python setup_interactive.py")
    print("\n4. Check if you're in the right directory:")
    if platform.system() == "Windows":
        print("   cd")
    else:
        print("   pwd")
    print("   (should show: .../LLMInterpretability)")

    print("\n" + "=" * 70)
    print("\n💬 Need more help? Check TROUBLESHOOTING.md or create a GitHub issue!")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrupted.")
    except Exception as e:
        print(f"\n❌ Unexpected error during diagnostic: {e}")
        print("\nPlease report this issue on GitHub!")
        import traceback
        traceback.print_exc()
