#!/usr/bin/env python3
"""
Interactive Setup Script

This script will guide you through setting up the LLM Interpretability Toolkit.
Just run it and follow the prompts!
"""

import sys
import os
import subprocess
import platform
from pathlib import Path


def print_header(text):
    """Print a nice header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step_num, total_steps, text):
    """Print a step indicator."""
    print(f"\n[Step {step_num}/{total_steps}] {text}")
    print("-" * 70)


def print_success(text):
    """Print a success message."""
    print(f"✅ {text}")


def print_error(text):
    """Print an error message."""
    print(f"❌ {text}")


def print_warning(text):
    """Print a warning message."""
    print(f"⚠️  {text}")


def print_info(text):
    """Print an info message."""
    print(f"ℹ️  {text}")


def check_python_version():
    """Check if Python version is adequate."""
    print_step(1, 7, "Checking Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error(f"Python 3.9+ required, you have {version.major}.{version.minor}")
        print_info("Please install Python 3.9 or higher from https://www.python.org/downloads/")
        return False

    print_success(f"Python {version.major}.{version.minor} is compatible!")
    return True


def check_pip():
    """Check if pip is installed."""
    print_step(2, 7, "Checking pip (Python Package Manager)")

    try:
        import pip
        print_success("pip is installed!")
        return True
    except ImportError:
        print_error("pip is not installed!")
        print_info("Installing pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--default-pip"])
            print_success("pip installed successfully!")
            return True
        except:
            print_error("Failed to install pip automatically")
            print_info("Please install pip manually: https://pip.pypa.io/en/stable/installation/")
            return False


def create_virtual_environment():
    """Create a virtual environment."""
    print_step(3, 7, "Creating Virtual Environment")

    venv_path = Path(".venv")

    if venv_path.exists():
        print_warning("Virtual environment already exists!")
        response = input("Do you want to recreate it? (y/n): ").lower().strip()
        if response == 'y':
            print_info("Removing old virtual environment...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            print_info("Using existing virtual environment")
            return True

    print_info("Creating virtual environment (this may take a minute)...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", ".venv"])
        print_success("Virtual environment created successfully!")
        print_info(f"Location: {venv_path.absolute()}")
        return True
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def get_venv_python():
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == "Windows":
        return Path(".venv") / "Scripts" / "python.exe"
    else:
        return Path(".venv") / "bin" / "python"


def install_dependencies():
    """Install required packages."""
    print_step(4, 7, "Installing Required Packages")

    print_info("This will download and install several packages.")
    print_info("This might take 5-15 minutes depending on your internet speed...")
    print_info("☕ Time to grab a coffee!\n")

    response = input("Ready to proceed? (y/n): ").lower().strip()
    if response != 'y':
        print_warning("Installation cancelled")
        return False

    venv_python = get_venv_python()

    # Upgrade pip first
    print_info("Upgrading pip...")
    try:
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                            stdout=subprocess.DEVNULL)
        print_success("pip upgraded")
    except:
        print_warning("Could not upgrade pip, continuing anyway...")

    # Install PyTorch (GPU-aware installation)
    print_info("\n📦 Installing PyTorch (this is the big one)...")
    print_info("Detecting GPU support...")

    # Check for NVIDIA GPU
    has_nvidia = False
    cuda_version = None
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            has_nvidia = True
            # Try to extract CUDA version
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    parts = line.split('CUDA Version:')
                    if len(parts) > 1:
                        version_str = parts[1].strip().split()[0]
                        try:
                            major = int(float(version_str))
                            cuda_version = major
                        except:
                            pass
    except:
        pass

    is_windows = platform.system() == "Windows"

    if has_nvidia:
        print_success(f"NVIDIA GPU detected! Installing PyTorch with CUDA support...")
        if cuda_version:
            print_info(f"Detected CUDA {cuda_version}.x")

        # Choose correct PyTorch version
        if cuda_version == 12:
            pytorch_url = "https://download.pytorch.org/whl/cu121"
        elif cuda_version == 11:
            pytorch_url = "https://download.pytorch.org/whl/cu118"
        else:
            # Default to CUDA 12.1
            pytorch_url = "https://download.pytorch.org/whl/cu121"

        # Windows needs --extra-index-url, Linux can use --index-url
        index_flag = "--extra-index-url" if is_windows else "--index-url"

        try:
            subprocess.check_call([
                str(venv_python), "-m", "pip", "install",
                "torch", "torchvision", "torchaudio", index_flag, pytorch_url
            ])
            print_success("PyTorch with CUDA installed!")
        except:
            print_warning("CUDA install failed, trying default PyTorch...")
            subprocess.check_call([str(venv_python), "-m", "pip", "install",
                                 "torch", "torchvision", "torchaudio"])
    else:
        print_warning("No NVIDIA GPU detected, installing CPU-only PyTorch...")
        print_info("This is fine for testing, but will be slower")
        try:
            subprocess.check_call([
                str(venv_python), "-m", "pip", "install",
                "torch", "torchvision", "torchaudio"
            ])
            print_success("PyTorch (CPU) installed!")
        except:
            print_error("PyTorch installation failed!")
            print_info("Try manually: pip install torch torchvision torchaudio")
            return False

    # Install other requirements
    print_info("\n📦 Installing other dependencies...")
    try:
        subprocess.check_call([
            str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print_success("All dependencies installed!")
        return True
    except Exception as e:
        print_error(f"Failed to install dependencies: {e}")
        print_info("You can try running: pip install -r requirements.txt")
        return False


def test_installation():
    """Test if everything is working."""
    print_step(5, 7, "Testing Installation")

    venv_python = get_venv_python()

    print_info("Testing imports...")

    test_script = """
import sys
try:
    import torch
    print("✅ PyTorch:", torch.__version__)
except ImportError as e:
    print("❌ PyTorch not found:", e)
    sys.exit(1)
except Exception as e:
    print("❌ PyTorch error:", e)
    sys.exit(1)

try:
    import transformer_lens
    # TransformerLens doesn't have __version__, just check if we can import HookedTransformer
    from transformer_lens import HookedTransformer
    print("✅ TransformerLens: Installed")
except ImportError as e:
    print("❌ TransformerLens not found:", e)
    sys.exit(1)
except Exception as e:
    print("❌ TransformerLens error:", e)
    sys.exit(1)

try:
    import numpy
    print("✅ NumPy:", numpy.__version__)
except ImportError as e:
    print("❌ NumPy not found:", e)
    sys.exit(1)
except Exception as e:
    print("❌ NumPy error:", e)
    sys.exit(1)

print("\\n✅ All core dependencies are working!")
"""

    try:
        result = subprocess.run(
            [str(venv_python), "-c", test_script],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print_error("Some imports failed!")
            print(result.stderr)
            return False
        print_success("All tests passed!")
        return True
    except Exception as e:
        print_error(f"Testing failed: {e}")
        return False


def download_test_model():
    """Download a small test model."""
    print_step(6, 7, "Downloading Test Model")

    print_info("Downloading gpt2-small for testing (about 500MB)...")
    print_info("This will be cached for future use.\n")

    venv_python = get_venv_python()

    download_script = """
from transformer_lens import HookedTransformer
import torch

print("Downloading gpt2-small...")
try:
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device="cpu"
    )
    print("✅ Model downloaded successfully!")
    print(f"Model has {model.cfg.n_layers} layers and {model.cfg.n_heads} attention heads")

    # Quick test
    test_text = "Hello, world!"
    tokens = model.to_tokens(test_text)
    print(f"\\n✅ Quick test: '{test_text}' → {tokens.shape[1]} tokens")
    print("\\n✅ Everything is working!")
except Exception as e:
    print(f"❌ Error: {e}")
    import sys
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [str(venv_python), "-c", download_script],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        print(result.stdout)
        if result.returncode != 0:
            print_error("Model download failed!")
            print(result.stderr)
            return False
        print_success("Test model ready!")
        return True
    except subprocess.TimeoutExpired:
        print_error("Download timed out. Your internet might be slow.")
        print_info("The model will download automatically when you first use the toolkit.")
        return True  # Don't fail setup
    except Exception as e:
        print_error(f"Model download failed: {e}")
        print_info("The model will download automatically when you first use the toolkit.")
        return True  # Don't fail setup


def show_next_steps():
    """Show what to do next."""
    print_step(7, 7, "Setup Complete! 🎉")

    print_success("The toolkit is ready to use!\n")

    print("📚 What's Next?\n")
    print("1️⃣  Run the interactive menu:")
    if platform.system() == "Windows":
        print("    .venv\\Scripts\\python.exe run_interactive.py")
    else:
        print("    .venv/bin/python run_interactive.py")

    print("\n2️⃣  Start the web interface:")
    if platform.system() == "Windows":
        print("    .venv\\Scripts\\python.exe -m streamlit run app.py")
    else:
        print("    .venv/bin/python -m streamlit run app.py")

    print("\n3️⃣  Open Jupyter notebooks:")
    if platform.system() == "Windows":
        print("    .venv\\Scripts\\jupyter.exe notebook")
    else:
        print("    .venv/bin/jupyter notebook")

    print("\n4️⃣  Read the documentation:")
    print("    - QUICKSTART.md for quick examples")
    print("    - README.md for detailed info")
    print("    - notebooks/ for tutorials")

    print("\n" + "=" * 70)
    print("\n💡 TIP: Always activate the virtual environment before running:")
    if platform.system() == "Windows":
        print("    .venv\\Scripts\\activate")
    else:
        print("    source .venv/bin/activate")
    print("\n" + "=" * 70)


def main():
    """Main setup routine."""
    print_header("🚀 LLM Interpretability Toolkit - Interactive Setup")

    print("Welcome! This script will set up everything you need.")
    print("It will take about 10-20 minutes depending on your internet speed.")
    print("\nThe setup will:")
    print("  • Check your Python installation")
    print("  • Create an isolated environment")
    print("  • Install all required packages")
    print("  • Download a test model")
    print("  • Verify everything works")

    response = input("\nReady to start? (y/n): ").lower().strip()
    if response != 'y':
        print("\nSetup cancelled. Run this script again when you're ready!")
        return

    # Run setup steps
    if not check_python_version():
        return

    if not check_pip():
        return

    if not create_virtual_environment():
        return

    if not install_dependencies():
        print_warning("\nSetup incomplete due to dependency installation issues.")
        print_info("You can try installing manually with: pip install -r requirements.txt")
        return

    if not test_installation():
        print_warning("\nSetup completed but some tests failed.")
        print_info("You might still be able to use the toolkit.")

    download_test_model()  # Optional, don't fail if this doesn't work

    show_next_steps()

    print("\n" + "=" * 70)
    print("🎉 Setup complete! Enjoy exploring AI interpretability!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user.")
        print("Run this script again to complete setup.")
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        print_info("Please report this issue on GitHub with the error message above.")
        import traceback
        traceback.print_exc()
