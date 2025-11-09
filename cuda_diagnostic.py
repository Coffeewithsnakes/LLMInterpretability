#!/usr/bin/env python3
"""
CUDA Diagnostic and Auto-Fix Tool

Detects common CUDA issues and provides fixes.

Usage:
    python cuda_diagnostic.py
    python cuda_diagnostic.py --fix  # Auto-fix issues
"""

import sys
import subprocess
import os
import argparse

def print_header(text):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_check(text, status=None):
    """Print check result."""
    if status is True:
        print(f"✅ {text}")
    elif status is False:
        print(f"❌ {text}")
    elif status is None:
        print(f"⚠️  {text}")
    else:
        print(f"ℹ️  {text}")

def run_command(cmd, capture=True):
    """Run shell command and return output."""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode, "", ""
    except Exception as e:
        return 1, "", str(e)

def check_nvidia_driver():
    """Check if NVIDIA driver is installed."""
    print_header("NVIDIA Driver Check")

    code, stdout, stderr = run_command("nvidia-smi")
    if code == 0:
        print_check("NVIDIA driver is installed", True)
        # Parse nvidia-smi output for GPU info
        for line in stdout.split('\n'):
            if 'NVIDIA' in line or 'CUDA Version' in line:
                print(f"   {line.strip()}")
        return True
    else:
        print_check("NVIDIA driver NOT found", False)
        print("\n💡 To install NVIDIA drivers:")
        print("   Ubuntu/Debian: sudo apt-get install nvidia-driver-<version>")
        print("   Check available GPUs: lspci | grep -i nvidia")
        return False

def check_cuda_toolkit():
    """Check if CUDA toolkit is installed."""
    print_header("CUDA Toolkit Check")

    code, stdout, _ = run_command("nvcc --version")
    if code == 0:
        print_check("CUDA toolkit is installed", True)
        print(f"   {stdout.split('release')[1].split(',')[0].strip() if 'release' in stdout else stdout}")
        return True
    else:
        print_check("CUDA toolkit NOT found", None)
        print("\n💡 Note: CUDA toolkit is optional for PyTorch.")
        print("   PyTorch includes its own CUDA libraries.")
        return False

def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print_header("PyTorch Check")

    try:
        import torch
        print_check("PyTorch is installed", True)
        print(f"   Version: {torch.__version__}")

        # Check if PyTorch has CUDA support
        if torch.cuda.is_available():
            print_check("PyTorch CUDA support: ENABLED", True)
            print(f"   CUDA version (PyTorch): {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            print(f"   Available GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({vram_gb:.1f} GB VRAM)")

            return True, "cuda"
        else:
            print_check("PyTorch CUDA support: DISABLED", False)
            print(f"   This is a CPU-only PyTorch installation")
            return True, "cpu-only"

    except ImportError:
        print_check("PyTorch is NOT installed", False)
        return False, None

def check_transformerlens():
    """Check TransformerLens installation."""
    print_header("TransformerLens Check")

    try:
        import transformer_lens
        print_check("TransformerLens is installed", True)
        return True
    except ImportError:
        print_check("TransformerLens is NOT installed", False)
        return False

def diagnose_issues(pytorch_status, pytorch_type, has_nvidia):
    """Diagnose common CUDA issues."""
    print_header("Diagnosis")

    issues = []
    fixes = []

    if not pytorch_status:
        issues.append("PyTorch is not installed")
        fixes.append({
            "issue": "PyTorch missing",
            "fix": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "description": "Install PyTorch with CUDA 12.1 support"
        })
    elif pytorch_type == "cpu-only" and has_nvidia:
        issues.append("PyTorch is CPU-only but you have an NVIDIA GPU")
        fixes.append({
            "issue": "CPU-only PyTorch with available GPU",
            "fix": "pip uninstall torch torchvision torchaudio -y && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "description": "Reinstall PyTorch with CUDA support"
        })
    elif pytorch_type == "cpu-only" and not has_nvidia:
        issues.append("No NVIDIA GPU detected - CPU-only mode is expected")
        print_check("Using CPU is normal for your system", True)

    if issues:
        print("\n⚠️  Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print_check("No issues detected! System is properly configured", True)

    return fixes

def apply_fixes(fixes, auto=False):
    """Apply fixes for detected issues."""
    if not fixes:
        return

    print_header("Available Fixes")

    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['description']}")
        print(f"   Issue: {fix['issue']}")
        print(f"   Command: {fix['fix']}")

    print("\n" + "=" * 70)

    if auto:
        print("\n🔧 Auto-fix mode enabled. Applying fixes...")
        for fix in fixes:
            print(f"\n▶️  {fix['description']}...")
            code, stdout, stderr = run_command(fix['fix'], capture=False)
            if code == 0:
                print_check(f"Fix applied successfully", True)
            else:
                print_check(f"Fix failed", False)
                if stderr:
                    print(f"   Error: {stderr}")
    else:
        print("\n💡 To apply these fixes:")
        print("   Run: python cuda_diagnostic.py --fix")
        print("   Or manually run the commands above")

def test_cuda_inference():
    """Test actual CUDA inference."""
    print_header("CUDA Inference Test")

    try:
        import torch

        if not torch.cuda.is_available():
            print_check("CUDA not available - skipping inference test", None)
            return False

        print("Testing CUDA tensor operations...")
        device = torch.device("cuda")

        # Create test tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)

        # Test computation
        z = torch.mm(x, y)

        print_check("CUDA tensor operations work", True)
        print(f"   Device: {z.device}")
        print(f"   Tensor shape: {z.shape}")

        # Test model loading
        print("\nTesting model loading on CUDA...")
        try:
            from transformer_lens import HookedTransformer

            model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
            print_check("Model loaded on CUDA successfully", True)
            print(f"   Model device: {next(model.parameters()).device}")

            # Test inference
            test_input = "Hello world"
            tokens = model.to_tokens(test_input)
            with torch.no_grad():
                output = model(tokens)

            print_check("CUDA inference successful", True)
            print(f"   Input: '{test_input}'")
            print(f"   Output shape: {output.shape}")

            return True

        except Exception as e:
            print_check(f"Model loading failed: {e}", False)
            return False

    except ImportError:
        print_check("PyTorch not installed - cannot test", False)
        return False
    except Exception as e:
        print_check(f"CUDA test failed: {e}", False)
        return False

def main():
    parser = argparse.ArgumentParser(description="CUDA Diagnostic and Auto-Fix Tool")
    parser.add_argument("--fix", action="store_true", help="Automatically apply fixes")
    parser.add_argument("--test", action="store_true", help="Run inference test")
    args = parser.parse_args()

    print("=" * 70)
    print("  🔍 CUDA Diagnostic Tool")
    print("=" * 70)
    print("\nThis tool will check your CUDA setup and offer fixes for common issues.\n")

    # Run checks
    has_nvidia = check_nvidia_driver()
    has_cuda_toolkit = check_cuda_toolkit()
    pytorch_status, pytorch_type = check_pytorch()
    has_transformerlens = check_transformerlens()

    # Diagnose issues
    fixes = diagnose_issues(pytorch_status, pytorch_type, has_nvidia)

    # Apply fixes if requested
    if args.fix and fixes:
        apply_fixes(fixes, auto=True)

        # Re-check after fixes
        print_header("Re-checking after fixes")
        check_pytorch()
    elif fixes:
        apply_fixes(fixes, auto=False)

    # Run inference test if requested or if everything looks good
    if args.test or (pytorch_status and pytorch_type == "cuda" and not fixes):
        test_cuda_inference()

    # Final recommendations
    print_header("Recommendations")

    if pytorch_status and pytorch_type == "cuda":
        print_check("Your system is ready for GPU-accelerated inference!", True)
        print("\n💡 Next steps:")
        print("   • Run tests with: python quick_sanity_check.py")
        print("   • Start the interactive menu: python run_interactive.py")
        print("   • All code will automatically use your GPU!")
    elif pytorch_status and pytorch_type == "cpu-only" and has_nvidia:
        print_check("You have a GPU but PyTorch is using CPU", False)
        print("\n💡 To fix:")
        print("   • Run: python cuda_diagnostic.py --fix")
        print("   • Or manually reinstall PyTorch with CUDA support")
    elif pytorch_status and pytorch_type == "cpu-only":
        print_check("CPU mode is normal for your system", True)
        print("\n💡 Everything will work, just slower than with a GPU")
    else:
        print_check("PyTorch needs to be installed", False)
        print("\n💡 To fix:")
        print("   • Run: python cuda_diagnostic.py --fix")
        print("   • Or: pip install -r requirements.txt")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
