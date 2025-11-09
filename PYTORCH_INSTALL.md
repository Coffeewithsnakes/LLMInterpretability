# PyTorch Installation Guide

## 🔥 Quick Fix for Windows CUDA Installation

If you're getting errors like:
```
ERROR: Could not find a version that satisfies the requirement torch
ERROR: No matching distribution found for torch
```

### ✅ Solution: Use `--extra-index-url` on Windows

Instead of `--index-url`, use `--extra-index-url`:

```powershell
# RECOMMENDED for Windows with CUDA 12.x
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Alternative: CUDA 11.8
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### Why This Happens

- `--index-url` **replaces** the default PyPI repository
- On Windows, this sometimes breaks package resolution
- `--extra-index-url` **adds** the PyTorch repository alongside PyPI
- This allows pip to find dependencies from both sources

---

## 📋 Complete Installation Instructions

### Step 1: Check Your CUDA Version

```powershell
nvidia-smi
```

Look for "CUDA Version: X.X" in the output.

### Step 2: Choose Your Installation Method

#### For CUDA 12.x (12.0, 12.1, 12.2, etc.):

```powershell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

#### For CUDA 11.8:

```powershell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

#### CPU-Only (No GPU):

```powershell
pip install torch torchvision torchaudio
```

### Step 3: Verify Installation

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected output with GPU:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3070
```

---

## 🔄 Reinstalling PyTorch (If Already Installed)

If you have CPU-only PyTorch and want GPU support:

### Step 1: Uninstall Current PyTorch

```powershell
pip uninstall torch torchvision torchaudio -y
```

### Step 2: Install with CUDA Support

```powershell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`.

---

## 🛠️ Using the CUDA Diagnostic Tool

This repository includes an automatic diagnostic tool:

```powershell
# Check what's wrong
python cuda_diagnostic.py

# Show alternative installation commands
python cuda_diagnostic.py

# Auto-fix (attempts to reinstall PyTorch)
python cuda_diagnostic.py --fix

# Full test including model loading
python cuda_diagnostic.py --test
```

**Or use the interactive menu:**

```powershell
python run_interactive.py
# → Option 8 (Testing & Diagnostics)
# → Option 5 (CUDA Diagnostic & Auto-Fix)
```

---

## 🐛 Common Issues

### Issue 1: "ERROR: No matching distribution found"

**Cause:** Using `--index-url` on Windows

**Fix:** Use `--extra-index-url` instead:
```powershell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: "torch.cuda.is_available() returns False"

**Possible causes:**
1. CPU-only PyTorch installed
2. NVIDIA driver not installed
3. CUDA version mismatch

**Diagnostic:**
```powershell
python cuda_diagnostic.py
```

**Fix for CPU-only PyTorch:**
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: "RuntimeError: CUDA out of memory"

**Cause:** Model too large for your GPU

**Fix:** Use a smaller model:
```python
# Instead of gpt2-large, use:
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
```

### Issue 4: NVIDIA driver not found

**Check:**
```powershell
nvidia-smi
```

**If this fails:**
1. Go to https://www.nvidia.com/Download/index.aspx
2. Select your GPU model
3. Download and install the driver
4. Restart your computer

---

## 📊 VRAM Requirements

Estimated GPU memory needed for common models:

| Model | Parameters | VRAM Needed |
|-------|-----------|-------------|
| gpt2-small | 124M | 2-3 GB |
| gpt2-medium | 355M | 3-4 GB |
| gpt2-large | 774M | 5-6 GB |
| gpt2-xl | 1.5B | 7-8 GB |
| Pythia-1B | 1B | 4-5 GB |
| Pythia-1.4B | 1.4B | 5-6 GB |
| Llama-3.2-1B | 1B | 4-5 GB |
| Llama-3.2-3B | 3B | 6-7 GB |

**Your RTX 3070 has 8GB VRAM** - can run up to gpt2-xl or Llama-3.2-3B.

---

## 🔗 Official PyTorch Resources

- **Official Installation Guide:** https://pytorch.org/get-started/locally/
- **Previous Versions:** https://pytorch.org/get-started/previous-versions/
- **CUDA Compatibility:** https://pytorch.org/get-started/locally/#linux-prerequisites

---

## ⚡ Quick Command Reference

```powershell
# Install PyTorch with CUDA (Windows)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is working
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU info
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Run diagnostic
python cuda_diagnostic.py

# Run tests (will auto-detect GPU)
python quick_sanity_check.py
```

---

## 💡 Pro Tips

1. **Always use `--extra-index-url` on Windows** instead of `--index-url`

2. **Check CUDA version first:**
   ```powershell
   nvidia-smi
   ```
   Match your PyTorch installation to your CUDA version.

3. **Verify after install:**
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Use the diagnostic tool:**
   ```powershell
   python cuda_diagnostic.py --test
   ```
   This loads a real model on GPU to verify everything works.

5. **Start with smaller models:**
   Use `gpt2-small` first to verify CUDA works, then scale up.

---

**Need more help?** Run the diagnostic tool or check TROUBLESHOOTING.md!
