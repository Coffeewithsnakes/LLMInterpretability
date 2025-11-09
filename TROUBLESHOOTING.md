# Troubleshooting Guide

Having issues? This guide will help you fix common problems!

## 🔍 Quick Diagnostic

First, run the diagnostic script to identify the problem:

```bash
python check_installation.py
```

This will tell you exactly what's wrong and how to fix it!

---

## Common Issues & Solutions

### 1. "python: command not found"

**Problem:** Python is not installed or not in your PATH.

**Solutions:**

#### Windows
- Try `py` instead of `python`:
  ```bash
  py setup_interactive.py
  ```
- Or reinstall Python and **check "Add Python to PATH"**

#### Mac/Linux
- Try `python3` instead of `python`:
  ```bash
  python3 setup_interactive.py
  ```
- Install Python: `brew install python@3.11` (Mac) or `sudo apt install python3.11` (Linux)

---

### 2. "No module named 'torch'" or "No module named 'transformer_lens'"

**Problem:** Dependencies are not installed.

**Solution:**
```bash
# Make sure you're in the project directory
cd path/to/LLMInterpretability

# Run the interactive setup
python setup_interactive.py

# Or install manually
pip install -r requirements.txt
```

**Still not working?**
```bash
# Install PyTorch separately first
pip install torch

# Then install other dependencies
pip install transformer-lens numpy matplotlib pandas
```

---

### 3. "CUDA out of memory" or "RuntimeError: CUDA error"

**Problem:** Trying to use GPU but don't have enough memory.

**Solution 1:** Use CPU instead
```python
detector = EmotionCircuitDetector(
    model_name="gpt2-small",
    device="cpu"  # Force CPU usage
)
```

**Solution 2:** Use a smaller model
```python
detector = EmotionCircuitDetector(
    model_name="gpt2-small"  # Instead of gpt2-medium or gpt2-large
)
```

**Solution 3:** Reduce batch size
```python
from src.circuit_discovery import PatchingConfig

config = PatchingConfig(
    model_name="gpt2-small",
    device="cuda",
    batch_size=1  # Reduce from default 8
)
```

---

### 4. "Permission denied" errors

**Problem:** Don't have permission to install or create files.

**Solutions:**

#### Windows
- Run Command Prompt as Administrator
- Right-click "Command Prompt" → "Run as administrator"

#### Mac/Linux
- Add `sudo` before the command:
  ```bash
  sudo python setup_interactive.py
  ```
- Or install in user directory:
  ```bash
  pip install --user -r requirements.txt
  ```

---

### 5. Virtual environment won't activate

**Problem:** Can't activate the virtual environment.

**Solutions:**

#### Windows
```bash
.venv\Scripts\activate.bat
```

#### Mac/Linux
```bash
source .venv/bin/activate
```

**If still not working:**
```bash
# Delete and recreate
rm -rf .venv  # Mac/Linux
# OR
rmdir /s .venv  # Windows

python -m venv .venv
```

---

### 6. Downloads are very slow or timing out

**Problem:** Slow internet or network issues.

**Solutions:**

1. **Use a wired connection** if possible
2. **Download during off-peak hours**
3. **Skip model download during setup** - it will download when you first use it:
   - Just press Ctrl+C when it starts downloading
   - Run the toolkit anyway, model will download on first use

4. **Use a mirror** for PyTorch:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

---

### 7. "ImportError: cannot import name 'HookedTransformer'"

**Problem:** Wrong version of TransformerLens.

**Solution:**
```bash
pip uninstall transformer-lens
pip install transformer-lens>=2.0.0
```

---

### 8. Jupyter notebook won't start

**Problem:** Jupyter not installed or not found.

**Solution:**
```bash
# Install Jupyter
pip install jupyter

# Start it
jupyter notebook

# Or use specific path
.venv/Scripts/jupyter notebook  # Windows
.venv/bin/jupyter notebook      # Mac/Linux
```

---

### 9. "ModuleNotFoundError" in Jupyter notebooks

**Problem:** Jupyter is using wrong Python kernel.

**Solution:**

1. Install ipykernel in your virtual environment:
   ```bash
   pip install ipykernel
   ```

2. Add your virtual environment as a kernel:
   ```bash
   python -m ipykernel install --user --name=llm-interp
   ```

3. In Jupyter, go to: Kernel → Change Kernel → llm-interp

---

### 10. Streamlit app won't start

**Problem:** Streamlit not installed.

**Solution:**
```bash
pip install streamlit
streamlit run app.py
```

---

### 11. Code runs but results seem wrong

**Problem:** Might be using wrong examples or parameters.

**Checklist:**
- ✅ Are you using enough examples? (At least 3-5 per category)
- ✅ Are your prompts clearly different? (Clean vs corrupted should be very different)
- ✅ Is your threshold too high? Try lowering it: `threshold=0.3`
- ✅ Are you using the right model? Small models might not show strong patterns

**Debug mode:**
```python
# Add more logging
import logging
logging.basicConfig(level=logging.INFO)
```

---

### 12. "out of memory" error on CPU

**Problem:** Not enough RAM.

**Solutions:**

1. **Use smaller examples:**
   ```python
   # Use fewer prompts
   happy_prompts = happy_prompts[:3]  # Just first 3
   ```

2. **Reduce model size:**
   ```python
   # Use distilgpt2 instead
   detector = EmotionCircuitDetector(model_name="distilgpt2")
   ```

3. **Close other applications** to free up RAM

---

### 13. Results are inconsistent

**Problem:** Randomness in generation or small sample size.

**Solutions:**

1. **Set random seed:**
   ```python
   import torch
   torch.manual_seed(42)
   ```

2. **Use more examples:**
   ```python
   # Use 10+ examples instead of 3-5
   ```

3. **Increase threshold:**
   ```python
   circuit = detector.discover_emotion_circuit(
       threshold=0.7  # Higher = more strict
   )
   ```

---

### 14. Process is taking too long

**Problem:** Circuit discovery is slow.

**Expected times:**
- Small model (GPT-2 small) on CPU: 2-5 minutes
- Medium model on CPU: 10-20 minutes
- Small model on GPU: 30-60 seconds

**Speed it up:**

1. **Use GPU if available:**
   ```python
   device="cuda"
   ```

2. **Skip pruning:**
   ```python
   circuit = detector.discover_emotion_circuit(
       ...,
       prune=False  # Skip pruning step
   )
   ```

3. **Use fewer examples:**
   ```python
   clean_prompts = clean_prompts[:5]  # Just 5 instead of 10
   ```

4. **Reduce components to check:**
   ```python
   config = PatchingConfig(
       patch_layers=[0, 5, 11],  # Only check specific layers
       patch_attn_heads=False    # Skip individual heads
   )
   ```

---

### 15. "SSL Certificate" or "Connection" errors

**Problem:** Network/firewall issues.

**Solutions:**

1. **Use a different network** (not corporate/school wifi)

2. **Disable VPN** temporarily

3. **Download models manually:**
   ```python
   from transformers import GPT2LMHeadModel
   model = GPT2LMHeadModel.from_pretrained("gpt2")
   ```

---

## 🆘 Still Stuck?

### Check the logs
Many errors print helpful details. Read the full error message!

### Run diagnostics
```bash
python check_installation.py
```

### Try a clean install
```bash
# Remove everything
rm -rf .venv
rm -rf ~/.cache/huggingface  # Remove cached models

# Start fresh
python setup_interactive.py
```

### Ask for help
1. **Check GitHub Issues:** Someone might have had the same problem
2. **Create a new issue:** Include:
   - Your operating system
   - Python version (`python --version`)
   - Full error message
   - What you were trying to do

### Minimal working example
Sometimes simpler is better:

```python
# The absolute simplest test
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
tokens = model.to_tokens("Hello world")
output = model(tokens)
print("Success!")
```

If this works, the issue is with the toolkit code, not your installation.

---

## 💡 Pro Tips

1. **Always activate your virtual environment first:**
   ```bash
   source .venv/bin/activate  # Mac/Linux
   .venv\Scripts\activate     # Windows
   ```

2. **Check you're in the right directory:**
   ```bash
   pwd  # Mac/Linux
   cd   # Windows
   # Should show: .../LLMInterpretability
   ```

3. **Update packages if they're old:**
   ```bash
   pip install --upgrade pip
   pip install --upgrade -r requirements.txt
   ```

4. **Use `pip list` to see what's installed:**
   ```bash
   pip list | grep torch
   pip list | grep transformer
   ```

5. **When in doubt, check the examples:**
   - Run `python examples/simple_emotion_circuit.py`
   - If this works, your installation is fine!

---

**Remember:** Most issues are simple! Don't give up - check the error message carefully and try the solutions above. 🚀
