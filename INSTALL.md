# Installation Guide for Complete Beginners

Welcome! This guide will walk you through installing this toolkit **step by step**. Don't worry if you're new to this - we'll explain everything!

## What You'll Need

- A computer with Windows, Mac, or Linux
- Internet connection
- About 5GB of free disk space
- 30 minutes of time

## Step 1: Install Python

### Do you have Python installed?

Open a terminal/command prompt and type:
```bash
python --version
```

**If you see something like "Python 3.9.0" or higher**: ✅ You're good! Skip to Step 2.

**If you see an error or a version lower than 3.9**: Install Python:

#### Windows
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 (or latest 3.x version)
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Click "Install Now"

#### Mac
1. Open Terminal (search for "Terminal" in Spotlight)
2. Install Homebrew if you don't have it:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Install Python:
   ```bash
   brew install python@3.11
   ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3-pip
```

## Step 2: Open a Terminal

### Windows
- Press `Windows + R`
- Type `cmd` and press Enter
- OR search for "Command Prompt" in the Start menu

### Mac
- Press `Cmd + Space`
- Type "Terminal" and press Enter

### Linux
- Press `Ctrl + Alt + T`
- OR search for "Terminal" in your applications

## Step 3: Navigate to Your Project

First, decide where you want to put this project. We recommend your home directory.

```bash
# Windows
cd %USERPROFILE%\Documents

# Mac/Linux
cd ~/Documents
```

Now get the code:

### Option A: If you have git installed
```bash
git clone https://github.com/Coffeewithsnakes/LLMInterpretability.git
cd LLMInterpretability
```

### Option B: Download ZIP
1. Go to the GitHub page
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file
5. Open terminal and navigate to the extracted folder:
   ```bash
   cd path/to/LLMInterpretability
   ```

## Step 4: Run the Automated Setup

We've created a script that does everything for you!

```bash
python setup_interactive.py
```

This will:
- ✅ Check your Python version
- ✅ Create a virtual environment (isolated workspace)
- ✅ Install all required packages
- ✅ Download a small test model
- ✅ Run a quick test
- ✅ Guide you through your first circuit discovery!

**Just follow the prompts!** The script will ask you questions and explain each step.

## Step 5: Start Using the Toolkit

After setup completes, you have three ways to use the toolkit:

### Option 1: Interactive Menu (Easiest! 🌟)
```bash
python run_interactive.py
```
This gives you a menu where you can choose what to do with arrow keys!

### Option 2: Web Interface (Pretty! 🎨)
```bash
streamlit run app.py
```
Opens a web page in your browser with buttons and forms!

### Option 3: Jupyter Notebooks (Learning! 📚)
```bash
jupyter notebook
```
Then open `notebooks/01_emotion_circuits.ipynb`

## Troubleshooting

### "python: command not found"
- **Windows**: Try `py` instead of `python`
- **Mac/Linux**: Try `python3` instead of `python`

### "Permission denied"
- **Mac/Linux**: Add `sudo` before the command
- Example: `sudo python setup_interactive.py`

### "No module named 'torch'"
The automated setup should install this, but if it doesn't:
```bash
pip install torch transformers
```

### "Out of memory" error
You're trying to run on a model that's too big. Use `gpt2-small` (the default) instead of larger models.

### "CUDA not available" warning
This is fine! It means you don't have a NVIDIA GPU. The toolkit will use your CPU instead (just slower).

### Still stuck?
Run the diagnostic script:
```bash
python check_installation.py
```

This will tell you exactly what's wrong and how to fix it!

## What's Next?

After installation, check out:
1. `QUICKSTART.md` - Quick examples
2. `run_interactive.py` - Interactive menu system
3. `app.py` - Web interface
4. `notebooks/` - Detailed tutorials

## Getting Help

- 📖 Read the troubleshooting section below
- 🐛 Check GitHub issues
- 💬 Ask in discussions
- 📧 Contact maintainers

---

## Detailed Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'transformer_lens'"

**Solution:**
```bash
pip install transformer-lens
```

### Issue: Virtual environment not activating

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### Issue: Installation is very slow

This is normal! Downloading PyTorch and other ML libraries takes time. On slow connections, it might take 15-30 minutes.

### Issue: "RuntimeError: CUDA out of memory"

**Solution:** Use CPU instead or a smaller model
```python
detector = EmotionCircuitDetector(
    model_name="gpt2-small",
    device="cpu"  # Force CPU usage
)
```

### Issue: Jupyter notebook won't start

Install Jupyter:
```bash
pip install jupyter
jupyter notebook
```

---

## Advanced: Manual Installation

If the automated setup doesn't work, here's how to do it manually:

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Install other requirements
pip install -r requirements.txt

# 6. Test installation
python -c "import transformer_lens; print('Success!')"
```

## System Requirements

### Minimum:
- Python 3.9+
- 4GB RAM
- 5GB disk space
- CPU (will be slow but works)

### Recommended:
- Python 3.10+
- 16GB RAM
- 10GB disk space
- NVIDIA GPU with 8GB+ VRAM (much faster)

### For Large Models:
- 32GB+ RAM
- NVIDIA GPU with 24GB+ VRAM
- 50GB+ disk space

---

**You're all set! Run `python run_interactive.py` to get started! 🚀**
