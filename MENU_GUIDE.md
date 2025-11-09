# Interactive Menu Guide

## Main Menu

When you run `python run_interactive.py`, you'll see:

```
======================================================================
  🔍 LLM Interpretability Toolkit - Interactive Menu
======================================================================

Welcome! This interactive tool will guide you through:
  • Discovering circuits in AI models
  • Detecting deceptive alignment
  • Finding power-seeking behaviors
  • And more!

No coding experience needed - just follow the prompts!

Main Menu - What would you like to do?
----------------------------------------------------------------------
  1. 🎭 Discover Emotion Circuits (Beginner-Friendly)
  2. 🕵️  Detect Deceptive Alignment
  3. ⚡ Find Power-Seeking Behaviors
  4. 🚀 Quick Demo (3 minutes)
  5. 🔬 Research: Reproduce Paper Results (99%+ Accuracy)
  6. 📚 Open Tutorial Notebooks
  7. 🌐 Launch Web Interface
  8. 🧪 Testing & Diagnostics
  9. ⚙️  Settings (Change Model)
  10. ℹ️  Help & Documentation
  0. Back/Exit
----------------------------------------------------------------------
```

## Research Menu (Option 5)

Choose option **5** from the main menu to access research-grade paper reproduction:

```
======================================================================
  🔍 LLM Interpretability Toolkit - Interactive Menu
======================================================================

🔬 RESEARCH: REPRODUCE PAPER RESULTS

Reproduce findings from 'Do LLMs Feel?' (arxiv.org/abs/2510.11328)
Target: 99.65% emotion control accuracy

This uses advanced 7-stage pipeline with:
  • 600+ carefully crafted emotion prompts
  • 392 MLP neurons + 168 attention heads
  • Validation against paper benchmarks

What would you like to do?
----------------------------------------------------------------------
  1. 🎯 Single Emotion Reproduction (10-15 min)
  2. 🎨 Full Paper: All 6 Emotions (60-90 min)
  3. ⚡ Quick Research Demo (5 min)
  4. 📄 View Research Documentation
  5. 🔄 Back to Main Menu
  0. Back/Exit
----------------------------------------------------------------------
```

### Option 1: Single Emotion Reproduction

**What it does:** Runs the complete 7-stage pipeline for one emotion
**Time:** 10-15 minutes
**When to use:** Testing research features, reproducing specific emotion
**Target accuracy:** 99%+

**Seven-Stage Pipeline:**
1. Prompt-based emotion elicitation (50 emotion + 50 neutral prompts)
2. Emotion direction extraction from residual streams
3. Steering-based generation validation
4. Local component identification (392 MLP neurons + 168 attention heads)
5. Emotion difference vector computation
6. Global circuit integration
7. Circuit-based emotion generation

**Emotions available:**
- Happiness
- Sadness
- Anger
- Fear
- Disgust
- Surprise

**Output Example:**
```
🎉 REPRODUCTION COMPLETE!
✅ Circuit discovered: 24 components
✅ Validation accuracy: 99.20%

📊 Benchmark vs Paper:
   Our accuracy:   99.20%
   Paper accuracy: 99.41%
   Difference:     -0.21%
   Status:         EXCELLENT

📝 Sample Generated Texts:
1. I feel absolutely wonderful today! Everything is going so well...
2. This is such a joyful moment, I can't help but smile...
3. What an amazing experience! I'm filled with happiness...

💾 Results saved to: results_happiness_gpt2-medium.txt
```

### Option 2: Full Paper Reproduction

**What it does:** Reproduces all 6 emotions from the paper
**Time:** 60-90 minutes
**When to use:** Complete validation, research publication
**Target accuracy:** 99%+ across all emotions

**Processes:**
- Happiness, Sadness, Anger, Fear, Disgust, Surprise
- 30 prompts per emotion (faster) or 50 for max accuracy
- Individual validation for each emotion
- Overall benchmark report

**Output Example:**
```
🎉 FULL PAPER REPRODUCTION COMPLETE!

HAPPINESS:
  Accuracy: 99.20%
  Components: 24
  Status: EXCELLENT

SADNESS:
  Accuracy: 98.85%
  Components: 22
  Status: EXCELLENT

[... results for all 6 emotions ...]

💾 Comprehensive results saved to: full_reproduction_gpt2-medium.txt
```

### Option 3: Quick Research Demo

**What it does:** Fast demo with reduced parameters
**Time:** ~5 minutes
**When to use:** Quick validation, learning the system
**Parameters:** 20 prompts, 200 MLP neurons, 100 attention heads

**Trade-off:**
- ✅ Fast to run
- ✅ Shows full pipeline
- ⚠️  Lower accuracy than full reproduction (~85-95%)

### Option 4: View Research Documentation

**What it does:** Displays research documentation overview
**Time:** Instant
**Shows:**
- RESEARCH_FEATURES.md overview
- Command-line script usage
- Model recommendations
- Paper link and key concepts

### Model Recommendations

The research menu shows model quality ratings:

**EXCELLENT Models:**
- ✅ Llama-3.2-3B (paper's model) - 99.65% target
- ✅ Pythia-1B / Pythia-1.4B - Research-grade
- ✅ GPT-2 Large / XL - 95-99% accuracy

**GOOD Models:**
- ⚠️  GPT-2 Medium - 90-95% accuracy (still useful)

**Not Recommended:**
- ❌ GPT-2 Small - Too small for research-grade results

**Tip:** Use Settings menu (Option 9) to change your model!

## Settings Menu (Option 9)

Choose option **9** to configure your model and view system info:

```
⚙️  SETTINGS

Current model: gpt2-medium

Settings Menu
----------------------------------------------------------------------
  1. 🔄 Change Model
  2. 📊 View Model Information
  3. 💻 System Info (GPU/CUDA Status)
  0. Back/Exit
----------------------------------------------------------------------
```

### Option 1: Change Model

**Interactive model selector** with full specifications:

```
Available models (sorted by size):

[1] gpt2-small (124M) - Fast & Light
    VRAM: 2-3GB | Speed: ⚡⚡⚡ | Quality: ⭐⭐
    Best for: Quick testing and demos

[2] gpt2-medium (355M) - Balanced ⭐
    VRAM: 3-4GB | Speed: ⚡⚡ | Quality: ⭐⭐⭐
    Best for: Most use cases, 3070 sweet spot

[3] gpt2-large (774M) - High Quality
    VRAM: 5-6GB | Speed: ⚡ | Quality: ⭐⭐⭐⭐
    Best for: Research-grade results

[... more models ...]

[7] Llama-3.2-3B - PAPER'S MODEL ⭐⭐
    VRAM: 6-7GB | Speed: ⚡ | Quality: ⭐⭐⭐⭐⭐
    Best for: 99.65% accuracy (requires HF token)

Your GPU: NVIDIA GeForce RTX 3070 (8GB VRAM)
```

**Features:**
- No memorization needed - all specs shown
- GPU compatibility warnings
- Saves preference to `.model_config`
- Used by all features automatically

### Option 3: System Info

**View GPU/CUDA status:**

```
💻 SYSTEM INFORMATION

🖥️  Platform: Linux
📦 Python: 3.10.12

CUDA & GPU Status:
  ✅ CUDA Available: Yes
  ✅ CUDA Version: 12.1
  ✅ GPU Count: 1

  GPU 0: NVIDIA GeForce RTX 3070
    • Total VRAM: 8.00 GB
    • VRAM Used: 0.45 GB
    • VRAM Free: 7.55 GB
    • Utilization: 5%

Model Compatibility:
  ✅ gpt2-small will fit! (2GB needed, 8.0GB available)
  ✅ gpt2-medium will fit! (4GB needed, 8.0GB available)
  ✅ gpt2-large will fit! (6GB needed, 8.0GB available)
```

## Testing & Diagnostics Menu (Option 8)

Choose option **8** from the main menu to access:

```
======================================================================
  🔍 LLM Interpretability Toolkit - Interactive Menu
======================================================================

🧪 TESTING & DIAGNOSTICS

These tools help you verify everything is working correctly.
Run these if you're having issues or after making changes.

Testing Menu - What would you like to test?
----------------------------------------------------------------------
  1. ⚡ Quick Sanity Check (30 seconds) - Test the bug fix
  2. 🧪 Full Test Suite (5-10 minutes) - Comprehensive testing
  3. 🔌 Hook API Tests - Low-level TransformerLens validation
  4. 🔍 Installation Check - Verify all dependencies
  5. 📊 View Testing Documentation
  0. Back/Exit
----------------------------------------------------------------------
```

### Option 1: Quick Sanity Check

**What it does:** Tests the specific hook bug fix that was causing crashes  
**Time:** ~30 seconds  
**When to use:** After installing, after updates, or if you had errors  
**Runs:** `quick_sanity_check.py`

Tests:
- ✅ Imports work
- ✅ Model loads
- ✅ Test circuit creation
- ✅ Emotion modulation with hooks (the bug fix!)
- ✅ Different intensity levels

**Output Example:**
```
[1/5] Testing imports...
✅ Imports successful

[2/5] Loading model...
✅ Model loaded

[3/5] Creating minimal test circuit...
✅ Test circuit created with 2 components

[4/5] Testing emotion modulation (THE CRITICAL TEST)...
✅ Modulation successful! Generated: 'I feel great about this'

[5/5] Testing with different intensities...
✅ All intensity levels work

🎉 ALL CHECKS PASSED!
```

### Option 2: Full Test Suite

**What it does:** Comprehensive testing of all functionality  
**Time:** 5-10 minutes (first run takes longer due to model download)  
**When to use:** Before committing changes, thorough validation  
**Runs:** `tests/test_basic.py`

Tests:
- ✅ All imports
- ✅ Model loading (GPT-2 small)
- ✅ Hook API (4 different methods)
- ✅ EmotionCircuitDetector (end-to-end)
- ✅ DeceptiveAlignmentDetector (end-to-end)
- ✅ PowerSeekingDetector (end-to-end)

**Output Example:**
```
Testing imports...
  ✅ torch 2.1.0
  ✅ transformer_lens
  ✅ circuit_discovery
  ✅ behavior_detection.EmotionCircuitDetector
  ✅ behavior_detection.DeceptiveAlignmentDetector
  ✅ behavior_detection.PowerSeekingDetector

Testing model loading...
  Loading gpt2-small (this may take a minute)...
  ✅ Model loaded: 12 layers, 12 heads
  ✅ Tokenization works: torch.Size([1, 3])
  ✅ Forward pass works: torch.Size([1, 3, 50257])

[... more tests ...]

======================================================================
  Test Summary
======================================================================
  ✅ PASS: Imports
  ✅ PASS: Model Loading
  ✅ PASS: Hook API
  ✅ PASS: EmotionCircuitDetector
  ✅ PASS: DeceptiveAlignmentDetector
  ✅ PASS: PowerSeekingDetector

  6/6 tests passed

  🎉 All tests passed!
```

### Option 3: Hook API Tests

**What it does:** Low-level validation of TransformerLens hook API  
**Time:** ~1 minute  
**When to use:** Debugging hook-related issues, understanding API  
**Runs:** `test_hooks.py`

Tests:
- ✅ `run_with_hooks()` for single forward passes
- ✅ `hooks()` context manager
- ✅ `add_hook()` and `reset_hooks()`
- ✅ Generation with hooks active

**Output Example:**
```
Loading model...
Model loaded!

Test 1: Checking add_hook return value
  Hook called on blocks.0.attn.hook_z
  add_hook returned: None
  Type: <class 'NoneType'>

Test 2: Running forward pass with hook
  Forward pass completed

[... more tests ...]

Test 6: Using hook context
  Generation with context: Hello world, this is a test
  Hooks automatically cleaned up

✅ All tests complete!
```

### Option 4: Installation Check

**What it does:** Verifies all dependencies are installed  
**Time:** <10 seconds  
**When to use:** After setup, troubleshooting import errors  
**Runs:** Built-in dependency checker

Checks:
- ✅ PyTorch
- ✅ TransformerLens
- ✅ NumPy
- ✅ Matplotlib
- ✅ Other optional dependencies

### Option 5: View Testing Documentation

**What it does:** Displays the TESTING.md documentation  
**Time:** Instant  
**When to use:** Learning about tests, troubleshooting  
**Shows:** First 50 lines of TESTING.md

## Quick Navigation

### I want to reproduce the research paper:
```
python run_interactive.py
→ Option 5 (Research: Reproduce Paper Results)
→ Option 1 (Single Emotion) or Option 2 (Full Paper)
```

### I want to test research features quickly:
```
python run_interactive.py
→ Option 5 (Research: Reproduce Paper Results)
→ Option 3 (Quick Research Demo)
```

### I want to change my model:
```
python run_interactive.py
→ Option 9 (Settings)
→ Option 1 (Change Model)
```

### I want to check if I'm using my GPU:
```
python run_interactive.py
→ Option 9 (Settings)
→ Option 3 (System Info)
```

### I just installed, want to verify it works:
```
python run_interactive.py
→ Option 8 (Testing & Diagnostics)
→ Option 1 (Quick Sanity Check)
```

### I made changes and want to test thoroughly:
```
python run_interactive.py
→ Option 8 (Testing & Diagnostics)
→ Option 2 (Full Test Suite)
```

### I'm getting hook-related errors:
```
python run_interactive.py
→ Option 8 (Testing & Diagnostics)
→ Option 3 (Hook API Tests)
```

### I'm getting import errors:
```
python run_interactive.py
→ Option 8 (Testing & Diagnostics)
→ Option 4 (Installation Check)
```

## Command-Line Alternatives

If you prefer to run tests from the command line directly:

```bash
# Quick sanity check
python quick_sanity_check.py

# Full test suite
python tests/test_basic.py
# or
python run_tests.py

# Hook API tests
python test_hooks.py

# Installation check
python check_installation.py
```

## Tips

1. **Run Quick Sanity Check first** - It's fast and catches most issues
2. **Full Test Suite before commits** - Ensures you haven't broken anything
3. **Use Testing Menu for convenience** - No need to remember script names
4. **Check the output carefully** - Tests provide detailed error messages
5. **First run takes longer** - Model downloads ~500MB and is cached

## Troubleshooting

### "ModuleNotFoundError"
→ Run Option 4 (Installation Check) to see what's missing  
→ Then: `pip install -r requirements.txt`

### "CUDA out of memory"
→ Tests use CPU by default, this shouldn't happen  
→ If it does, check that you didn't modify test files to use GPU

### "Model download fails"
→ Check internet connection  
→ Ensure ~2GB free disk space  
→ Try again (downloads resume from where they stopped)

### "Tests pass but demo fails"
→ Run Option 3 (Hook API Tests) to verify hook implementation  
→ Check error logs: `demo_error.txt`

## What Changed Recently

**Before this update:**
- Had to remember: `python quick_sanity_check.py` vs `python test_hooks.py` vs `python tests/test_basic.py`
- No central place for testing
- Easy to forget which test does what

**After this update:**
- Everything in one menu: `python run_interactive.py` → Option 7
- Clear descriptions of each test
- Expected time for each test shown
- Confirmation for long-running tests
- Easy to navigate between tests

## Philosophy

Following Uncle Bob's testing principles:
- **F**ast - Quick sanity check runs in 30s
- **I**ndependent - Each test runs standalone
- **R**epeatable - Same results every time
- **S**elf-validating - Clear ✅/❌ feedback
- **T**imely - Run tests when you need them

Happy testing! 🧪
