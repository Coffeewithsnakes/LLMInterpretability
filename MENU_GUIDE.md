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
  5. 📚 Open Tutorial Notebooks
  6. 🌐 Launch Web Interface
  7. 🧪 Testing & Diagnostics
  8. ℹ️  Help & Documentation
  0. Back/Exit
----------------------------------------------------------------------
```

## Testing & Diagnostics Menu (Option 7)

Choose option **7** from the main menu to access:

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

### I just installed, want to verify it works:
```
python run_interactive.py
→ Option 7 (Testing & Diagnostics)
→ Option 1 (Quick Sanity Check)
```

### I made changes and want to test thoroughly:
```
python run_interactive.py
→ Option 7 (Testing & Diagnostics)
→ Option 2 (Full Test Suite)
```

### I'm getting hook-related errors:
```
python run_interactive.py
→ Option 7 (Testing & Diagnostics)
→ Option 3 (Hook API Tests)
```

### I'm getting import errors:
```
python run_interactive.py
→ Option 7 (Testing & Diagnostics)
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
