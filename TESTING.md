# Testing Guide

This document explains how to test the LLM Interpretability Toolkit to ensure everything works correctly.

## Quick Start

### Run All Tests
```bash
python run_tests.py
```

This will run the comprehensive test suite that checks:
- ✅ All imports work
- ✅ Model loading works
- ✅ Hook API works correctly
- ✅ Emotion circuit detection works
- ✅ Deceptive alignment detection works
- ✅ Power-seeking detection works

**Expected time:** 5-10 minutes (includes model download on first run)

### Run Individual Tests

```bash
# Run just the basic test suite
python tests/test_basic.py

# Test just the hook API
python test_hooks.py
```

## Test Files

### `tests/test_basic.py`
Comprehensive test suite covering all major functionality:
- **test_imports()**: Verifies all modules can be imported
- **test_model_loading()**: Tests GPT-2 small model loading
- **test_hook_api()**: Tests TransformerLens hook API
- **test_emotion_circuit_detector()**: Tests emotion circuit discovery
- **test_deceptive_alignment_detector()**: Tests deception detection
- **test_power_seeking_detector()**: Tests power-seeking detection

### `test_hooks.py`
Low-level test of the TransformerLens hook API to ensure we're using it correctly:
- Tests `run_with_hooks()`
- Tests `hooks()` context manager
- Tests `add_hook()` and `reset_hooks()`
- Tests generation with hooks

## What Gets Tested

### 1. Import Tests
Ensures all modules can be imported without errors:
```python
from src.circuit_discovery import CircuitFinder, Circuit, CircuitValidator
from src.behavior_detection import EmotionCircuitDetector
from src.behavior_detection import DeceptiveAlignmentDetector
from src.behavior_detection import PowerSeekingDetector
```

### 2. Model Loading Tests
Verifies that:
- GPT-2 small can be loaded
- Tokenization works
- Forward passes work
- Model is on correct device (CPU/GPU)

### 3. Hook API Tests
Critical tests for the hook functionality:
- `run_with_hooks()` works for single forward passes
- `hooks()` context manager works for temporary hooks
- `generate()` works with hooks active
- Hooks are properly cleaned up after use

### 4. Circuit Discovery Tests
End-to-end tests for each detector:
- Dataset preparation (including padding)
- Circuit discovery with minimal examples
- Circuit component analysis
- Emotion modulation (for emotion circuits)

## Common Test Failures

### Import Errors
**Error:** `ModuleNotFoundError: No module named 'torch'`
**Fix:** Install dependencies with `pip install -r requirements.txt`

### Model Download Fails
**Error:** Network errors during model download
**Fix:** 
- Check internet connection
- Ensure you have ~2GB free disk space
- Try again (downloads are cached)

### CUDA/GPU Errors
**Error:** `CUDA out of memory`
**Fix:** Tests use CPU by default, but if you modified them:
```python
detector = EmotionCircuitDetector(model_name="gpt2-small", device="cpu")
```

### Hook API Errors
**Error:** `'NoneType' object has no attribute 'remove'`
**Fix:** This was a bug that's been fixed. Make sure you have the latest code:
```bash
git pull origin main
```

## Adding New Tests

When adding new functionality, add tests to `tests/test_basic.py`:

```python
def test_my_new_feature():
    """Test description."""
    print("\nTesting my new feature...")
    
    try:
        # Your test code here
        from src.my_module import MyClass
        obj = MyClass()
        result = obj.do_something()
        
        print(f"  ✅ Feature works: {result}")
        return True
    except Exception as e:
        print(f"  ❌ Feature failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Add to the tests list in main():
tests = [
    # ... existing tests ...
    ("My New Feature", test_my_new_feature),
]
```

## Testing Before Commits

**Always run tests before committing changes:**

```bash
# 1. Run the test suite
python run_tests.py

# 2. Check output for failures
#    All tests should show ✅ PASS

# 3. If any tests fail, fix them before committing

# 4. Once all pass, commit:
git add -A
git commit -m "Your commit message"
git push
```

## Continuous Testing

For development, you can run tests in watch mode:

```bash
# Install pytest-watch (optional)
pip install pytest-watch

# Run tests on file changes
ptw tests/
```

## Test Coverage

Our test suite aims for:
- ✅ 100% import coverage
- ✅ 100% API surface coverage
- ✅ Core functionality coverage
- ⚠️  Edge cases coverage (ongoing)

## Debugging Failed Tests

### Enable Verbose Output
All test functions print detailed output. Look for the ❌ markers to see what failed.

### Check Error Logs
Some tests save error logs:
- `error_log.txt` - General errors
- `demo_error.txt` - Demo errors
- `deception_error.txt` - Deception detection errors
- `power_seeking_error.txt` - Power-seeking detection errors

### Run Tests Individually
```bash
# Run just one test
python -c "from tests.test_basic import test_hook_api; test_hook_api()"
```

### Check GPU/CPU
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Performance Benchmarks

Expected test times on modern hardware:

| Test                    | CPU Time   | GPU Time  |
|-------------------------|-----------|-----------|
| test_imports           | < 1s      | < 1s      |
| test_model_loading     | 10-30s    | 5-15s     |
| test_hook_api          | 5-10s     | 2-5s      |
| test_emotion_detector  | 60-120s   | 20-40s    |
| test_deception_detector| 60-120s   | 20-40s    |
| test_power_detector    | 60-120s   | 20-40s    |
| **TOTAL**              | **5-8min**| **2-4min**|

First run includes model download (~500MB), which adds 1-5 minutes depending on internet speed.

## Getting Help

If tests fail and you can't figure out why:

1. Check `TROUBLESHOOTING.md`
2. Run `python check_installation.py`
3. Check the error logs mentioned above
4. Create a GitHub issue with:
   - Which test failed
   - Full error output
   - Your system info (`python --version`, OS, etc.)

## Test Philosophy

We follow Uncle Bob's testing principles:
- **F.I.R.S.T.** - Fast, Independent, Repeatable, Self-validating, Timely
- Tests should be readable and maintainable
- Each test should test one thing
- Tests should fail for only one reason
- Tests should provide clear error messages

Happy testing! 🧪
