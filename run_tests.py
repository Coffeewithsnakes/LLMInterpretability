#!/usr/bin/env python3
"""
Test Runner for LLM Interpretability Toolkit

This script runs all tests and provides clear feedback on what works and what doesn't.
Run this before committing any changes to ensure nothing is broken.

Usage:
    python run_tests.py                  # Run all tests
    python run_tests.py --quick          # Run only quick tests (no model loading)
    python run_tests.py --integration    # Run only integration tests
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run LLM Interpretability Toolkit tests")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests (no model loading)")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    args = parser.parse_args()

    print("=" * 70)
    print("  LLM Interpretability Toolkit - Test Runner")
    print("=" * 70)

    if args.quick:
        print("\n🏃 Running QUICK tests (imports and basic checks only)...")
        print("This should take < 10 seconds\n")
    elif args.integration:
        print("\n🔬 Running INTEGRATION tests (full end-to-end)...")
        print("This will take 5-10 minutes\n")
    else:
        print("\n🧪 Running ALL tests...")
        print("This will take 5-10 minutes\n")

    # Import and run tests
    try:
        from tests import test_basic
        result = test_basic.main()
        sys.exit(result)
    except ImportError:
        print("❌ Could not import test module!")
        print("\n Make sure you're in the project root directory")
        print("   cd /path/to/LLMInterpretability")
        print("   python run_tests.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
