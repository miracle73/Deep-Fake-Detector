
#!/usr/bin/env python3
"""
Comprehensive test runner for deepfake detection project
"""

import sys
import os
from pathlib import Path
import unittest
import time
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_all_tests(verbose: bool = True, fast_mode: bool = False):
    """Run all project tests"""
    
    print("🧪 DEEPFAKE DETECTOR TEST SUITE")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if fast_mode:
        print("⚡ Running in FAST MODE (skipping slow tests)")
    
    # Discover and run tests
    test_dir = project_root / "tests"
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    start_dir = str(test_dir)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Count tests
    test_count = suite.countTestCases()
    print(f"📊 Found {test_count} tests")
    
    # Run tests
    start_time = time.time()
    
    if verbose:
        runner = unittest.TextTestRunner(
            verbosity=2,
            stream=sys.stdout,
            buffer=True
        )
    else:
        runner = unittest.TextTestRunner(
            verbosity=1,
            stream=sys.stdout,
            buffer=True
        )
    
    result = runner.run(suite)
    
    # Print summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("📈 TEST SUMMARY")
    print("=" * 50)
    
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Print failures
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
            if verbose:
                print(f"    {traceback}")
    
    # Print errors
    if result.errors:
        print(f"\n⚠️ ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
            if verbose:
                print(f"    {traceback}")
    
    # Overall result
    success = result.wasSuccessful()
    if success:
        print(f"\n✅ ALL TESTS PASSED! 🎉")
    else:
        print(f"\n❌ SOME TESTS FAILED")
    
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success

def run_specific_test_module(module_name: str, verbose: bool = True):
    """Run tests from a specific module"""
    print(f"🧪 Running tests from {module_name}")
    print("=" * 50)
    
    try:
        # Import and run specific test module
        if module_name == "models":
            from tests.test_models import run_tests
            result = run_tests()
        elif module_name == "data":
            from tests.test_data import run_data_tests
            result = run_data_tests()
        else:
            print(f"❌ Unknown test module: {module_name}")
            return False
        
        return result.wasSuccessful()
    
    except ImportError as e:
        print(f"❌ Failed to import test module {module_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def check_test_environment():
    """Check if test environment is properly set up"""
    print("🔍 Checking test environment...")
    
    requirements = {
        "torch": "PyTorch",
        "torchvision": "TorchVision", 
        "PIL": "Pillow",
        "numpy": "NumPy",
        "sklearn": "scikit-learn"
    }
    
    missing = []
    
    for module, name in requirements.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (missing)")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing)}")
        print("   Install with: pip install torch torchvision pillow numpy scikit-learn")
        return False
    
    print("\n✅ Test environment ready!")
    return True

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run deepfake detector tests')
    parser.add_argument('--module', type=str, choices=['models', 'data'], 
                        help='Run tests for specific module only')
    parser.add_argument('--fast', action='store_true',
                        help='Run in fast mode (skip slow tests)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--check-env', action='store_true',
                        help='Check test environment and exit')
    
    args = parser.parse_args()
    
    # Check environment if requested
    if args.check_env:
        env_ok = check_test_environment()
        sys.exit(0 if env_ok else 1)
    
    # Check environment before running tests
    if not check_test_environment():
        print("\n❌ Test environment not ready. Use --check-env for details.")
        sys.exit(1)
    
    verbose = not args.quiet
    
    try:
        if args.module:
            # Run specific module tests
            success = run_specific_test_module(args.module, verbose=verbose)
        else:
            # Run all tests
            success = run_all_tests(verbose=verbose, fast_mode=args.fast)
        
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()