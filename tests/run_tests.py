"""Run all test suites for the SPC project.

This script discovers and runs all tests in this tests/ directory,
providing a summary of results.

Test runner script.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py -v                 # Run with verbose output
    python tests/run_tests.py test_graph.py      # Run specific test file
    python tests/run_tests.py -h                 # Show help
"""

import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add parent directories to path so we can import spc module
_TESTS_DIR = Path(__file__).resolve().parent
_ROOT = _TESTS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run_all_tests(verbosity=2):
    """Discover and run all tests in this directory.

    Args:
        verbosity (int): Verbosity level for test output (0, 1, or 2).

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    # Discover all tests matching test_*.py pattern in this directory
    loader = unittest.TestLoader()
    suite = loader.discover(str(_TESTS_DIR), pattern="test_*.py")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all tests for the SPC project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py              Run all tests with normal verbosity
  python run_tests.py -v           Run all tests with verbose output
  python run_tests.py -q           Run all tests quietly
  python run_tests.py test_graph.py   Run only test_graph.py tests
        """,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase verbosity (same as unittest -v)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Decrease verbosity (same as unittest -q)",
    )

    parser.add_argument(
        "test_file",
        nargs="?",
        default=None,
        help="Specific test file to run (e.g., test_graph.py)",
    )

    args = parser.parse_args()

    # Determine verbosity level
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1

    # Run tests
    if args.test_file:
        # Run specific test file
        loader = unittest.TestLoader()
        suite = loader.discover(str(_TESTS_DIR), pattern=args.test_file)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        success = result.wasSuccessful()
    else:
        # Run all tests
        success = run_all_tests(verbosity=verbosity)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
