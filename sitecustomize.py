"""
Coverage.py subprocess support

This file is automatically imported by Python when it starts.
When COVERAGE_PROCESS_START is set, it starts coverage in subprocesses,
allowing multiprocessing code (like driver.py worker processes) to be tracked.

This enables accurate coverage measurement for parallel test execution.
"""

import os


# Only start coverage if explicitly enabled
if "COVERAGE_PROCESS_START" in os.environ:
    try:
        import coverage

        coverage.process_startup()
    except ImportError:
        # Coverage not installed, silently skip
        pass
