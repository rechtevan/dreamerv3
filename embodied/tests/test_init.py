"""Tests for embodied/__init__.py - Package initialization.

Coverage goal: 100% (from 75%)

Tests cover:
- Package version attribute
- colored_traceback ImportError handling
- Module imports
"""

import sys
from unittest.mock import patch


class TestPackageInit:
    """Test package initialization"""

    def test_version_attribute(self):
        """Test package has __version__ attribute"""
        import embodied

        assert hasattr(embodied, "__version__")
        assert isinstance(embodied.__version__, str)
        # Should be semantic version format
        parts = embodied.__version__.split(".")
        assert len(parts) >= 2  # At least major.minor

    def test_colored_traceback_import_error(self):
        """Test colored_traceback ImportError is handled gracefully"""
        # Simulate colored_traceback not being installed
        with patch.dict("sys.modules", {"colored_traceback": None}):
            # Force reimport to trigger the try/except block
            import importlib

            import embodied

            # Reload to execute the import block again
            importlib.reload(embodied)

            # Should not raise an error
            assert embodied is not None

    def test_core_module_imported(self):
        """Test core module is imported"""
        import embodied

        # Core classes should be available (from embodied.core)
        assert hasattr(embodied, "Agent")
        assert hasattr(embodied, "Env")
        assert hasattr(embodied, "Driver")
        assert hasattr(embodied, "Replay")
        assert hasattr(embodied, "Wrapper")

    def test_submodules_imported(self):
        """Test submodules are imported"""
        import embodied

        # Submodules should be available
        assert hasattr(embodied, "envs")
        assert hasattr(embodied, "jax")
        assert hasattr(embodied, "run")

    def test_import_order_works(self):
        """Test that import order doesn't cause circular import"""
        # This should not raise ImportError
        import embodied

        # Agent base class should be available before jax module
        assert embodied.Agent is not None
        assert embodied.jax is not None
