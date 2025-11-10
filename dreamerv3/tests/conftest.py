"""
Pytest configuration for dreamerv3 tests

Configures JAX to work properly in test environment
"""

import jax
import pytest


def pytest_configure(config):
    """Configure JAX for testing"""
    # Allow host-to-device transfers in tests (required for JAX 0.4.33+)
    jax.config.update("jax_transfer_guard", "allow")

    # Use CPU for tests
    jax.config.update("jax_platform_name", "cpu")

    # Disable JIT for easier debugging if needed
    # jax.config.update("jax_disable_jit", True)


@pytest.fixture(autouse=True, scope="function")
def reset_jax_config():
    """Ensure JAX transfer guard is set for each test function"""
    # Reset JAX config before each test to prevent state pollution from agent tests
    jax.config.update("jax_transfer_guard", "allow")
    yield
    # Could also reset after test if needed
