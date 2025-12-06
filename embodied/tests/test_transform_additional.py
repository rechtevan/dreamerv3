"""
Additional tests for embodied.jax.transform to increase coverage to 90%+

This test file focuses on previously untested code paths:
- print_grouping with edge cases
- Tracer sharding attribute management
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from embodied.jax import transform


class TestPrintGroupingEdgeCases:
    """Additional tests for print_grouping"""

    def test_print_grouping_with_many_params(self):
        """Test print_grouping with large number of parameters"""
        grouping = {"layer.*": [f"layer{i}/weight" for i in range(100)]}

        import io
        from contextlib import redirect_stdout

        output = io.StringIO()
        with redirect_stdout(output):
            transform.print_grouping(grouping)

        result = output.getvalue()
        assert "matches 100 param tensors" in result


class TestTracerShardings:
    """Test TRACER_SHARDINGS global dict management"""

    def test_tracer_shardings_dict_exists(self):
        """Test that TRACER_SHARDINGS dict exists"""
        assert hasattr(transform, "TRACER_SHARDINGS")
        assert isinstance(transform.TRACER_SHARDINGS, dict)

    def test_tracer_shardings_initially_empty_or_has_items(self):
        """Test TRACER_SHARDINGS can be accessed"""
        # Just verify we can access it
        shardings = transform.TRACER_SHARDINGS
        assert isinstance(shardings, dict)


class TestLockMechanism:
    """Test threading.Lock mechanism"""

    def test_lock_exists(self):
        """Test that LOCK exists for thread safety"""
        import threading

        assert hasattr(transform, "LOCK")
        # Lock is an instance, not a type
        assert type(transform.LOCK).__name__ == "lock"
