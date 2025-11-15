"""
Tests for embodied.jax.internal - JAX internal utilities

Coverage goal: 90% (from 14.04%)

Tests cover:
- get_named_axes: Get named JAX axes
- get_data_axes: Get data axes
- is_multihost: Check multihost configuration
- mesh: Create JAX mesh from devices
- grouped_ckpt_fns: Checkpoint function grouping
- ckpt_fn: Checkpoint function creation
- setup: JAX configuration setup

Note: Some functions (device_put, to_local/global) require complex
JAX distributed setup and are challenging to test in isolation.
"""

import math
import os
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from embodied.jax import internal


class TestGetNamedAxes:
    """Test named axes retrieval"""

    @patch("embodied.jax.internal.jax.lax.axis_index")
    def test_get_named_axes_none(self, mock_axis_index):
        """Test get_named_axes when no axes are defined"""
        mock_axis_index.side_effect = NameError("No axis")

        axes = internal.get_named_axes()

        assert axes == []

    @patch("embodied.jax.internal.jax.lax.axis_index")
    def test_get_named_axes_single(self, mock_axis_index):
        """Test get_named_axes with single axis"""

        def axis_check(name):
            if name == "a":
                return 0
            raise NameError("No axis")

        mock_axis_index.side_effect = axis_check

        axes = internal.get_named_axes()

        assert "a" in axes
        assert len(axes) == 1

    @patch("embodied.jax.internal.jax.lax.axis_index")
    def test_get_named_axes_multiple(self, mock_axis_index):
        """Test get_named_axes with multiple axes"""

        def axis_check(name):
            if name in ("a", "b", "c"):
                return ord(name) - ord("a")
            raise NameError("No axis")

        mock_axis_index.side_effect = axis_check

        axes = internal.get_named_axes()

        assert "a" in axes
        assert "b" in axes
        assert "c" in axes


class TestGetDataAxes:
    """Test data axes retrieval"""

    @patch("embodied.jax.internal.jax.lax.axis_index")
    def test_get_data_axes_none(self, mock_axis_index):
        """Test get_data_axes when no axes defined"""
        mock_axis_index.side_effect = NameError("No axis")

        axes = internal.get_data_axes()

        assert axes == ()

    @patch("embodied.jax.internal.jax.lax.axis_index")
    def test_get_data_axes_both(self, mock_axis_index):
        """Test get_data_axes when both 'd' and 'f' exist"""

        def axis_check(name):
            if name in ("d", "f"):
                return 0
            raise NameError("No axis")

        mock_axis_index.side_effect = axis_check

        axes = internal.get_data_axes()

        assert axes == ("d", "f")

    @patch("embodied.jax.internal.jax.lax.axis_index")
    def test_get_data_axes_only_d(self, mock_axis_index):
        """Test get_data_axes when only 'd' exists"""

        def axis_check(name):
            if name == "d":
                return 0
            raise NameError("No axis")

        mock_axis_index.side_effect = axis_check

        axes = internal.get_data_axes()

        # Should return empty tuple if both aren't present
        assert axes == ()


class TestIsMultihost:
    """Test multihost detection"""

    @patch("embodied.jax.internal.jax.process_count")
    def test_is_multihost_single(self, mock_count):
        """Test is_multihost with single process"""
        mock_count.return_value = 1

        assert internal.is_multihost() is False

    @patch("embodied.jax.internal.jax.process_count")
    def test_is_multihost_multiple(self, mock_count):
        """Test is_multihost with multiple processes"""
        mock_count.return_value = 4

        assert internal.is_multihost() is True

    @patch("embodied.jax.internal.jax.process_count")
    def test_is_multihost_two(self, mock_count):
        """Test is_multihost with exactly two processes"""
        mock_count.return_value = 2

        assert internal.is_multihost() is True


class TestMesh:
    """Test mesh creation from devices"""

    @pytest.mark.skipif(len(jax.devices()) < 2, reason="Requires at least 2 devices")
    def test_mesh_simple_shape(self):
        """Test mesh creation with simple shape"""
        devices = jax.devices()[:2]
        shape = "2"
        names = ("d",)

        mesh = internal.mesh(devices, shape, names)

        assert mesh.shape == {"d": 2}
        assert mesh.axis_names == names
        assert len(mesh.devices.flat) == 2

    @pytest.mark.skipif(len(jax.devices()) < 4, reason="Requires at least 4 devices")
    def test_mesh_2d_shape(self):
        """Test mesh creation with 2D shape"""
        devices = jax.devices()[:4]
        shape = "2,2"
        names = ("d", "m")

        mesh = internal.mesh(devices, shape, names)

        assert mesh.shape == {"d": 2, "m": 2}
        assert mesh.axis_names == names
        assert len(mesh.devices.flat) == 4

    @pytest.mark.skipif(len(jax.devices()) < 8, reason="Requires at least 8 devices")
    def test_mesh_with_minus_one(self):
        """Test mesh creation with -1 in shape (auto-infer dimension)"""
        devices = jax.devices()[:8]
        shape = "2,-1"  # Should auto-compute as 4
        names = ("d", "m")

        mesh = internal.mesh(devices, shape, names)

        assert mesh.shape == {"d": 2, "m": 4}
        assert len(mesh.devices.flat) == 8

    def test_mesh_single_device(self):
        """Test mesh creation with single device"""
        devices = jax.devices()[:1]
        shape = "1"
        names = ("d",)

        mesh = internal.mesh(devices, shape, names)

        assert mesh.shape == {"d": 1}
        assert len(mesh.devices.flat) == 1

    def test_mesh_invalid_shape_raises(self):
        """Test mesh creation raises on invalid shape"""
        devices = jax.devices()[:3]
        shape = "2,2"  # Requires 4 devices but only 3 provided
        names = ("d", "m")

        with pytest.raises(AssertionError):
            internal.mesh(devices, shape, names)

    def test_mesh_multiple_minus_one_raises(self):
        """Test mesh creation raises with multiple -1 in shape"""
        devices = jax.devices()[:4]
        shape = "-1,-1"  # Not allowed
        names = ("d", "m")

        with pytest.raises(AssertionError):
            internal.mesh(devices, shape, names)


class TestGroupedCkptFns:
    """Test grouped checkpoint function creation"""

    def test_grouped_ckpt_fns_no_chunking(self):
        """Test grouped_ckpt_fns with chunksize <= 0 (no chunking)"""
        # Create simple params with known mesh
        mesh_obj = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh_obj, P())

        params = {
            "layer1/weight": jax.device_put(jnp.ones((10, 20)), sharding),
            "layer1/bias": jax.device_put(jnp.ones(10), sharding),
        }

        result = internal.grouped_ckpt_fns(params, chunksize=0)

        # With chunksize <= 0, should have single group with all params
        assert len(result) == 1
        keys, gather_fn, shard_fn = result[0]
        assert set(keys) == {"layer1/weight", "layer1/bias"}
        assert callable(gather_fn)
        assert callable(shard_fn)

    @pytest.mark.slow
    def test_grouped_ckpt_fns_with_chunking(self):
        """Test grouped_ckpt_fns with small chunksize (creates multiple groups)"""
        mesh_obj = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh_obj, P())

        # Create params with known sizes
        params = {
            "layer1/weight": jax.device_put(
                jnp.ones((100, 200), dtype=jnp.float32), sharding
            ),  # 80KB
            "layer2/weight": jax.device_put(
                jnp.ones((100, 200), dtype=jnp.float32), sharding
            ),  # 80KB
        }

        # Use small chunksize to force multiple groups
        chunksize = 100 * 1024  # 100KB - should create 2 groups

        result = internal.grouped_ckpt_fns(params, chunksize=chunksize)

        # Should create 2 groups (one for each param)
        assert len(result) == 2
        all_keys = []
        for keys, gather_fn, shard_fn in result:
            all_keys.extend(keys)
            assert callable(gather_fn)
            assert callable(shard_fn)
        assert set(all_keys) == {"layer1/weight", "layer2/weight"}


class TestCkptFn:
    """Test checkpoint function creation"""

    def test_ckpt_fn_basic(self):
        """Test ckpt_fn creates gather and shard functions"""
        mesh_obj = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh_obj, P())

        params = {
            "weight": jax.device_put(jnp.ones((10, 20)), sharding),
        }

        gather_fn, shard_fn = internal.ckpt_fn(params, compile=False)

        # Should return lowered functions (not compiled)
        assert hasattr(gather_fn, "compile")
        assert hasattr(shard_fn, "compile")

    def test_ckpt_fn_compiled(self):
        """Test ckpt_fn with compilation"""
        mesh_obj = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh_obj, P())

        params = {
            "weight": jax.device_put(jnp.ones((10, 20)), sharding),
        }

        gather_fn, shard_fn = internal.ckpt_fn(params, compile=True)

        # Should return compiled functions
        assert callable(gather_fn)
        assert callable(shard_fn)

    def test_ckpt_fn_multiple_params(self):
        """Test ckpt_fn with multiple parameters"""
        mesh_obj = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh_obj, P())

        params = {
            "layer1/weight": jax.device_put(jnp.ones((10, 20)), sharding),
            "layer1/bias": jax.device_put(jnp.ones(10), sharding),
            "layer2/weight": jax.device_put(jnp.ones((20, 5)), sharding),
        }

        gather_fn, shard_fn = internal.ckpt_fn(params, compile=False)

        assert hasattr(gather_fn, "compile")
        assert hasattr(shard_fn, "compile")


class TestSetup:
    """Test JAX setup configuration"""

    def test_setup_with_mock_devices(self):
        """Test setup configures mock devices"""
        # Mock JAX config to avoid actually changing global state
        with (
            patch.object(jax.config, "update") as mock_update,
            patch.dict("os.environ", {}, clear=True),
        ):
            internal.setup(mock_devices=4)

            # Should add mock devices flag to XLA_FLAGS
            assert "XLA_FLAGS" in os.environ
            assert "xla_force_host_platform_device_count=4" in os.environ["XLA_FLAGS"]

    def test_setup_with_xladump(self):
        """Test setup configures XLA dump directory"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            dump_dir = f"{tmpdir}/xladump"

            with (
                patch.object(jax.config, "update"),
                patch.dict("os.environ", {}, clear=True),
            ):
                internal.setup(xladump=dump_dir)

                # Should create dump directory and add flags
                import elements

                assert elements.Path(dump_dir).exists()
                assert "XLA_FLAGS" in os.environ
                assert f"xla_dump_to={dump_dir}" in os.environ["XLA_FLAGS"]
                assert "xla_dump_hlo_as_long_text" in os.environ["XLA_FLAGS"]

    def test_setup_with_gpu_platform(self):
        """Test setup adds GPU-specific flags"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            internal.setup(platform="gpu", gpuflags=True)

            # Should add GPU-specific XLA flags
            assert "XLA_FLAGS" in os.environ
            xla_flags = os.environ["XLA_FLAGS"]
            assert "xla_gpu_enable_latency_hiding_scheduler=true" in xla_flags
            assert "xla_gpu_all_gather_combine_threshold_bytes" in xla_flags

    def test_setup_deterministic_mode(self):
        """Test setup configures deterministic mode"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            internal.setup(deterministic=True)

            # Should set TF_CUDNN_DETERMINISTIC and add XLA flag
            assert os.environ.get("TF_CUDNN_DETERMINISTIC") == "1"
            assert "XLA_FLAGS" in os.environ
            assert "xla_gpu_deterministic_ops=true" in os.environ["XLA_FLAGS"]

    def test_setup_prealloc(self):
        """Test setup configures memory preallocation"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            internal.setup(prealloc=True)

            assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "true"

        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            internal.setup(prealloc=False)

            assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"

    def test_setup_debug_mode(self):
        """Test setup configures debug mode"""
        with patch.object(jax.config, "update") as mock_update:
            internal.setup(debug=True, jit=True)

            # Should disable most optimizations in debug mode
            calls = {call[0][0]: call[0][1] for call in mock_update.call_args_list}
            assert calls.get("jax_disable_most_optimizations") is True

    def test_setup_jit_disabled(self):
        """Test setup can disable JIT compilation"""
        with patch.object(jax.config, "update") as mock_update:
            internal.setup(jit=False)

            calls = {call[0][0]: call[0][1] for call in mock_update.call_args_list}
            assert calls.get("jax_disable_jit") is True


# Note: The following functions are challenging to test without full distributed setup:
# - fetch_async(): Requires actual device data and multihost setup
# - device_put(): Requires proper sharding and multihost configuration
# - local_sharding(): Requires mesh with local_mesh
# - to_local() / _to_local(): Requires distributed arrays with addressable shards
# - to_global() / _to_global(): Requires distributed arrays
# - move(): Requires multihost setup and actual data movement
#
# These functions would benefit from integration tests that set up a proper
# JAX distributed environment, but are difficult to unit test in isolation.
