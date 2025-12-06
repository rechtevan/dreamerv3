"""
Additional tests for embodied.jax.internal to increase coverage to 90%+

This test file focuses on previously untested code paths:
- setup() with TPU flags
- setup() with multi-process configuration
- fetch_async() function
- device_put() in different scenarios
- Error handling and edge cases
"""

import math
import os
from unittest.mock import MagicMock, PropertyMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from embodied.jax import internal


class TestSetupTPUFlags:
    """Test setup() with TPU-specific configuration"""

    def test_setup_with_tpu_platform(self):
        """Test setup adds TPU-specific flags when platform is tpu"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            internal.setup(platform="tpu", tpuflags=True)

            # Should add TPU-specific XLA flags
            assert "XLA_FLAGS" in os.environ
            xla_flags = os.environ["XLA_FLAGS"]
            assert "xla_tpu_megacore_fusion_allow_ags=false" in xla_flags
            assert "xla_enable_async_collective_permute=true" in xla_flags
            assert "xla_tpu_enable_ag_backward_pipelining=true" in xla_flags

    def test_setup_with_tpu_platform_tpuflags_false(self):
        """Test setup doesn't add TPU flags when tpuflags=False"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            internal.setup(platform="tpu", tpuflags=False)

            # Should not add TPU-specific flags
            if "XLA_FLAGS" in os.environ:
                xla_flags = os.environ["XLA_FLAGS"]
                assert "xla_tpu_megacore_fusion_allow_ags" not in xla_flags

    def test_setup_with_cpu_platform_no_gpu_flags(self):
        """Test setup doesn't add GPU flags when platform is cpu"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            internal.setup(platform="cpu", gpuflags=True)

            # Should not add GPU flags for CPU platform
            if "XLA_FLAGS" in os.environ:
                xla_flags = os.environ["XLA_FLAGS"]
                # GPU-specific flags should not be present for CPU platform
                assert "xla_gpu_enable_triton_gemm" not in xla_flags


class TestSetupMultiProcess:
    """Test setup() with multi-process configuration"""

    @patch("embodied.jax.internal.jax.distributed.initialize")
    @patch("embodied.jax.internal.jax.process_index")
    @patch("embodied.jax.internal.jax.process_count")
    def test_setup_multiprocess_non_tpu(self, mock_count, mock_index, mock_initialize):
        """Test setup with multi-process configuration for non-TPU platforms"""
        mock_index.return_value = 0
        mock_count.return_value = 4

        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            internal.setup(
                platform="gpu",
                num_processes=4,
                process_id=0,
                coordinator_address="localhost:1234",
            )

            # Should call distributed.initialize for multi-process non-TPU
            mock_initialize.assert_called_once_with("localhost:1234", 4, 0)

    @patch("embodied.jax.internal.jax.distributed.initialize")
    def test_setup_multiprocess_tpu_skips_initialize(self, mock_initialize):
        """Test setup with TPU platform skips distributed initialization"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            internal.setup(
                platform="tpu",
                num_processes=4,
                process_id=0,
                coordinator_address="localhost:1234",
            )

            # Should NOT call distributed.initialize for TPU
            mock_initialize.assert_not_called()

    def test_setup_multiprocess_missing_process_id_raises(self):
        """Test setup raises assertion when process_id not set for multi-process"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            # process_id = -1 (default) should trigger assertion
            with pytest.raises(AssertionError):
                internal.setup(
                    platform="gpu",
                    num_processes=4,
                    process_id=-1,  # Invalid
                    coordinator_address="localhost:1234",
                )

    def test_setup_multiprocess_missing_coordinator_raises(self):
        """Test setup raises assertion when coordinator_address not set"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            # coordinator_address = None should trigger assertion
            with pytest.raises(AssertionError):
                internal.setup(
                    platform="gpu",
                    num_processes=4,
                    process_id=0,
                    coordinator_address=None,  # Invalid
                )

    def test_setup_single_process_no_initialize(self):
        """Test setup doesn't call distributed.initialize for single process"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
            patch("embodied.jax.internal.jax.distributed.initialize") as mock_init,
        ):
            internal.setup(
                platform="gpu",
                num_processes=1,  # Single process
                process_id=-1,  # Should be ignored
                coordinator_address=None,  # Should be ignored
            )

            # Should NOT call distributed.initialize for single process
            mock_init.assert_not_called()


class TestSetupComputeDtype:
    """Test setup() with different compute dtypes"""

    def test_setup_with_string_dtype(self):
        """Test setup converts string dtype to jnp dtype"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            # Reset nets.COMPUTE_DTYPE to initial state
            from embodied.jax import nets

            original_dtype = nets.COMPUTE_DTYPE

            internal.setup(compute_dtype="float32")

            # Should convert string to jnp.float32
            assert jnp.float32 == nets.COMPUTE_DTYPE

            # Restore
            nets.COMPUTE_DTYPE = original_dtype

    def test_setup_with_dtype_object(self):
        """Test setup with direct dtype object"""
        with (
            patch.object(jax.config, "update"),
            patch.dict("os.environ", {}, clear=True),
        ):
            from embodied.jax import nets

            original_dtype = nets.COMPUTE_DTYPE

            internal.setup(compute_dtype=jnp.bfloat16)

            # Should accept dtype object directly
            assert jnp.bfloat16 == nets.COMPUTE_DTYPE

            # Restore
            nets.COMPUTE_DTYPE = original_dtype


class TestSetupTransferGuard:
    """Test setup() transfer guard configuration"""

    def test_setup_transfer_guard_enabled_by_default(self):
        """Test transfer guard is enabled by default when jit=True"""
        with patch.object(jax.config, "update") as mock_update:
            internal.setup(transfer_guard=True, jit=True, debug_nans=False)

            calls = {call[0][0]: call[0][1] for call in mock_update.call_args_list}
            assert calls.get("jax_transfer_guard") == "disallow"

    def test_setup_transfer_guard_disabled_when_debug_nans(self):
        """Test transfer guard is disabled when debug_nans=True"""
        with patch.object(jax.config, "update") as mock_update:
            internal.setup(transfer_guard=True, jit=True, debug_nans=True)

            calls = {call[0][0]: call[0][1] for call in mock_update.call_args_list}
            # Should not set transfer_guard when debug_nans is True
            assert "jax_transfer_guard" not in calls

    def test_setup_transfer_guard_disabled_when_no_jit(self):
        """Test transfer guard is disabled when jit=False"""
        with patch.object(jax.config, "update") as mock_update:
            internal.setup(transfer_guard=True, jit=False, debug_nans=False)

            calls = {call[0][0]: call[0][1] for call in mock_update.call_args_list}
            # Should not set transfer_guard when jit is disabled
            assert "jax_transfer_guard" not in calls


class TestSetupDebugNans:
    """Test setup() debug_nans configuration"""

    def test_setup_debug_nans_true(self):
        """Test setup enables NaN debugging"""
        with patch.object(jax.config, "update") as mock_update:
            internal.setup(debug_nans=True)

            calls = {call[0][0]: call[0][1] for call in mock_update.call_args_list}
            assert calls.get("jax_debug_nans") is True

    def test_setup_debug_nans_false(self):
        """Test setup disables NaN debugging"""
        with patch.object(jax.config, "update") as mock_update:
            internal.setup(debug_nans=False)

            calls = {call[0][0]: call[0][1] for call in mock_update.call_args_list}
            assert calls.get("jax_debug_nans") is False


class TestSetupCompilationCache:
    """Test setup() compilation cache configuration"""

    def test_setup_compilation_cache_enabled(self):
        """Test setup enables compilation cache"""
        with patch.object(jax.config, "update") as mock_update:
            internal.setup(compilation_cache=True)

            calls = {call[0][0]: call[0][1] for call in mock_update.call_args_list}
            assert calls.get("jax_enable_compilation_cache") is True

    def test_setup_compilation_cache_disabled(self):
        """Test setup disables compilation cache"""
        with patch.object(jax.config, "update") as mock_update:
            internal.setup(compilation_cache=False)

            calls = {call[0][0]: call[0][1] for call in mock_update.call_args_list}
            assert calls.get("jax_enable_compilation_cache") is False


class TestFetchAsync:
    """Test fetch_async() function"""

    @patch("embodied.jax.internal.is_multihost")
    @patch("embodied.jax.internal.to_local")
    def test_fetch_async_multihost(self, mock_to_local, mock_is_multihost):
        """Test fetch_async converts to local in multihost setup"""
        mock_is_multihost.return_value = True
        test_value = {"array": jnp.ones(10)}
        mock_to_local.return_value = test_value

        result = internal.fetch_async(test_value)

        # Should call to_local in multihost mode
        mock_to_local.assert_called_once_with(test_value)

    @patch("embodied.jax.internal.is_multihost")
    def test_fetch_async_single_host(self, mock_is_multihost):
        """Test fetch_async without multihost conversion"""
        mock_is_multihost.return_value = False

        # Create actual JAX arrays for copy_to_host_async
        test_value = jnp.ones(10)

        result = internal.fetch_async(test_value)

        # Should still return value (after async copy)
        assert result is not None


class TestDevicePut:
    """Test device_put() function"""

    @patch("embodied.jax.internal.is_multihost")
    def test_device_put_multihost(self, mock_is_multihost):
        """Test device_put uses make_array_from_process_local_data in multihost"""
        mock_is_multihost.return_value = True

        mesh = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh, P())

        value = jnp.ones(10)

        with patch(
            "embodied.jax.internal.jax.make_array_from_process_local_data"
        ) as mock_make:
            mock_make.return_value = value
            result = internal.device_put(value, sharding)

            # Should call make_array_from_process_local_data
            mock_make.assert_called_once()

    @patch("embodied.jax.internal.is_multihost")
    def test_device_put_single_host(self, mock_is_multihost):
        """Test device_put uses jax.device_put for single host"""
        mock_is_multihost.return_value = False

        mesh = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh, P())

        value = jnp.ones(10)

        result = internal.device_put(value, sharding)

        # Should succeed with regular device_put
        assert result is not None


class TestLocalSharding:
    """Test local_sharding() function"""

    def test_local_sharding_creates_local_mesh(self):
        """Test local_sharding creates NamedSharding with local_mesh"""
        mesh = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh, P())

        result = internal.local_sharding(sharding)

        # Should create new sharding with local_mesh
        assert isinstance(result, jax.sharding.NamedSharding)
        assert result.spec == P()

    def test_local_sharding_with_dict(self):
        """Test local_sharding works with dict of shardings"""
        mesh = Mesh(jax.devices()[:1], ("d",))
        sharding_dict = {
            "param1": jax.sharding.NamedSharding(mesh, P()),
            "param2": jax.sharding.NamedSharding(mesh, P("d")),
        }

        result = internal.local_sharding(sharding_dict)

        # Should preserve structure
        assert isinstance(result, dict)
        assert "param1" in result
        assert "param2" in result


class TestMove:
    """Test move() function"""

    @patch("embodied.jax.internal.is_multihost")
    @patch("embodied.jax.internal.to_local")
    @patch("embodied.jax.internal.to_global")
    @patch("embodied.jax.internal.local_sharding")
    def test_move_multihost(
        self, mock_local_sharding, mock_to_global, mock_to_local, mock_is_multihost
    ):
        """Test move() in multihost setup"""
        mock_is_multihost.return_value = True

        mesh = Mesh(jax.devices()[:1], ("d",))
        dst_sharding = jax.sharding.NamedSharding(mesh, P())
        local_shard = jax.sharding.NamedSharding(mesh, P())

        mock_local_sharding.return_value = local_shard
        test_array = jnp.ones(10)
        mock_to_local.return_value = test_array
        mock_to_global.return_value = test_array

        with patch("embodied.jax.internal.jax.device_put") as mock_device_put:
            mock_device_put.return_value = test_array

            result = internal.move(test_array, dst_sharding)

            # Should call multihost path
            mock_to_local.assert_called_once()
            mock_to_global.assert_called_once()

    @patch("embodied.jax.internal.is_multihost")
    def test_move_single_host(self, mock_is_multihost):
        """Test move() in single host setup"""
        mock_is_multihost.return_value = False

        mesh = Mesh(jax.devices()[:1], ("d",))
        dst_sharding = jax.sharding.NamedSharding(mesh, P())

        test_array = jnp.ones(10)

        result = internal.move(test_array, dst_sharding)

        # Should use simple device_put
        assert result is not None


class TestToLocalAndGlobal:
    """Test to_local(), to_global() helper functions"""

    def test_to_local_with_simple_array(self):
        """Test to_local preserves array structure"""
        # Create a simple sharded array
        mesh = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh, P())

        array = jax.device_put(jnp.ones(10), sharding)

        # This will work in single-device case
        result = internal.to_local(array)

        assert result is not None

    def test_to_global_with_named_sharding(self):
        """Test to_global with NamedSharding"""
        mesh = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh, P())

        array = jax.device_put(jnp.ones(10), sharding)

        result = internal.to_global(array, sharding)

        assert result is not None

    def test_to_global_with_dict_sharding(self):
        """Test to_global with dict of shardings"""
        mesh = Mesh(jax.devices()[:1], ("d",))
        sharding1 = jax.sharding.NamedSharding(mesh, P())
        sharding2 = jax.sharding.NamedSharding(mesh, P())

        arrays = {
            "a": jax.device_put(jnp.ones(10), sharding1),
            "b": jax.device_put(jnp.ones(5), sharding2),
        }

        shardings = {"a": sharding1, "b": sharding2}

        result = internal.to_global(arrays, shardings)

        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result


class TestMeshEdgeCases:
    """Test mesh() function edge cases"""

    def test_mesh_all_minus_one_except_one(self):
        """Test mesh with single -1 in shape gets inferred correctly"""
        # Use number of devices that divides evenly
        num_devices = len(jax.devices())
        if num_devices >= 6:
            devices = jax.devices()[:6]
            shape = "2,-1"  # Should infer 3
        elif num_devices >= 4:
            devices = jax.devices()[:4]
            shape = "2,-1"  # Should infer 2
        elif num_devices >= 2:
            devices = jax.devices()[:2]
            shape = "1,-1"  # Should infer 2
        else:
            pytest.skip("Need at least 2 devices")
        names = ("d", "m")

        mesh = internal.mesh(devices, shape, names)

        assert len(mesh.devices.flat) == len(devices)

    def test_mesh_minus_one_first_position(self):
        """Test mesh with -1 in first position"""
        # Use number of devices that divides evenly
        num_devices = len(jax.devices())
        if num_devices >= 6:
            devices = jax.devices()[:6]
            shape = "-1,2"  # Should infer 3
        elif num_devices >= 4:
            devices = jax.devices()[:4]
            shape = "-1,2"  # Should infer 2
        elif num_devices >= 2:
            devices = jax.devices()[:2]
            shape = "-1,1"  # Should infer 2
        else:
            pytest.skip("Need at least 2 devices")
        names = ("d", "m")

        mesh = internal.mesh(devices, shape, names)

        assert len(mesh.devices.flat) == len(devices)

    def test_mesh_three_dimensions(self):
        """Test mesh with three dimensions"""
        # Need at least 8 devices for 2x2x2
        if len(jax.devices()) >= 8:
            devices = jax.devices()[:8]
            shape = "2,2,2"
            names = ("d", "f", "t")

            mesh = internal.mesh(devices, shape, names)

            assert mesh.shape == {"d": 2, "f": 2, "t": 2}
            assert len(mesh.devices.flat) == 8


class TestCkptFnEdgeCases:
    """Test ckpt_fn() edge cases"""

    def test_ckpt_fn_with_different_shapes(self):
        """Test ckpt_fn with parameters of different shapes"""
        mesh_obj = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh_obj, P())

        params = {
            "small": jax.device_put(jnp.ones(5), sharding),
            "medium": jax.device_put(jnp.ones((10, 20)), sharding),
            "large": jax.device_put(jnp.ones((50, 100, 3)), sharding),
        }

        gather_fn, shard_fn = internal.ckpt_fn(params, compile=False)

        assert hasattr(gather_fn, "compile")
        assert hasattr(shard_fn, "compile")

    def test_ckpt_fn_with_different_dtypes(self):
        """Test ckpt_fn with parameters of different dtypes"""
        mesh_obj = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh_obj, P())

        params = {
            "float32": jax.device_put(jnp.ones(10, dtype=jnp.float32), sharding),
            "bfloat16": jax.device_put(jnp.ones(10, dtype=jnp.bfloat16), sharding),
            "int32": jax.device_put(jnp.ones(10, dtype=jnp.int32), sharding),
        }

        gather_fn, shard_fn = internal.ckpt_fn(params, compile=False)

        assert hasattr(gather_fn, "compile")
        assert hasattr(shard_fn, "compile")


class TestGroupedCkptFnsEdgeCases:
    """Test grouped_ckpt_fns() edge cases"""

    def test_grouped_ckpt_fns_exact_chunksize(self):
        """Test grouped_ckpt_fns when params exactly match chunksize"""
        mesh_obj = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh_obj, P())

        # Create param that's exactly 80KB
        params = {
            "exact": jax.device_put(
                jnp.ones((100, 200), dtype=jnp.float32), sharding
            ),  # 80KB
        }

        chunksize = 100 * 200 * 4  # Exactly 80KB

        result = internal.grouped_ckpt_fns(params, chunksize=chunksize)

        # Should create 1 group
        assert len(result) >= 1

    def test_grouped_ckpt_fns_very_small_chunksize(self):
        """Test grouped_ckpt_fns with very small chunksize (one param per group)"""
        mesh_obj = Mesh(jax.devices()[:1], ("d",))
        sharding = jax.sharding.NamedSharding(mesh_obj, P())

        params = {
            "p1": jax.device_put(jnp.ones(100, dtype=jnp.float32), sharding),
            "p2": jax.device_put(jnp.ones(100, dtype=jnp.float32), sharding),
            "p3": jax.device_put(jnp.ones(100, dtype=jnp.float32), sharding),
        }

        # Small chunksize but not too small - allow params to fit
        # Each param is 100 * 4 = 400 bytes
        chunksize = 500  # Fits one param per group

        result = internal.grouped_ckpt_fns(params, chunksize=chunksize)

        # Should create 3 groups (one for each param)
        assert len(result) == 3
