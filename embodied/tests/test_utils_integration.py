"""
Integration tests for embodied.jax.utils - Normalize and SlowModel classes

Coverage goal: Improve utils.py from 53.90% toward 80%+

Tests cover:
- Normalize: Normalization with meanstd, perc, and none implementations
- SlowModel: Slow-updating target network wrapper
- Integration with ninjax context and state management
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import pytest

from embodied.jax import nets, utils


class TestNormalizeIntegration:
    """Integration tests for Normalize class"""

    def test_normalize_none_implementation(self):
        """Test Normalize with 'none' implementation returns identity"""

        class Model(nj.Module):
            def __call__(self, x):
                norm = self.sub("norm", utils.Normalize, "none")
                return norm(x, update=True)

        model = Model(name="model")
        x = jnp.array([1.0, 2.0, 3.0], jnp.float32)
        state = nj.init(model)({}, x, seed=0)
        state, (mean, std) = nj.pure(model)(state, x)

        # None implementation returns 0.0 mean, 1.0 std
        assert jnp.allclose(mean, 0.0)
        assert jnp.allclose(std, 1.0)

    def test_normalize_meanstd_implementation(self):
        """Test Normalize with 'meanstd' implementation"""

        class Model(nj.Module):
            def __call__(self, x):
                norm = self.sub("norm", utils.Normalize, "meanstd", rate=0.1)
                return norm(x, update=True)

        model = Model(name="model")
        x = jnp.array([1.0, 2.0, 3.0], jnp.float32)

        state = nj.init(model)({}, x, seed=0)

        # First update
        state, (mean1, std1) = nj.pure(model)(state, x)
        assert jnp.isfinite(mean1)
        assert jnp.isfinite(std1)
        assert std1 > 0

        # Second update with different data
        x2 = jnp.array([4.0, 5.0, 6.0], jnp.float32)
        state, (mean2, std2) = nj.pure(model)(state, x2)

        # Mean and std should change with new data
        assert not jnp.allclose(mean1, mean2)

    def test_normalize_perc_implementation(self):
        """Test Normalize with 'perc' percentile implementation"""

        class Model(nj.Module):
            def __call__(self, x):
                norm = self.sub(
                    "norm", utils.Normalize, "perc", rate=0.1, perclo=10.0, perchi=90.0
                )
                return norm(x, update=True)

        model = Model(name="model")
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], jnp.float32)

        state = nj.init(model)({}, x, seed=0)
        state, (lo, scale) = nj.pure(model)(state, x)

        # Should return percentiles
        assert jnp.isfinite(lo)
        assert jnp.isfinite(scale)
        assert scale > 0  # Scale should be positive

    def test_normalize_update_false(self):
        """Test Normalize with update=False doesn't update stats"""

        class Model(nj.Module):
            def __call__(self, x, update):
                norm = self.sub("norm", utils.Normalize, "meanstd", rate=0.1)
                return norm(x, update=update)

        model = Model(name="model")
        x = jnp.array([1.0, 2.0, 3.0], jnp.float32)

        state = nj.init(model)({}, x, update=True, seed=0)

        # Update with data
        state, (mean1, std1) = nj.pure(model)(state, x, update=True)

        # Call without update
        x2 = jnp.array([10.0, 20.0, 30.0], jnp.float32)
        state, (mean2, std2) = nj.pure(model)(state, x2, update=False)

        # Stats should remain same when update=False
        assert jnp.allclose(mean1, mean2)
        assert jnp.allclose(std1, std2)

    def test_normalize_debias_enabled(self):
        """Test Normalize with debias correction"""

        class Model(nj.Module):
            def __call__(self, x):
                norm = self.sub(
                    "norm", utils.Normalize, "meanstd", rate=0.1, debias=True
                )
                return norm(x, update=True)

        model = Model(name="model")
        x = jnp.array([2.0, 4.0, 6.0], jnp.float32)

        state = nj.init(model)({}, x, seed=0)

        # Check that corr variable exists
        assert "model/norm/corr/value" in state

        state, (mean, std) = nj.pure(model)(state, x)
        assert jnp.isfinite(mean)
        assert jnp.isfinite(std)

    def test_normalize_debias_disabled(self):
        """Test Normalize with debias disabled"""

        class Model(nj.Module):
            def __call__(self, x):
                norm = self.sub(
                    "norm", utils.Normalize, "meanstd", rate=0.1, debias=False
                )
                return norm(x, update=True)

        model = Model(name="model")
        x = jnp.array([2.0, 4.0, 6.0], jnp.float32)

        state = nj.init(model)({}, x, seed=0)

        # Corr variable should not exist when debias=False
        assert "model/norm/corr/value" not in state

        state, (mean, std) = nj.pure(model)(state, x)
        assert jnp.isfinite(mean)

    def test_normalize_limit_parameter(self):
        """Test Normalize respects minimum limit for std"""

        class Model(nj.Module):
            def __call__(self, x):
                # Set a high limit
                norm = self.sub("norm", utils.Normalize, "meanstd", limit=1.0)
                return norm(x, update=True)

        model = Model(name="model")
        # Use constant data (std should be very small)
        x = jnp.ones(10, jnp.float32)

        state = nj.init(model)({}, x, seed=0)
        state, (mean, std) = nj.pure(model)(state, x)

        # Std should be at least the limit
        assert std >= 1.0

    def test_normalize_invalid_impl_raises(self):
        """Test Normalize raises on invalid implementation"""

        class Model(nj.Module):
            def __call__(self, x):
                norm = self.sub("norm", utils.Normalize, "invalid")
                return norm(x, update=True)

        model = Model(name="model")
        x = jnp.ones(5, jnp.float32)

        with pytest.raises(NotImplementedError, match="invalid"):
            state = nj.init(model)({}, x, seed=0)

    def test_normalize_multiple_updates(self):
        """Test Normalize updates stats over multiple calls"""

        class Model(nj.Module):
            def __call__(self, x):
                norm = self.sub("norm", utils.Normalize, "meanstd", rate=0.2)
                return norm(x, update=True)

        model = Model(name="model")

        state = nj.init(model)({}, jnp.ones(5), seed=0)

        # Update with different data 5 times
        data = [
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([4.0, 5.0, 6.0]),
            jnp.array([7.0, 8.0, 9.0]),
            jnp.array([10.0, 11.0, 12.0]),
            jnp.array([13.0, 14.0, 15.0]),
        ]

        means = []
        for x in data:
            state, (mean, std) = nj.pure(model)(state, x)
            means.append(float(mean))

        # Mean should generally increase as we feed higher values
        assert means[-1] > means[0]

    def test_normalize_perc_percentile_range(self):
        """Test Normalize perc computes correct percentile range"""

        class Model(nj.Module):
            def __call__(self, x):
                norm = self.sub(
                    "norm", utils.Normalize, "perc", rate=0.5, perclo=0.0, perchi=100.0
                )
                return norm(x, update=True)

        model = Model(name="model")
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], jnp.float32)

        state = nj.init(model)({}, x, seed=0)
        state, (lo, scale) = nj.pure(model)(state, x)

        # With high rate and full range, should approximate min/max
        assert jnp.isfinite(lo)
        assert jnp.isfinite(scale)

    def test_normalize_state_variables_created(self):
        """Test Normalize creates correct state variables"""

        class MeanStdModel(nj.Module):
            def __call__(self, x):
                norm = self.sub("norm", utils.Normalize, "meanstd")
                return norm(x, update=True)

        class PercModel(nj.Module):
            def __call__(self, x):
                norm = self.sub("norm", utils.Normalize, "perc")
                return norm(x, update=True)

        x = jnp.ones(5, jnp.float32)

        # MeanStd should have mean and sqrs
        meanstd_model = MeanStdModel(name="meanstd")
        state = nj.init(meanstd_model)({}, x, seed=0)
        assert "meanstd/norm/mean/value" in state
        assert "meanstd/norm/sqrs/value" in state

        # Perc should have lo and hi
        perc_model = PercModel(name="perc")
        state = nj.init(perc_model)({}, x, seed=0)
        assert "perc/norm/lo/value" in state
        assert "perc/norm/hi/value" in state


class TestSlowModelIntegration:
    """Integration tests for SlowModel class"""

    def test_slowmodel_basic_initialization(self):
        """Test SlowModel initializes and copies parameters"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 10)(x)

        class TrainingSetup(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                slow = utils.SlowModel(self.sub("slow", SimpleModel), source=model)
                return model(x), slow(x)

        setup = TrainingSetup(name="setup")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(setup)({}, x, seed=0)
        state, (out1, out2) = nj.pure(setup)(state, x)

        # Both models should exist
        assert "setup/model/linear/kernel" in state
        assert "setup/slow/linear/kernel" in state

        # Initially, slow model copies from source
        assert jnp.allclose(
            state["setup/model/linear/kernel"], state["setup/slow/linear/kernel"]
        )

    def test_slowmodel_forward_pass(self):
        """Test SlowModel forward pass produces same output initially"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 5)(x)

        class TrainingSetup(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                slow = utils.SlowModel(self.sub("slow", SimpleModel), source=model)
                return model(x), slow(x)

        setup = TrainingSetup(name="setup")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(setup)({}, x, seed=0)
        state, (out_model, out_slow) = nj.pure(setup)(state, x)

        # Initially outputs should be identical (same params)
        assert jnp.allclose(out_model, out_slow, atol=1e-5)

    def test_slowmodel_update_mechanism(self):
        """Test SlowModel update method updates parameters"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class UpdateTest(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                slow = utils.SlowModel(
                    self.sub("slow", SimpleModel), source=model, rate=1.0
                )

                # Just do forward pass and update
                out = model(x)
                slow.update()
                return out

        updater = UpdateTest(name="update")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(updater)({}, x, seed=0)
        state, _ = nj.pure(updater)(state, x)

        # Update counter should increment
        assert state["update/slow_count/value"] == 1

    def test_slowmodel_rate_validation(self):
        """Test SlowModel validates rate parameter"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return x

        # rate >= 0.5 should fail (unless rate == 1.0)
        with pytest.raises(AssertionError):
            slow = utils.SlowModel(
                SimpleModel(name="slow"), source=SimpleModel(name="src"), rate=0.6
            )

        # rate == 1.0 should be OK
        slow = utils.SlowModel(
            SimpleModel(name="slow"), source=SimpleModel(name="src"), rate=1.0
        )
        assert slow.rate == 1.0

        # rate < 0.5 should be OK
        slow = utils.SlowModel(
            SimpleModel(name="slow"), source=SimpleModel(name="src"), rate=0.3
        )
        assert slow.rate == 0.3

    def test_slowmodel_getattr_forwarding(self):
        """Test SlowModel forwards attribute access to wrapped model"""

        class ModelWithAttribute(nj.Module):
            custom_attr: str = "test_value"

            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class Setup(nj.Module):
            def __call__(self, x):
                model = self.sub("model", ModelWithAttribute)
                slow = utils.SlowModel(
                    self.sub("slow", ModelWithAttribute), source=model, rate=1.0
                )
                # Access attribute through slow model
                return slow.custom_attr

        setup = Setup(name="setup")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(setup)({}, x, seed=0)
        state, attr = nj.pure(setup)(state, x)

        assert attr == "test_value"

    def test_slowmodel_count_variable_created(self):
        """Test SlowModel creates count variable"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return x

        class Setup(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                slow = utils.SlowModel(self.sub("slow", SimpleModel), source=model)
                return model(x)

        setup = Setup(name="setup")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(setup)({}, x, seed=0)

        # Count variable should exist
        assert "setup/slow_count/value" in state
        assert state["setup/slow_count/value"] == 0

    def test_slowmodel_every_parameter(self):
        """Test SlowModel respects every parameter for update frequency"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class UpdateTest(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                # Update every 3 steps
                slow = utils.SlowModel(
                    self.sub("slow", SimpleModel), source=model, rate=1.0, every=3
                )
                out = model(x)
                slow.update()
                return out

        updater = UpdateTest(name="update")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(updater)({}, x, seed=0)

        # Call update 5 times
        for i in range(5):
            state, _ = nj.pure(updater)(state, x)
            # Count should increment each time
            assert state["update/slow_count/value"] == i + 1

    def test_slowmodel_with_multiple_layers(self):
        """Test SlowModel with model containing multiple layers"""

        class MultiLayerModel(nj.Module):
            def __call__(self, x):
                x = self.sub("linear1", nets.Linear, 32)(x)
                x = jax.nn.relu(x)
                x = self.sub("linear2", nets.Linear, 16)(x)
                x = jax.nn.relu(x)
                x = self.sub("linear3", nets.Linear, 8)(x)
                return x

        class Setup(nj.Module):
            def __call__(self, x):
                model = self.sub("model", MultiLayerModel)
                slow = utils.SlowModel(self.sub("slow", MultiLayerModel), source=model)
                return model(x), slow(x)

        setup = Setup(name="setup")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(setup)({}, x, seed=0)
        state, (out1, out2) = nj.pure(setup)(state, x)

        # All layers should be copied
        assert "setup/model/linear1/kernel" in state
        assert "setup/model/linear2/kernel" in state
        assert "setup/model/linear3/kernel" in state
        assert "setup/slow/linear1/kernel" in state
        assert "setup/slow/linear2/kernel" in state
        assert "setup/slow/linear3/kernel" in state

        # Initially all should match
        assert jnp.allclose(
            state["setup/model/linear1/kernel"], state["setup/slow/linear1/kernel"]
        )
        assert jnp.allclose(
            state["setup/model/linear2/kernel"], state["setup/slow/linear2/kernel"]
        )


class TestNormalizeSlowModelIntegration:
    """Test Normalize and SlowModel working together"""

    def test_normalize_with_slow_model_pattern(self):
        """Test common pattern: normalize + slow target network"""

        class ValueModel(nj.Module):
            def __call__(self, x):
                linear = self.sub("linear", nets.Linear, 1)
                return linear(x)

        class TrainingSetup(nj.Module):
            def __call__(self, x, update_norm):
                # Normalize input
                norm = self.sub("norm", utils.Normalize, "meanstd", rate=0.01)
                mean, std = norm(x, update=update_norm)
                x_normalized = (x - mean) / std

                # Value network and slow target
                value = self.sub("value", ValueModel)
                slow_value = utils.SlowModel(
                    self.sub("slow_value", ValueModel), source=value, rate=0.01
                )

                # Forward pass
                v1 = value(x_normalized)
                v2 = slow_value(x_normalized)

                return v1, v2, mean, std

        setup = TrainingSetup(name="setup")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE) * 2.0

        state = nj.init(setup)({}, x, update_norm=True, seed=0)
        state, (v1, v2, mean, std) = nj.pure(setup)(state, x, update_norm=True)

        # Both value networks should produce outputs
        assert v1.shape == (4, 1)
        assert v2.shape == (4, 1)

        # Normalization should work
        assert jnp.isfinite(mean)
        assert std > 0

        # Initially slow and fast should be same
        assert jnp.allclose(v1, v2, atol=1e-3)

    def test_normalize_none_with_slowmodel(self):
        """Test that 'none' normalization works with SlowModel"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class Setup(nj.Module):
            def __call__(self, x):
                # No-op normalization
                norm = self.sub("norm", utils.Normalize, "none")
                mean, std = norm(x, update=True)

                # Models
                model = self.sub("model", SimpleModel)
                slow = utils.SlowModel(self.sub("slow", SimpleModel), source=model)

                return model(x), slow(x), mean, std

        setup = Setup(name="setup")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(setup)({}, x, seed=0)
        state, (out1, out2, mean, std) = nj.pure(setup)(state, x)

        # None norm should return identity
        assert mean == 0.0
        assert std == 1.0

        # Models should work
        assert out1.shape == (4, 1)
        assert out2.shape == (4, 1)
