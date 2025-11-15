"""
Integration tests for embodied.jax.opt - Optimizer class with ninjax modules

Coverage goal: Improve opt.py from 45.08% toward 80%+

Tests cover:
- Optimizer initialization with single/multiple modules
- Optimizer __call__ with loss function
- Gradient scaling for float16 compute dtype
- Parameter summarization
- Metrics computation (loss, grad_norm, update_rms, etc.)
- Step counting
- Integration with optax optimizers
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
import pytest

from embodied.jax import nets, opt


class TestOptimizerIntegration:
    """Integration tests for Optimizer class"""

    def test_optimizer_basic_initialization(self):
        """Test Optimizer initializes with single module"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                linear = self.sub("linear", nets.Linear, 10)
                return linear(x)

        class TrainingStep(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                optimizer_fn = optax.adam(1e-3)
                optimizer = self.sub("opt", opt.Optimizer, model, optimizer_fn)

                def loss_fn(x):
                    return jnp.mean(model(x) ** 2).astype(jnp.float32)

                return optimizer(loss_fn, x)

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(trainer)({}, x, seed=0)
        state, metrics = nj.pure(trainer)(state, x)

        # Check optimizer step increments (state has /value suffix)
        assert state["train/opt/step/value"] == 1
        assert metrics["opt/updates"] == 1

    def test_optimizer_with_multiple_modules(self):
        """Test Optimizer with multiple modules"""

        class Encoder(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 32)(x)

        class Decoder(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 16)(x)

        class TrainingStep(nj.Module):
            def __call__(self, x):
                encoder = self.sub("encoder", Encoder)
                decoder = self.sub("decoder", Decoder)
                optimizer_fn = optax.adam(1e-3)
                optimizer = self.sub(
                    "opt", opt.Optimizer, [encoder, decoder], optimizer_fn
                )

                def loss_fn(x):
                    encoded = encoder(x)
                    decoded = decoder(encoded)
                    return jnp.mean(decoded**2).astype(jnp.float32)

                return optimizer(loss_fn, x)

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(trainer)({}, x, seed=0)
        state, metrics = nj.pure(trainer)(state, x)

        # Check optimizer step increments (state has /value suffix)
        assert state["train/opt/step/value"] == 1
        assert metrics["opt/updates"] == 1

    def test_optimizer_simple_training_step(self):
        """Test Optimizer performs single training step"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                linear = self.sub("linear", nets.Linear, 1)
                return linear(x)

        class TrainingStep(nj.Module):
            def __call__(self, x, target):
                model = self.sub("model", SimpleModel)
                optimizer_fn = optax.adam(1e-3)
                optimizer = self.sub("opt", opt.Optimizer, model, optimizer_fn)

                def loss_fn(x, target):
                    pred = model(x)
                    return jnp.mean((pred - target) ** 2).astype(jnp.float32)

                metrics = optimizer(loss_fn, x, target)
                return metrics

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        target = jnp.zeros((4, 1), nets.COMPUTE_DTYPE)

        state = nj.init(trainer)({}, x, target, seed=0)
        state, metrics = nj.pure(trainer)(state, x, target)

        # Check metrics (optimizer adds its own name as prefix)
        assert "opt/loss" in metrics
        assert "opt/updates" in metrics
        assert "opt/grad_norm" in metrics
        assert "opt/grad_rms" in metrics
        assert "opt/update_rms" in metrics
        assert "opt/param_rms" in metrics
        assert "opt/param_count" in metrics

        # Step should increment (step is in state with full path)
        assert state["train/opt/step/value"] == 1

    def test_optimizer_with_aux_output(self):
        """Test Optimizer with has_aux=True"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class TrainingStep(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                optimizer_fn = optax.sgd(1e-2, momentum=0.9)
                optimizer = self.sub("opt", opt.Optimizer, model, optimizer_fn)

                def loss_fn_with_aux(x):
                    pred = model(x)
                    loss = jnp.mean(pred**2).astype(jnp.float32)
                    aux = {"pred": pred, "debug": jnp.array(42.0)}
                    return loss, aux

                metrics, aux = optimizer(loss_fn_with_aux, x, has_aux=True)
                return metrics, aux

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(trainer)({}, x, seed=0)
        state, (metrics, aux) = nj.pure(trainer)(state, x)

        # Check aux output is returned
        assert "pred" in aux
        assert "debug" in aux
        assert aux["debug"] == 42.0

        # Check metrics still exist (optimizer adds its own name as prefix)
        assert "opt/loss" in metrics

    def test_optimizer_multiple_steps(self):
        """Test Optimizer step counter increments correctly"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class TrainingStep(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                optimizer = self.sub("opt", opt.Optimizer, model, optax.adam(1e-3))

                def loss_fn(x):
                    return jnp.mean(model(x) ** 2).astype(jnp.float32)

                return optimizer(loss_fn, x)

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(trainer)({}, x, seed=0)

        # Run 5 training steps
        for i in range(5):
            state, metrics = nj.pure(trainer)(state, x)
            assert state["train/opt/step/value"] == i + 1
            assert metrics["opt/updates"] == i + 1

    def test_optimizer_parameters_update(self):
        """Test Optimizer actually updates parameters"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class TrainingStep(nj.Module):
            def __call__(self, x, target):
                model = self.sub("model", SimpleModel)
                optimizer = self.sub("opt", opt.Optimizer, model, optax.adam(1e-2))

                def loss_fn(x, target):
                    pred = model(x)
                    return jnp.mean((pred - target) ** 2).astype(jnp.float32)

                return optimizer(loss_fn, x, target)

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        target = jnp.zeros((4, 1), nets.COMPUTE_DTYPE)

        state = nj.init(trainer)({}, x, target, seed=0)
        initial_params = state["train/model/linear/kernel"].copy()

        # Run training step
        state, metrics = nj.pure(trainer)(state, x, target)
        updated_params = state["train/model/linear/kernel"]

        # Parameters should have changed
        assert not jnp.allclose(initial_params, updated_params)

    def test_optimizer_loss_decreases(self):
        """Test loss decreases over multiple optimization steps"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class TrainingStep(nj.Module):
            def __call__(self, x, target):
                model = self.sub("model", SimpleModel)
                optimizer = self.sub("opt", opt.Optimizer, model, optax.adam(1e-1))

                def loss_fn(x, target):
                    pred = model(x)
                    return jnp.mean((pred - target) ** 2).astype(jnp.float32)

                return optimizer(loss_fn, x, target)

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        target = jnp.zeros((4, 1), nets.COMPUTE_DTYPE)

        state = nj.init(trainer)({}, x, target, seed=0)

        losses = []
        for _ in range(10):
            state, metrics = nj.pure(trainer)(state, x, target)
            losses.append(float(metrics["opt/loss"]))

        # Loss should generally decrease (may have some noise)
        assert losses[-1] < losses[0]

    def test_optimizer_metrics_structure(self):
        """Test Optimizer returns all expected metrics"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class TrainingStep(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                optimizer = self.sub("opt", opt.Optimizer, model, optax.adam(1e-3))

                def loss_fn(x):
                    return jnp.mean(model(x) ** 2).astype(jnp.float32)

                return optimizer(loss_fn, x)

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        state = nj.init(trainer)({}, x, seed=0)
        state, metrics = nj.pure(trainer)(state, x)

        # Check all expected metrics exist (optimizer adds its own name as prefix)
        expected_keys = [
            "opt/loss",
            "opt/updates",
            "opt/grad_norm",
            "opt/grad_rms",
            "opt/update_rms",
            "opt/param_rms",
            "opt/param_count",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

        # Check metric types
        assert jnp.isfinite(metrics["opt/loss"])
        assert metrics["opt/updates"] >= 0
        assert jnp.isfinite(metrics["opt/grad_norm"])
        assert metrics["opt/param_count"] > 0

    def test_optimizer_with_different_optimizers(self):
        """Test Optimizer works with different optax optimizers"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        optimizers_to_test = [
            optax.adam(1e-3),
            optax.sgd(1e-2, momentum=0.9),
            optax.adamw(1e-3),
            optax.rmsprop(1e-3),
        ]

        for optimizer_fn in optimizers_to_test:

            class TrainingStep(nj.Module):
                def __call__(self, x):
                    model = self.sub("model", SimpleModel)
                    optimizer = self.sub("opt", opt.Optimizer, model, optimizer_fn)

                    def loss_fn(x):
                        return jnp.mean(model(x) ** 2).astype(jnp.float32)

                    return optimizer(loss_fn, x)

            trainer = TrainingStep(name="train")
            x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

            state = nj.init(trainer)({}, x, seed=0)
            state, metrics = nj.pure(trainer)(state, x)

            # Should work with any optax optimizer (optimizer adds its own name as prefix)
            assert "opt/loss" in metrics
            assert state["train/opt/step/value"] == 1


class TestOptimizerGradientScaling:
    """Test Optimizer gradient scaling for float16 compute dtype"""

    def test_optimizer_no_scaling_for_float32(self):
        """Test gradient scaling is disabled for float32 dtype"""

        # Temporarily change compute dtype to float32
        original_dtype = nets.COMPUTE_DTYPE
        try:
            nets.COMPUTE_DTYPE = jnp.float32

            class SimpleModel(nj.Module):
                def __call__(self, x):
                    return self.sub("linear", nets.Linear, 1)(x)

            class TrainingStep(nj.Module):
                def __call__(self, x):
                    model = self.sub("model", SimpleModel)
                    optimizer = self.sub("opt", opt.Optimizer, model, optax.adam(1e-3))

                    def loss_fn(x):
                        return jnp.mean(model(x) ** 2)

                    return optimizer(loss_fn, x)

            trainer = TrainingStep(name="train")
            x = jnp.ones((4, 16), jnp.float32)

            state = nj.init(trainer)({}, x, seed=0)
            state, metrics = nj.pure(trainer)(state, x)

            # Should NOT have grad_scale or grad_overflow metrics (optimizer adds its own name)
            assert "opt/grad_scale" not in metrics
            assert "opt/grad_overflow" not in metrics

        finally:
            nets.COMPUTE_DTYPE = original_dtype

    def test_optimizer_scaling_for_float16(self):
        """Test gradient scaling is enabled for float16 dtype"""

        # Temporarily change compute dtype to float16
        original_dtype = nets.COMPUTE_DTYPE
        try:
            nets.COMPUTE_DTYPE = jnp.float16

            class SimpleModel(nj.Module):
                def __call__(self, x):
                    return self.sub("linear", nets.Linear, 1)(x)

            class TrainingStep(nj.Module):
                def __call__(self, x):
                    model = self.sub("model", SimpleModel)
                    optimizer = self.sub("opt", opt.Optimizer, model, optax.adam(1e-3))

                    def loss_fn(x):
                        return jnp.mean(model(x).astype(jnp.float32) ** 2)

                    return optimizer(loss_fn, x)

            trainer = TrainingStep(name="train")
            x = jnp.ones((4, 16), jnp.float16)

            state = nj.init(trainer)({}, x, seed=0)
            state, metrics = nj.pure(trainer)(state, x)

            # SHOULD have grad_scale and grad_overflow metrics
            assert "opt/grad_scale" in metrics
            assert "opt/grad_overflow" in metrics

            # Check grad_scale state variables exist (state has /value suffix)
            assert "train/opt/grad_scale/value" in state
            assert "train/opt/good_steps/value" in state

        finally:
            nets.COMPUTE_DTYPE = original_dtype


class TestOptimizerParameterSummarization:
    """Test Optimizer parameter summarization"""

    def test_optimizer_parameter_summary_depth(self):
        """Test _summarize_params with different depths"""

        class MultiLayerModel(nj.Module):
            def __call__(self, x):
                encoder = self.sub("encoder", nets.MLP, 2, 64)
                decoder = self.sub("decoder", nets.MLP, 2, 64)
                return decoder(encoder(x))

        class TrainingStep(nj.Module):
            def __call__(self, x):
                model = self.sub("model", MultiLayerModel)
                optimizer = self.sub("opt", opt.Optimizer, model, optax.adam(1e-3))

                def loss_fn(x):
                    return jnp.mean(model(x) ** 2).astype(jnp.float32)

                return optimizer(loss_fn, x)

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        # Run one step to trigger parameter summary printing
        state = nj.init(trainer)({}, x, seed=0)
        state, metrics = nj.pure(trainer)(state, x)

        # Verify training succeeded (summary is printed during first step)
        assert state is not None
        assert "train/opt/step/value" in state
        assert metrics["opt/updates"] == 1


class TestOptimizerErrorHandling:
    """Test Optimizer error handling and validation"""

    def test_optimizer_validates_loss_dtype(self):
        """Test Optimizer validates loss is float32"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class TrainingStep(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                optimizer = self.sub("opt", opt.Optimizer, model, optax.adam(1e-3))

                def loss_fn(x):
                    # Return wrong dtype
                    return jnp.mean(model(x).astype(jnp.float16) ** 2)

                return optimizer(loss_fn, x)

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        with pytest.raises(AssertionError):
            state = nj.init(trainer)({}, x, seed=0)

    def test_optimizer_validates_loss_shape(self):
        """Test Optimizer validates loss is scalar"""

        class SimpleModel(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 1)(x)

        class TrainingStep(nj.Module):
            def __call__(self, x):
                model = self.sub("model", SimpleModel)
                optimizer = self.sub("opt", opt.Optimizer, model, optax.adam(1e-3))

                def loss_fn(x):
                    # Return wrong shape (not scalar)
                    pred = model(x)
                    return ((pred - 0.5) ** 2).astype(jnp.float32)  # Shape (4, 1)

                return optimizer(loss_fn, x)

        trainer = TrainingStep(name="train")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        with pytest.raises(AssertionError):
            state = nj.init(trainer)({}, x, seed=0)
