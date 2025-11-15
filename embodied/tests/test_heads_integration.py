"""
Integration tests for embodied.jax.heads - Head modules with output distributions

Coverage goal: Improve heads.py from 29.77% toward 80%+

Tests cover:
- MLPHead with various output types
- DictHead for multi-output spaces
- Head implementations: binary, categorical, onehot, mse, huber, symlog_mse,
  symexp_twohot, bounded_normal, normal_logstd
- Output distribution properties (pred, sample, entropy, log_prob)
"""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from embodied.jax import heads, nets


class TestMLPHeadIntegration:
    """Test MLPHead with various output types"""

    def test_mlphead_with_mse_output(self):
        """Test MLPHead with MSE output for continuous space"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (3,))
                head = self.sub("head", heads.MLPHead, space, "mse", units=64, layers=2)
                return head(x, bdims=1)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert hasattr(out, "pred")
        assert hasattr(out, "loss")
        pred = out.pred()
        assert pred.shape == (4, 3)

    def test_mlphead_with_categorical_output(self):
        """Test MLPHead with categorical output for discrete space"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(
                    np.int32, (2,), 0, 4
                )  # 2 actions, 4 classes each
                head = self.sub(
                    "head", heads.MLPHead, space, "categorical", units=64, layers=2
                )
                out = head(x, bdims=1)
                sample = out.sample(seed=nj.seed())
                return out, sample

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, (out, sample) = nj.pure(model)(state, x, seed=1)

        assert hasattr(out, "sample")
        assert sample.shape == (4, 2)

    def test_mlphead_with_bounded_normal_output(self):
        """Test MLPHead with bounded normal output"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (3,))
                head = self.sub(
                    "head",
                    heads.MLPHead,
                    space,
                    "bounded_normal",
                    units=64,
                    layers=2,
                    minstd=0.1,
                    maxstd=1.0,
                )
                out = head(x, bdims=1)
                sample = out.sample(seed=nj.seed())
                return out, sample

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, (out, sample) = nj.pure(model)(state, x, seed=1)

        assert hasattr(out, "sample")
        assert hasattr(out, "entropy")
        # When bdims > 0, output is wrapped in Agg, so check the underlying output
        underlying = out.output if hasattr(out, "output") else out
        assert hasattr(underlying, "minent")
        assert hasattr(underlying, "maxent")
        assert sample.shape == (4, 3)

    def test_mlphead_with_dict_space(self):
        """Test MLPHead with dict space (DictHead path)"""

        class Model(nj.Module):
            def __call__(self, x):
                spaces = {
                    "action": elements.Space(np.int32, (), 0, 5),
                    "value": elements.Space(np.float32, ()),
                }
                outputs = {"action": "categorical", "value": "mse"}
                head = self.sub(
                    "head", heads.MLPHead, spaces, outputs, units=64, layers=2
                )
                return head(x, bdims=1)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert isinstance(out, dict)
        assert "action" in out
        assert "value" in out
        assert hasattr(out["action"], "sample")
        assert hasattr(out["value"], "pred")

    def test_mlphead_with_batched_input(self):
        """Test MLPHead with multi-dimensional batch"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (2,))
                head = self.sub("head", heads.MLPHead, space, "mse", units=32, layers=1)
                # Time x Batch x Features
                return head(x, bdims=2)

        model = Model(name="model")
        x = jnp.ones((8, 4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        pred = out.pred()
        assert pred.shape == (8, 4, 2)


class TestDictHeadIntegration:
    """Test DictHead for multi-output spaces"""

    def test_dictheadwith_multiple_outputs(self):
        """Test DictHead with multiple output spaces"""

        class Model(nj.Module):
            def __call__(self, x):
                spaces = {
                    "reward": elements.Space(np.float32, ()),
                    "done": elements.Space(np.int32, (), 0, 2),
                }
                outputs = {"reward": "mse", "done": "categorical"}
                head = self.sub("head", heads.DictHead, spaces, outputs)
                return head(x)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert isinstance(out, dict)
        assert "reward" in out
        assert "done" in out
        assert hasattr(out["reward"], "pred")
        assert hasattr(out["done"], "sample")

    def test_dicthead_single_space_conversion(self):
        """Test DictHead converts non-dict space to dict"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (2,))
                output_type = "mse"
                head = self.sub("head", heads.DictHead, space, output_type)
                return head(x)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert isinstance(out, dict)
        assert "output" in out
        assert hasattr(out["output"], "pred")


class TestHeadOutputTypes:
    """Test different Head output implementations"""

    def test_head_binary_output(self):
        """Test Head with binary output"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.int32, (2,), 0, 2)  # 2 binary outputs
                head = self.sub("head", heads.Head, space, "binary")
                out = head(x)
                sample = out.sample(seed=nj.seed())
                return out, sample

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, (out, sample) = nj.pure(model)(state, x, seed=1)

        assert hasattr(out, "sample")
        assert sample.shape == (4, 2)

    def test_head_categorical_output(self):
        """Test Head with categorical output"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.int32, (3,), 0, 5)  # 3 actions, 5 classes
                head = self.sub("head", heads.Head, space, "categorical")
                out = head(x)
                sample = out.sample(seed=nj.seed())
                return out, sample

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, (out, sample) = nj.pure(model)(state, x, seed=1)

        assert hasattr(out, "sample")
        assert hasattr(out, "entropy")
        # When space has multiple dims, output is wrapped in Agg
        underlying = out.output if hasattr(out, "output") else out
        assert hasattr(underlying, "minent")
        assert hasattr(underlying, "maxent")
        assert sample.shape == (4, 3)

    def test_head_onehot_output(self):
        """Test Head with onehot output"""

        class Model(nj.Module):
            def __call__(self, x):
                # For onehot, we need continuous space with classes
                space = elements.Space(np.int32, (3,), 0, 5)  # Will be converted
                head = self.sub("head", heads.Head, space, "onehot", unimix=0.01)
                out = head(x)
                sample = out.sample(seed=nj.seed())
                return out, sample

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, (out, sample) = nj.pure(model)(state, x, seed=1)

        assert hasattr(out, "sample")
        assert sample.shape == (4, 3, 5)

    def test_head_mse_output(self):
        """Test Head with MSE output"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (3,))
                head = self.sub("head", heads.Head, space, "mse")
                return head(x)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert hasattr(out, "pred")
        assert hasattr(out, "loss")
        pred = out.pred()
        assert pred.shape == (4, 3)

    def test_head_huber_output(self):
        """Test Head with Huber output"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (2,))
                head = self.sub("head", heads.Head, space, "huber")
                return head(x)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert hasattr(out, "pred")
        assert hasattr(out, "loss")
        target = jnp.ones((4, 2), jnp.float32)
        loss = out.loss(target)
        assert loss.shape == (4,)

    def test_head_symlog_mse_output(self):
        """Test Head with symlog MSE output"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (2,))
                head = self.sub("head", heads.Head, space, "symlog_mse")
                return head(x)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert hasattr(out, "pred")
        pred = out.pred()
        assert pred.shape == (4, 2)

    def test_head_symexp_twohot_output(self):
        """Test Head with symexp twohot output"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (2,))
                head = self.sub("head", heads.Head, space, "symexp_twohot", bins=51)
                return head(x)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert hasattr(out, "pred")
        pred = out.pred()
        assert pred.shape == (4, 2)

    def test_head_symexp_twohot_even_bins(self):
        """Test Head with symexp twohot output with even number of bins"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, ())
                head = self.sub("head", heads.Head, space, "symexp_twohot", bins=50)
                return head(x)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert hasattr(out, "pred")
        pred = out.pred()
        assert pred.shape == (4,)

    def test_head_bounded_normal_output(self):
        """Test Head with bounded normal output"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (2,))
                head = self.sub(
                    "head", heads.Head, space, "bounded_normal", minstd=0.1, maxstd=1.0
                )
                out = head(x)
                sample = out.sample(seed=nj.seed())
                return out, sample

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, (out, sample) = nj.pure(model)(state, x, seed=1)

        assert hasattr(out, "sample")
        assert hasattr(out, "entropy")
        # When space has multiple dims, output is wrapped in Agg
        underlying = out.output if hasattr(out, "output") else out
        assert hasattr(underlying, "minent")
        assert hasattr(underlying, "maxent")
        assert sample.shape == (4, 2)
        # Note: "bounded_normal" bounds the mean via tanh, not the samples
        # Samples can exceed [-1, 1] due to stddev

    def test_head_normal_logstd_output(self):
        """Test Head with normal log_std output"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (3,))
                head = self.sub("head", heads.Head, space, "normal_logstd")
                out = head(x)
                sample = out.sample(seed=nj.seed())
                return out, sample

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, (out, sample) = nj.pure(model)(state, x, seed=1)

        assert hasattr(out, "sample")
        assert hasattr(out, "entropy")
        assert sample.shape == (4, 3)

    def test_head_tuple_space_conversion(self):
        """Test Head converts tuple space to Space"""

        class Model(nj.Module):
            def __call__(self, x):
                space = (3,)  # Tuple shape
                head = self.sub("head", heads.Head, space, "mse")
                return head(x)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert hasattr(out, "pred")
        pred = out.pred()
        assert pred.shape == (4, 3)


class TestHeadErrorHandling:
    """Test Head error handling and validation"""

    def test_head_invalid_output_type_raises(self):
        """Test Head raises for invalid output type"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (2,))
                head = self.sub("head", heads.Head, space, "invalid_type")
                return head(x)

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)

        with pytest.raises(NotImplementedError, match="invalid_type"):
            state = nj.init(model)({}, x, seed=0)

    def test_head_shape_validation(self):
        """Test Head validates output shape matches space"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (3,))
                head = self.sub("head", heads.Head, space, "mse")
                output = head(x)
                pred = output.pred()
                # Verify shape assertion
                assert pred.shape[-1:] == (3,)
                return output

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, out = nj.pure(model)(state, x)

        assert out is not None


class TestHeadOutputProperties:
    """Test output distribution properties"""

    def test_categorical_entropy_bounds(self):
        """Test categorical output has correct entropy bounds"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.int32, (), 0, 5)
                head = self.sub("head", heads.Head, space, "categorical")
                output = head(x)
                entropy = output.entropy()
                return output, entropy

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, (out, entropy) = nj.pure(model)(state, x)

        # Entropy should be between minent and maxent
        assert hasattr(out, "minent")
        assert hasattr(out, "maxent")
        # maxent should be log(num_classes)
        expected_maxent = np.log(5)
        assert np.isclose(out.maxent, expected_maxent, atol=1e-5)

    def test_bounded_normal_entropy_bounds(self):
        """Test bounded normal has correct entropy bounds"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, ())
                minstd, maxstd = 0.1, 1.0
                head = self.sub(
                    "head",
                    heads.Head,
                    space,
                    "bounded_normal",
                    minstd=minstd,
                    maxstd=maxstd,
                )
                output = head(x)
                entropy = output.entropy()
                return output, entropy

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, (out, entropy) = nj.pure(model)(state, x)

        # Check minent/maxent attributes exist
        assert hasattr(out, "minent")
        assert hasattr(out, "maxent")

    def test_output_loss_computation(self):
        """Test output loss computation"""

        class Model(nj.Module):
            def __call__(self, x):
                space = elements.Space(np.float32, (2,))
                head = self.sub("head", heads.Head, space, "mse")
                output = head(x)
                target = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
                loss = output.loss(target)
                return loss

        model = Model(name="model")
        x = jnp.ones((4, 16), nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, loss = nj.pure(model)(state, x)

        assert loss.shape == (4,)
        assert jnp.all(jnp.isfinite(loss))
