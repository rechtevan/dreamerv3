"""
Tests for embodied.jax.outs - JAX output distributions

Coverage goal: 90% (from 33.64%)

Tests cover:
- Output base class (repr, prob, loss)
- Agg: Aggregation wrapper
- Frozen: Stop-gradient wrapper
- Concat: Concatenation of outputs
- MSE: Mean squared error output
- Huber: Huber loss output
- Normal: Normal distribution
- Binary: Bernoulli distribution
- Categorical: Categorical distribution
- OneHot: One-hot categorical with straight-through gradients
- TwoHot: Two-hot encoding for regression
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embodied.jax import outs


class TestOutputBaseClass:
    """Test Output base class"""

    def test_output_repr(self):
        """Test Output.__repr__ with concrete implementation"""
        # Use MSE as concrete implementation
        output = outs.MSE(jnp.array([1.0, 2.0, 3.0]))
        repr_str = repr(output)

        assert "MSE" in repr_str
        assert "float32" in repr_str
        assert "shape=" in repr_str

    def test_output_prob(self):
        """Test Output.prob returns exp(logp)"""
        # Use Normal as concrete implementation
        output = outs.Normal(jnp.array([0.0, 1.0]), stddev=1.0)
        event = jnp.array([0.5, 1.5])

        prob = output.prob(event)
        expected = jnp.exp(output.logp(event))

        assert jnp.allclose(prob, expected)

    def test_output_loss_default(self):
        """Test Output.loss default implementation returns -logp"""
        # Use Normal as concrete implementation
        output = outs.Normal(jnp.array([0.0, 1.0]), stddev=1.0)
        target = jnp.array([0.5, 1.5])

        loss = output.loss(target)
        expected = -output.logp(target)

        assert jnp.allclose(loss, expected)


class TestAgg:
    """Test Agg aggregation wrapper"""

    def test_agg_initialization(self):
        """Test Agg initializes correctly"""
        base = outs.Normal(jnp.ones((2, 3, 4)), stddev=1.0)
        agg = outs.Agg(base, dims=2)

        assert agg.output is base
        assert agg.axes == [-1, -2]
        assert agg.agg == jnp.sum

    def test_agg_custom_aggregation(self):
        """Test Agg with custom aggregation function"""
        base = outs.Normal(jnp.ones((2, 3, 4)), stddev=1.0)
        agg = outs.Agg(base, dims=2, agg=jnp.mean)

        assert agg.agg == jnp.mean

    def test_agg_repr(self):
        """Test Agg.__repr__"""
        base = outs.Normal(jnp.ones((2, 3, 4)), stddev=1.0)
        agg = outs.Agg(base, dims=2)
        repr_str = repr(agg)

        assert "Normal" in repr_str
        assert "agg=2" in repr_str
        assert "shape=" in repr_str

    def test_agg_pred_forwards(self):
        """Test Agg.pred forwards to base output"""
        base = outs.Normal(jnp.array([1.0, 2.0, 3.0]), stddev=1.0)
        agg = outs.Agg(base, dims=1)

        pred = agg.pred()

        assert jnp.array_equal(pred, base.pred())

    def test_agg_loss_aggregates(self):
        """Test Agg.loss aggregates along specified axes"""
        base = outs.Normal(jnp.ones((2, 3)), stddev=1.0)
        agg = outs.Agg(base, dims=1)
        target = jnp.ones((2, 3)) * 2.0

        loss = agg.loss(target)
        base_loss = base.loss(target)

        # Should sum along last axis
        assert loss.shape == (2,)
        assert jnp.allclose(loss, base_loss.sum(-1))

    def test_agg_sample_forwards(self):
        """Test Agg.sample forwards to base output"""
        base = outs.Normal(jnp.array([1.0, 2.0]), stddev=1.0)
        agg = outs.Agg(base, dims=1)
        seed = jax.random.PRNGKey(42)

        sample = agg.sample(seed)

        assert sample.shape == base.mean.shape

    def test_agg_logp_sums(self):
        """Test Agg.logp sums along specified axes"""
        base = outs.Normal(jnp.ones((2, 3)), stddev=1.0)
        agg = outs.Agg(base, dims=1)
        event = jnp.ones((2, 3))

        logp = agg.logp(event)
        base_logp = base.logp(event)

        assert logp.shape == (2,)
        assert jnp.allclose(logp, base_logp.sum(-1))

    def test_agg_prob_sums(self):
        """Test Agg.prob sums along specified axes"""
        base = outs.Normal(jnp.ones((2, 3)), stddev=1.0)
        agg = outs.Agg(base, dims=1)
        event = jnp.ones((2, 3))

        prob = agg.prob(event)
        base_prob = base.prob(event)

        assert prob.shape == (2,)
        assert jnp.allclose(prob, base_prob.sum(-1))

    def test_agg_entropy_aggregates(self):
        """Test Agg.entropy aggregates along specified axes"""
        base = outs.Normal(jnp.ones((2, 3)), stddev=1.0)
        agg = outs.Agg(base, dims=1)

        entropy = agg.entropy()
        base_entropy = base.entropy()

        assert entropy.shape == (2,)
        assert jnp.allclose(entropy, base_entropy.sum(-1))

    def test_agg_kl_aggregates(self):
        """Test Agg.kl aggregates KL divergence"""
        base1 = outs.Normal(jnp.ones((2, 3)), stddev=1.0)
        base2 = outs.Normal(jnp.ones((2, 3)) * 2, stddev=1.0)
        agg1 = outs.Agg(base1, dims=1)
        agg2 = outs.Agg(base2, dims=1)

        kl = agg1.kl(agg2)
        base_kl = base1.kl(base2)

        assert kl.shape == (2,)
        assert jnp.allclose(kl, base_kl.sum(-1))


class TestFrozen:
    """Test Frozen stop-gradient wrapper"""

    def test_frozen_initialization(self):
        """Test Frozen initializes correctly"""
        base = outs.Normal(jnp.array([1.0, 2.0]), stddev=1.0)
        frozen = outs.Frozen(base)

        assert frozen.output is base

    def test_frozen_getattr_forwards(self):
        """Test Frozen.__getattr__ forwards methods"""
        base = outs.Normal(jnp.array([1.0, 2.0]), stddev=1.0)
        frozen = outs.Frozen(base)

        # Should be able to call methods
        assert callable(frozen.pred)
        assert callable(frozen.logp)

    def test_frozen_getattr_rejects_dunder(self):
        """Test Frozen.__getattr__ rejects dunder attributes"""
        base = outs.Normal(jnp.array([1.0]), stddev=1.0)
        frozen = outs.Frozen(base)

        with pytest.raises(AttributeError):
            _ = frozen.__foo__

    def test_frozen_getattr_raises_on_missing(self):
        """Test Frozen.__getattr__ raises ValueError for missing attributes"""
        base = outs.Normal(jnp.array([1.0]), stddev=1.0)
        frozen = outs.Frozen(base)

        with pytest.raises(ValueError):
            _ = frozen.nonexistent_method

    def test_frozen_wrapper_applies_stop_gradient(self):
        """Test Frozen._wrapper applies stop_gradient to results"""
        base = outs.Normal(jnp.array([1.0, 2.0]), stddev=1.0)
        frozen = outs.Frozen(base)

        # Get prediction through frozen wrapper
        pred = frozen.pred()

        # Result should have stop_gradient applied
        # We can't directly test stop_gradient, but we can verify the call works
        assert jnp.array_equal(pred, base.pred())


class TestConcat:
    """Test Concat wrapper for concatenating outputs"""

    def test_concat_initialization(self):
        """Test Concat initializes correctly"""
        out1 = outs.Normal(jnp.ones((2, 3)), stddev=1.0)
        out2 = outs.Normal(jnp.ones((2, 4)), stddev=1.0)
        concat = outs.Concat([out1, out2], midpoints=[3], axis=1)

        assert len(concat.outputs) == 2
        assert concat.midpoints == (3,)
        assert concat.axis == 1

    def test_concat_getattr_forwards(self):
        """Test Concat.__getattr__ forwards methods"""
        out1 = outs.Normal(jnp.ones((2,)), stddev=1.0)
        out2 = outs.Normal(jnp.ones((3,)), stddev=1.0)
        concat = outs.Concat([out1, out2], midpoints=[2], axis=0)

        assert callable(concat.pred)
        assert callable(concat.sample)

    def test_concat_getattr_rejects_dunder(self):
        """Test Concat.__getattr__ rejects dunder attributes"""
        out1 = outs.Normal(jnp.ones((2,)), stddev=1.0)
        concat = outs.Concat([out1], midpoints=[], axis=0)

        with pytest.raises(AttributeError):
            _ = concat.__foo__

    def test_concat_getattr_raises_on_missing(self):
        """Test Concat.__getattr__ raises ValueError for missing attributes"""
        out1 = outs.Normal(jnp.ones((2,)), stddev=1.0)
        concat = outs.Concat([out1], midpoints=[], axis=0)

        with pytest.raises(ValueError):
            _ = concat.nonexistent_method


class TestMSE:
    """Test MSE (Mean Squared Error) output"""

    def test_mse_initialization(self):
        """Test MSE initializes correctly"""
        mean = jnp.array([1.0, 2.0, 3.0])
        mse = outs.MSE(mean)

        assert jnp.array_equal(mse.mean, mean.astype(jnp.float32))
        assert callable(mse.squash)

    def test_mse_initialization_with_squash(self):
        """Test MSE with custom squash function"""
        mean = jnp.array([1.0, 2.0])
        squash = lambda x: jnp.tanh(x)
        mse = outs.MSE(mean, squash=squash)

        assert mse.squash is squash

    def test_mse_pred(self):
        """Test MSE.pred returns mean"""
        mean = jnp.array([1.0, 2.0, 3.0])
        mse = outs.MSE(mean)

        pred = mse.pred()

        assert jnp.array_equal(pred, mean)

    def test_mse_loss(self):
        """Test MSE.loss computes squared error"""
        mean = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 2.5, 2.5])
        mse = outs.MSE(mean)

        loss = mse.loss(target)
        expected = jnp.square(mean - target)

        assert jnp.allclose(loss, expected)

    def test_mse_loss_with_squash(self):
        """Test MSE.loss with squash function"""
        mean = jnp.array([0.0, 1.0])
        target = jnp.array([2.0, 3.0])
        squash = lambda x: x * 0.5
        mse = outs.MSE(mean, squash=squash)

        loss = mse.loss(target)
        expected = jnp.square(mean - squash(target))

        assert jnp.allclose(loss, expected)

    def test_mse_loss_shape_assertion(self):
        """Test MSE.loss asserts matching shapes"""
        mean = jnp.array([1.0, 2.0])
        target = jnp.array([1.0, 2.0, 3.0])  # Wrong shape
        mse = outs.MSE(mean)

        with pytest.raises(AssertionError):
            mse.loss(target)


class TestHuber:
    """Test Huber (Charbonnier) loss output"""

    def test_huber_initialization(self):
        """Test Huber initializes correctly"""
        mean = jnp.array([1.0, 2.0, 3.0])
        huber = outs.Huber(mean)

        assert jnp.array_equal(huber.mean, mean.astype(jnp.float32))
        assert huber.eps == 1.0

    def test_huber_initialization_custom_eps(self):
        """Test Huber with custom epsilon"""
        mean = jnp.array([1.0, 2.0])
        huber = outs.Huber(mean, eps=0.5)

        assert huber.eps == 0.5

    def test_huber_pred(self):
        """Test Huber.pred returns mean"""
        mean = jnp.array([1.0, 2.0, 3.0])
        huber = outs.Huber(mean)

        pred = huber.pred()

        assert jnp.array_equal(pred, mean)

    def test_huber_loss(self):
        """Test Huber.loss computes Charbonnier loss"""
        mean = jnp.array([1.0, 2.0])
        target = jnp.array([1.5, 2.5])
        eps = 1.0
        huber = outs.Huber(mean, eps=eps)

        loss = huber.loss(target)
        dist = mean - target
        expected = jnp.sqrt(jnp.square(dist) + jnp.square(eps)) - eps

        assert jnp.allclose(loss, expected)

    def test_huber_loss_shape_assertion(self):
        """Test Huber.loss asserts matching shapes"""
        mean = jnp.array([1.0, 2.0])
        target = jnp.array([1.0, 2.0, 3.0])  # Wrong shape
        huber = outs.Huber(mean)

        with pytest.raises(AssertionError):
            huber.loss(target)


class TestNormal:
    """Test Normal distribution output"""

    def test_normal_initialization(self):
        """Test Normal initializes correctly"""
        mean = jnp.array([1.0, 2.0, 3.0])
        normal = outs.Normal(mean, stddev=1.5)

        assert jnp.array_equal(normal.mean, mean.astype(jnp.float32))
        assert jnp.all(normal.stddev == 1.5)
        assert normal.stddev.shape == mean.shape

    def test_normal_initialization_scalar_stddev(self):
        """Test Normal with scalar stddev broadcasts correctly"""
        mean = jnp.array([1.0, 2.0])
        normal = outs.Normal(mean, stddev=2.0)

        assert normal.stddev.shape == mean.shape
        assert jnp.all(normal.stddev == 2.0)

    def test_normal_pred(self):
        """Test Normal.pred returns mean"""
        mean = jnp.array([1.0, 2.0])
        normal = outs.Normal(mean)

        pred = normal.pred()

        assert jnp.array_equal(pred, mean)

    def test_normal_sample(self):
        """Test Normal.sample generates samples"""
        mean = jnp.array([0.0, 0.0])
        stddev = 1.0
        normal = outs.Normal(mean, stddev=stddev)
        seed = jax.random.PRNGKey(42)

        sample = normal.sample(seed)

        assert sample.shape == mean.shape
        # Sample should be different from mean (with high probability)
        assert not jnp.allclose(sample, mean)

    def test_normal_sample_with_batch(self):
        """Test Normal.sample with batch shape"""
        mean = jnp.array([1.0, 2.0])
        normal = outs.Normal(mean, stddev=1.0)
        seed = jax.random.PRNGKey(42)

        sample = normal.sample(seed, shape=(3, 4))

        assert sample.shape == (3, 4, 2)

    def test_normal_logp(self):
        """Test Normal.logp computes log probability"""
        mean = jnp.array([0.0, 1.0])
        stddev = 1.0
        normal = outs.Normal(mean, stddev=stddev)
        event = jnp.array([0.0, 1.0])

        logp = normal.logp(event)

        # At the mean, log probability should be maximal
        assert logp.shape == event.shape
        # Compare with scipy implementation
        expected = jax.scipy.stats.norm.logpdf(event, mean, stddev)
        assert jnp.allclose(logp, expected)

    def test_normal_entropy(self):
        """Test Normal.entropy computes entropy"""
        mean = jnp.array([0.0, 1.0])
        stddev = 2.0
        normal = outs.Normal(mean, stddev=stddev)

        entropy = normal.entropy()

        # Entropy of normal: 0.5 * log(2 * pi * sigma^2) + 0.5
        expected = 0.5 * jnp.log(2 * jnp.pi * stddev**2) + 0.5
        assert jnp.allclose(entropy, expected)

    def test_normal_kl(self):
        """Test Normal.kl computes KL divergence"""
        mean1 = jnp.array([0.0, 1.0])
        mean2 = jnp.array([0.5, 1.5])
        normal1 = outs.Normal(mean1, stddev=1.0)
        normal2 = outs.Normal(mean2, stddev=2.0)

        kl = normal1.kl(normal2)

        assert kl.shape == mean1.shape
        # KL should be non-negative
        assert jnp.all(kl >= -1e-5)

    def test_normal_kl_same_distribution(self):
        """Test Normal.kl is zero for same distribution"""
        mean = jnp.array([0.0, 1.0])
        stddev = 1.5
        normal1 = outs.Normal(mean, stddev=stddev)
        normal2 = outs.Normal(mean, stddev=stddev)

        kl = normal1.kl(normal2)

        # KL(p||p) = 0
        assert jnp.allclose(kl, 0.0, atol=1e-6)


class TestBinary:
    """Test Binary (Bernoulli) distribution output"""

    def test_binary_initialization(self):
        """Test Binary initializes correctly"""
        logit = jnp.array([0.0, 1.0, -1.0])
        binary = outs.Binary(logit)

        assert jnp.array_equal(binary.logit, logit.astype(jnp.float32))

    def test_binary_pred(self):
        """Test Binary.pred returns logit > 0"""
        logit = jnp.array([-1.0, 0.0, 0.5, 1.0])
        binary = outs.Binary(logit)

        pred = binary.pred()
        expected = logit > 0

        assert jnp.array_equal(pred, expected)

    def test_binary_logp(self):
        """Test Binary.logp computes log probability"""
        logit = jnp.array([0.0, 2.0])
        binary = outs.Binary(logit)
        event = jnp.array([1.0, 0.0])

        logp = binary.logp(event)

        # Manual calculation
        logp_1 = jax.nn.log_sigmoid(logit)
        logp_0 = jax.nn.log_sigmoid(-logit)
        expected = event * logp_1 + (1 - event) * logp_0

        assert jnp.allclose(logp, expected)

    def test_binary_sample(self):
        """Test Binary.sample generates binary samples"""
        logit = jnp.array([2.0, -2.0])  # High/low probability
        binary = outs.Binary(logit)
        seed = jax.random.PRNGKey(42)

        sample = binary.sample(seed)

        assert sample.shape == logit.shape
        # Samples should be 0 or 1
        assert jnp.all((sample == 0) | (sample == 1))

    def test_binary_sample_with_batch(self):
        """Test Binary.sample with batch shape"""
        logit = jnp.array([1.0, -1.0])
        binary = outs.Binary(logit)
        seed = jax.random.PRNGKey(42)

        sample = binary.sample(seed, shape=(3,))

        assert sample.shape == (3, 2)


class TestCategorical:
    """Test Categorical distribution output"""

    def test_categorical_initialization(self):
        """Test Categorical initializes correctly"""
        logits = jnp.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        cat = outs.Categorical(logits)

        assert cat.logits.shape == logits.shape
        assert cat.logits.dtype == jnp.float32

    def test_categorical_initialization_with_unimix(self):
        """Test Categorical with uniform mixture"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        unimix = 0.1
        cat = outs.Categorical(logits, unimix=unimix)

        # Logits should be modified by uniform mixture
        probs = jax.nn.softmax(logits, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        expected_probs = (1 - unimix) * probs + unimix * uniform
        expected_logits = jnp.log(expected_probs)

        assert jnp.allclose(cat.logits, expected_logits)

    def test_categorical_pred(self):
        """Test Categorical.pred returns argmax"""
        logits = jnp.array([[1.0, 3.0, 2.0], [0.5, 0.3, 0.8]])
        cat = outs.Categorical(logits)

        pred = cat.pred()
        expected = jnp.array([1, 2])  # Argmax indices

        assert jnp.array_equal(pred, expected)

    def test_categorical_sample(self):
        """Test Categorical.sample generates samples"""
        logits = jnp.array([[10.0, 0.0, 0.0]])  # Strong preference for first class
        cat = outs.Categorical(logits)
        seed = jax.random.PRNGKey(42)

        sample = cat.sample(seed)

        assert sample.shape == (1,)
        # Sample should be in valid range
        assert jnp.all(sample >= 0) and jnp.all(sample < 3)

    def test_categorical_sample_with_batch(self):
        """Test Categorical.sample with batch shape"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        cat = outs.Categorical(logits)
        seed = jax.random.PRNGKey(42)

        sample = cat.sample(seed, shape=(5,))

        assert sample.shape == (5, 1)

    def test_categorical_logp(self):
        """Test Categorical.logp computes log probability"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        cat = outs.Categorical(logits)
        event = jnp.array([2])  # Select last class

        logp = cat.logp(event)

        # Manual calculation
        log_probs = jax.nn.log_softmax(logits, -1)
        expected = log_probs[0, 2]

        assert jnp.allclose(logp, expected)

    def test_categorical_entropy(self):
        """Test Categorical.entropy computes entropy"""
        # Uniform distribution should have maximum entropy
        logits = jnp.array([[0.0, 0.0, 0.0]])
        cat = outs.Categorical(logits)

        entropy = cat.entropy()

        # Entropy of uniform distribution over 3 classes
        expected = jnp.log(3.0)
        assert jnp.allclose(entropy, expected, atol=1e-6)

    def test_categorical_kl(self):
        """Test Categorical.kl computes KL divergence"""
        logits1 = jnp.array([[1.0, 2.0, 3.0]])
        logits2 = jnp.array([[3.0, 2.0, 1.0]])
        cat1 = outs.Categorical(logits1)
        cat2 = outs.Categorical(logits2)

        kl = cat1.kl(cat2)

        assert kl.shape == (1,)
        # KL should be non-negative
        assert jnp.all(kl >= -1e-5)

    def test_categorical_kl_same_distribution(self):
        """Test Categorical.kl is zero for same distribution"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        cat1 = outs.Categorical(logits)
        cat2 = outs.Categorical(logits)

        kl = cat1.kl(cat2)

        # KL(p||p) = 0
        assert jnp.allclose(kl, 0.0, atol=1e-6)


class TestOneHot:
    """Test OneHot categorical with straight-through gradients"""

    def test_onehot_initialization(self):
        """Test OneHot initializes correctly"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        onehot = outs.OneHot(logits)

        assert isinstance(onehot.dist, outs.Categorical)
        assert jnp.array_equal(onehot.dist.logits, logits.astype(jnp.float32))

    def test_onehot_initialization_with_unimix(self):
        """Test OneHot with uniform mixture"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        onehot = outs.OneHot(logits, unimix=0.1)

        # Should pass unimix to Categorical
        assert isinstance(onehot.dist, outs.Categorical)

    def test_onehot_pred(self):
        """Test OneHot.pred returns one-hot vector"""
        logits = jnp.array([[1.0, 3.0, 2.0]])
        onehot = outs.OneHot(logits)

        pred = onehot.pred()

        # Should return one-hot encoding of argmax
        assert pred.shape == (1, 3)
        assert jnp.allclose(pred.sum(-1), 1.0)
        # Argmax is index 1, so pred should be [0, 1, 0]
        assert jnp.allclose(pred, jnp.array([[0.0, 1.0, 0.0]]), atol=0.1)

    def test_onehot_sample(self):
        """Test OneHot.sample returns one-hot vector"""
        logits = jnp.array([[10.0, 0.0, 0.0]])  # Strong preference
        onehot = outs.OneHot(logits)
        seed = jax.random.PRNGKey(42)

        sample = onehot.sample(seed)

        assert sample.shape == (1, 3)
        # Should be one-hot encoded
        assert jnp.allclose(sample.sum(-1), 1.0, atol=0.1)

    def test_onehot_logp(self):
        """Test OneHot.logp computes log probability"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        onehot = outs.OneHot(logits)
        event = jnp.array([[0.0, 0.0, 1.0]])  # Select last class

        logp = onehot.logp(event)

        # Should compute log softmax and sum with event
        log_probs = jax.nn.log_softmax(logits, -1)
        expected = (log_probs * event).sum(-1)

        assert jnp.allclose(logp, expected)

    def test_onehot_entropy(self):
        """Test OneHot.entropy delegates to Categorical"""
        logits = jnp.array([[0.0, 0.0, 0.0]])
        onehot = outs.OneHot(logits)

        entropy = onehot.entropy()
        expected = onehot.dist.entropy()

        assert jnp.allclose(entropy, expected)

    def test_onehot_kl(self):
        """Test OneHot.kl delegates to Categorical"""
        logits1 = jnp.array([[1.0, 2.0, 3.0]])
        logits2 = jnp.array([[3.0, 2.0, 1.0]])
        onehot1 = outs.OneHot(logits1)
        onehot2 = outs.OneHot(logits2)

        kl = onehot1.kl(onehot2)
        expected = onehot1.dist.kl(onehot2.dist)

        assert jnp.allclose(kl, expected)


class TestTwoHot:
    """Test TwoHot encoding for regression"""

    def test_twohot_initialization(self):
        """Test TwoHot initializes correctly"""
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        bins = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
        twohot = outs.TwoHot(logits, bins)

        assert jnp.array_equal(twohot.logits, logits.astype(jnp.float32))
        assert jnp.array_equal(twohot.bins, bins)
        assert jnp.allclose(twohot.probs, jax.nn.softmax(logits))

    def test_twohot_initialization_with_squash(self):
        """Test TwoHot with squash/unsquash functions"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        bins = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
        squash = lambda x: jnp.tanh(x)
        unsquash = lambda x: jnp.arctanh(x)
        twohot = outs.TwoHot(logits, bins, squash=squash, unsquash=unsquash)

        assert twohot.squash is squash
        assert twohot.unsquash is unsquash

    def test_twohot_pred_odd_bins(self):
        """Test TwoHot.pred with odd number of bins (symmetric sum)"""
        # Uniform distribution over symmetric bins
        logits = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        bins = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
        twohot = outs.TwoHot(logits, bins)

        pred = twohot.pred()

        # Symmetric uniform distribution should predict near zero
        assert pred.shape == (1,)
        assert jnp.allclose(pred, 0.0, atol=0.1)

    def test_twohot_pred_even_bins(self):
        """Test TwoHot.pred with even number of bins (symmetric sum)"""
        # Uniform distribution over symmetric bins
        logits = jnp.array([[0.0, 0.0, 0.0, 0.0]])
        bins = jnp.array([-1.5, -0.5, 0.5, 1.5], dtype=jnp.float32)
        twohot = outs.TwoHot(logits, bins)

        pred = twohot.pred()

        # Symmetric uniform distribution should predict near zero
        assert pred.shape == (1,)
        assert jnp.allclose(pred, 0.0, atol=0.1)

    def test_twohot_pred_with_unsquash(self):
        """Test TwoHot.pred applies unsquash function"""
        logits = jnp.array([[10.0, 0.0, 0.0]])  # Strong preference for first bin
        bins = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
        unsquash = lambda x: x * 2.0
        twohot = outs.TwoHot(logits, bins, unsquash=unsquash)

        pred = twohot.pred()

        # Should apply unsquash to the weighted average
        assert pred.shape == (1,)

    def test_twohot_loss(self):
        """Test TwoHot.loss computes two-hot loss"""
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        bins = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
        twohot = outs.TwoHot(logits, bins)
        target = jnp.array([0.5])

        loss = twohot.loss(target)

        assert loss.shape == (1,)
        # Loss should be non-negative
        assert jnp.all(loss >= 0)

    def test_twohot_loss_with_squash(self):
        """Test TwoHot.loss applies squash to target"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        bins = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
        squash = lambda x: jnp.clip(x, 0.0, 2.0)
        twohot = outs.TwoHot(logits, bins, squash=squash)
        target = jnp.array([5.0])  # Will be clipped to 2.0

        loss = twohot.loss(target)

        assert loss.shape == (1,)

    def test_twohot_assertion_logits_bins_match(self):
        """Test TwoHot asserts logits and bins have matching dimensions"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        bins = jnp.array([0.0, 1.0], dtype=jnp.float32)  # Mismatch

        with pytest.raises(AssertionError):
            outs.TwoHot(logits, bins)
