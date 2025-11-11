"""
Integration tests for embodied.jax.nets - ninjax module testing

This file tests the ninjax.Module classes in nets.py that require full JAX
context, state management, and integration with ninjax infrastructure.

Target: Improve coverage from 66.03% to 85%+

Modules tested:
- Linear, BlockLinear (linear transformations)
- Conv2D, Conv3D (convolutions)
- Norm (normalization layers)
- Embed, DictEmbed (embedding layers)
- Attention (multi-head attention)
- MLP (multi-layer perceptron)
- Transformer (transformer blocks)
- GRU (recurrent layer)
"""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from embodied.jax import nets


class TestLinearIntegration:
    """Integration tests for Linear module"""

    def test_linear_forward_pass(self):
        """Test Linear module forward pass with ninjax context"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 128)(x)

        model = Model(name="model")
        x = jnp.ones((4, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (4, 128)
        assert y.dtype == nets.COMPUTE_DTYPE
        assert "model/linear/kernel" in state
        assert "model/linear/bias" in state
        assert state["model/linear/kernel"].shape == (64, 128)

    def test_linear_multi_dimensional_output(self):
        """Test Linear with tuple output units"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, (8, 16))(x)

        model = Model(name="model")
        x = jnp.ones((4, 32), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (4, 8, 16)

    def test_linear_without_bias(self):
        """Test Linear module without bias"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 128, bias=False)(x)

        model = Model(name="model")
        x = jnp.ones((4, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert "model/linear/kernel" in state
        assert "model/linear/bias" not in state

    def test_linear_gradient_flow(self):
        """Test gradient computation through Linear"""

        class Model(nj.Module):
            def __call__(self, x):
                y = self.sub("linear", nets.Linear, 32)(x)
                return jnp.sum(y**2)

        model = Model(name="model")
        x = jnp.ones((4, 16), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)

        def loss_fn(x):
            return nj.pure(model)(state, x)[1]

        grads = jax.grad(loss_fn)(x)

        assert grads.shape == x.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_linear_with_outscale(self):
        """Test Linear with output scaling"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("linear", nets.Linear, 64, outscale=0.1)(x)

        model = Model(name="model")
        x = jnp.ones((4, 32), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        # Output should be scaled down
        assert jnp.abs(y).mean() < 1.0


class TestBlockLinearIntegration:
    """Integration tests for BlockLinear module"""

    def test_blocklinear_forward_pass(self):
        """Test BlockLinear forward pass"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("blocklinear", nets.BlockLinear, units=64, blocks=4)(x)

        model = Model(name="model")
        x = jnp.ones((2, 32), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 64)
        assert "model/blocklinear/kernel" in state
        assert state["model/blocklinear/kernel"].shape == (4, 8, 16)

    def test_blocklinear_blocks_divisibility(self):
        """Test BlockLinear with proper block alignment"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("blocklinear", nets.BlockLinear, units=128, blocks=8)(x)

        model = Model(name="model")
        x = jnp.ones((2, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 128)


class TestConv2DIntegration:
    """Integration tests for Conv2D module"""

    def test_conv2d_forward_pass(self):
        """Test Conv2D forward pass"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("conv", nets.Conv2D, depth=32, kernel=3, stride=1)(x)

        model = Model(name="model")
        x = jnp.ones((2, 16, 16, 3), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 16, 16, 32)
        assert "model/conv/kernel" in state
        assert state["model/conv/kernel"].shape == (3, 3, 3, 32)

    def test_conv2d_stride(self):
        """Test Conv2D with stride"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("conv", nets.Conv2D, depth=64, kernel=3, stride=2)(x)

        model = Model(name="model")
        x = jnp.ones((2, 32, 32, 3), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 16, 16, 64)

    def test_conv2d_transpose(self):
        """Test Conv2D transposed (upsampling)"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub(
                    "conv", nets.Conv2D, depth=16, kernel=3, stride=2, transp=True
                )(x)

        model = Model(name="model")
        x = jnp.ones((2, 8, 8, 32), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 16, 16, 16)

    def test_conv2d_groups(self):
        """Test Conv2D with grouped convolution"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("conv", nets.Conv2D, depth=64, kernel=3, groups=4)(x)

        model = Model(name="model")
        x = jnp.ones((2, 16, 16, 32), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 16, 16, 64)

    def test_conv2d_gradient_flow(self):
        """Test gradient computation through Conv2D"""

        class Model(nj.Module):
            def __call__(self, x):
                y = self.sub("conv", nets.Conv2D, depth=16, kernel=3)(x)
                return jnp.sum(y**2)

        model = Model(name="model")
        x = jnp.ones((2, 8, 8, 3), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)

        def loss_fn(x):
            return nj.pure(model)(state, x)[1]

        grads = jax.grad(loss_fn)(x)

        assert grads.shape == x.shape
        assert jnp.all(jnp.isfinite(grads))


class TestConv3DIntegration:
    """Integration tests for Conv3D module"""

    def test_conv3d_forward_pass(self):
        """Test Conv3D forward pass"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("conv", nets.Conv3D, depth=32, kernel=3, stride=1)(x)

        model = Model(name="model")
        x = jnp.ones((2, 8, 8, 8, 3), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 8, 8, 8, 32)
        assert "model/conv/kernel" in state

    def test_conv3d_transpose(self):
        """Test Conv3D transposed"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub(
                    "conv", nets.Conv3D, depth=16, kernel=3, stride=2, transp=True
                )(x)

        model = Model(name="model")
        x = jnp.ones((2, 4, 4, 4, 32), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 8, 8, 8, 16)


class TestNormIntegration:
    """Integration tests for Norm module"""

    def test_norm_rms(self):
        """Test RMS normalization"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("norm", nets.Norm, "rms")(x)

        model = Model(name="model")
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        # RMS norm should normalize to have RMS ≈ 1
        assert y.shape == x.shape
        assert "model/norm/scale" in state

    def test_norm_layer(self):
        """Test layer normalization"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("norm", nets.Norm, "layer")(x)

        model = Model(name="model")
        x = jnp.ones((2, 4, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == x.shape
        assert "model/norm/scale" in state
        assert "model/norm/shift" in state

    def test_norm_none(self):
        """Test no normalization"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("norm", nets.Norm, "none")(x)

        model = Model(name="model")
        x = jnp.ones((2, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert jnp.array_equal(y, x)

    def test_norm_without_scale_shift(self):
        """Test normalization without learnable parameters"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("norm", nets.Norm, "rms", scale=False, shift=False)(x)

        model = Model(name="model")
        x = jnp.ones((2, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        # No learnable parameters
        assert "model/norm/scale" not in state
        assert "model/norm/shift" not in state

    def test_norm_custom_eps(self):
        """Test normalization with custom epsilon"""

        class Model(nj.Module):
            def __call__(self, x):
                norm = self.sub("norm", nets.Norm, "rms1em6")  # eps = 1e-6
                return norm(x)

        model = Model(name="model")
        x = jnp.ones((2, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == x.shape


class TestEmbedIntegration:
    """Integration tests for Embed module"""

    def test_embed_forward_pass(self):
        """Test Embed forward pass"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("embed", nets.Embed, classes=10, units=64)(x)

        model = Model(name="model")
        x = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (4, 64)
        assert "model/embed/table" in state
        assert state["model/embed/table"].shape == (10, 64)

    def test_embed_with_shape(self):
        """Test Embed with multi-dimensional indices"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub(
                    "embed", nets.Embed, classes=10, units=32, shape=(2, 2)
                )(x)

        model = Model(name="model")
        x = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 2, 32)

    def test_embed_combine(self):
        """Test Embed with combine=True"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub(
                    "embed", nets.Embed, classes=10, units=64, shape=(3,), combine=True
                )(x)

        model = Model(name="model")
        x = jnp.array([0, 1, 2], dtype=jnp.int32)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        # With combine=True, embeddings are summed
        assert y.shape == (64,)


class TestDictEmbedIntegration:
    """Integration tests for DictEmbed module"""

    def test_dictembed_forward_pass(self):
        """Test DictEmbed with discrete spaces"""
        spaces = {
            "action": elements.Space(np.int32, (), 0, 4),
            "reward": elements.Space(np.float32, ()),
        }

        class Model(nj.Module):
            def __call__(self, xs):
                return self.sub("dictembed", nets.DictEmbed, spaces, units=64)(
                    xs, bshape=(2,)
                )

        model = Model(name="model")
        xs = {
            "action": jnp.array([0, 1], dtype=jnp.int32),
            "reward": jnp.array([1.0, 2.0], dtype=jnp.float32),
        }
        state = nj.init(model)({}, xs, seed=0)
        state, y = nj.pure(model)(state, xs)

        assert y.shape == (2, 64)
        assert "model/dictembed/init" in state

    def test_dictembed_onehot_impl(self):
        """Test DictEmbed with onehot implementation"""
        spaces = {
            "action": elements.Space(np.int32, (), 0, 8),
        }

        class Model(nj.Module):
            def __call__(self, xs):
                return self.sub(
                    "dictembed", nets.DictEmbed, spaces, units=32, impl="onehot"
                )(xs, bshape=(4,))

        model = Model(name="model")
        xs = {"action": jnp.array([0, 1, 2, 3], dtype=jnp.int32)}
        state = nj.init(model)({}, xs, seed=0)
        state, y = nj.pure(model)(state, xs)

        assert y.shape == (4, 32)

    def test_dictembed_lookup_impl(self):
        """Test DictEmbed with lookup implementation"""
        spaces = {
            "action": elements.Space(np.int32, (), 0, 8),
        }

        class Model(nj.Module):
            def __call__(self, xs):
                return self.sub(
                    "dictembed", nets.DictEmbed, spaces, units=32, impl="lookup"
                )(xs, bshape=(4,))

        model = Model(name="model")
        xs = {"action": jnp.array([0, 1, 2, 3], dtype=jnp.int32)}
        state = nj.init(model)({}, xs, seed=0)
        state, y = nj.pure(model)(state, xs)

        assert y.shape == (4, 32)


class TestMLPIntegration:
    """Integration tests for MLP module"""

    def test_mlp_forward_pass(self):
        """Test MLP forward pass"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("mlp", nets.MLP, layers=3, units=128)(x)

        model = Model(name="model")
        x = jnp.ones((4, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (4, 128)
        # Check that all layers are created
        assert "model/mlp/linear0/kernel" in state
        assert "model/mlp/linear1/kernel" in state
        assert "model/mlp/linear2/kernel" in state
        assert "model/mlp/norm0/scale" in state

    def test_mlp_activation_functions(self):
        """Test MLP with different activation functions"""
        for act_name in ["silu", "relu", "tanh"]:

            class Model(nj.Module):
                def __call__(self, x):
                    return self.sub("mlp", nets.MLP, layers=2, units=64, act=act_name)(
                        x
                    )

            model = Model(name="model")
            x = jnp.ones((4, 32), dtype=nets.COMPUTE_DTYPE)
            state = nj.init(model)({}, x, seed=0)
            state, y = nj.pure(model)(state, x)

            assert y.shape == (4, 64)

    def test_mlp_gradient_flow(self):
        """Test gradient computation through MLP"""

        class Model(nj.Module):
            def __call__(self, x):
                y = self.sub("mlp", nets.MLP, layers=2, units=64)(x)
                return jnp.sum(y**2)

        model = Model(name="model")
        x = jnp.ones((4, 32), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)

        def loss_fn(x):
            return nj.pure(model)(state, x)[1]

        grads = jax.grad(loss_fn)(x)

        assert grads.shape == x.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_mlp_with_batch_dimensions(self):
        """Test MLP preserves leading batch dimensions"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("mlp", nets.MLP, layers=2, units=64)(x)

        model = Model(name="model")
        x = jnp.ones((2, 3, 32), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 3, 64)


class TestAttentionIntegration:
    """Integration tests for Attention module"""

    def test_attention_forward_pass(self):
        """Test Attention forward pass"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("attn", nets.Attention, heads=4, rope=False)(x)

        model = Model(name="model")
        x = jnp.ones((2, 8, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 8, 64)
        assert "model/attn/qkv/kernel" in state
        assert "model/attn/proj/kernel" in state

    def test_attention_with_mask(self):
        """Test Attention with attention mask"""

        class Model(nj.Module):
            def __call__(self, x, mask):
                return self.sub("attn", nets.Attention, heads=4, rope=False)(
                    x, mask=mask
                )

        model = Model(name="model")
        x = jnp.ones((2, 8, 64), dtype=nets.COMPUTE_DTYPE)
        # Causal mask
        mask = jnp.tril(jnp.ones((2, 8, 8), dtype=bool))
        state = nj.init(model)({}, x, mask, seed=0)
        state, y = nj.pure(model)(state, x, mask)

        assert y.shape == (2, 8, 64)

    def test_attention_with_rope(self):
        """Test Attention with rotary position embeddings"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub("attn", nets.Attention, heads=4, rope=True)(x)

        model = Model(name="model")
        x = jnp.ones((2, 8, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 8, 64)

    def test_attention_multi_query(self):
        """Test Attention with multi-query (fewer KV heads)"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub(
                    "attn", nets.Attention, heads=8, kv_heads=2, rope=False
                )(x)

        model = Model(name="model")
        x = jnp.ones((2, 8, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 8, 64)
        # Should have separate q, k, v projections
        assert "model/attn/q/kernel" in state
        assert "model/attn/k/kernel" in state
        assert "model/attn/v/kernel" in state

    def test_attention_with_qknorm(self):
        """Test Attention with QK normalization"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub(
                    "attn", nets.Attention, heads=4, qknorm="rms", rope=False
                )(x)

        model = Model(name="model")
        x = jnp.ones((2, 8, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 8, 64)
        assert "model/attn/normq/scale" in state
        assert "model/attn/normk/scale" in state

    def test_attention_gradient_flow(self):
        """Test gradient computation through Attention"""

        class Model(nj.Module):
            def __call__(self, x):
                y = self.sub("attn", nets.Attention, heads=4, rope=False)(x)
                return jnp.sum(y**2)

        model = Model(name="model")
        x = jnp.ones((2, 8, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)

        def loss_fn(x):
            return nj.pure(model)(state, x)[1]

        grads = jax.grad(loss_fn)(x)

        assert grads.shape == x.shape
        assert jnp.all(jnp.isfinite(grads))


class TestTransformerIntegration:
    """Integration tests for Transformer module"""

    def test_transformer_forward_pass(self):
        """Test Transformer forward pass"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub(
                    "trans", nets.Transformer, units=64, layers=2, heads=4, rope=False
                )(x)

        model = Model(name="model")
        x = jnp.ones((2, 8, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 8, 64)
        assert "model/trans/layer0/mha/qkv/kernel" in state
        assert "model/trans/layer1/mha/qkv/kernel" in state
        assert "model/trans/outnorm/scale" in state

    def test_transformer_with_glu(self):
        """Test Transformer with GLU activation"""

        class Model(nj.Module):
            def __call__(self, x):
                return self.sub(
                    "trans",
                    nets.Transformer,
                    units=64,
                    layers=2,
                    heads=4,
                    glu=True,
                    rope=False,
                )(x)

        model = Model(name="model")
        x = jnp.ones((2, 8, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        assert y.shape == (2, 8, 64)

    def test_transformer_gradient_flow(self):
        """Test gradient computation through Transformer"""

        class Model(nj.Module):
            def __call__(self, x):
                y = self.sub(
                    "trans", nets.Transformer, units=64, layers=2, heads=4, rope=False
                )(x)
                return jnp.sum(y**2)

        model = Model(name="model")
        x = jnp.ones((2, 8, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)

        def loss_fn(x):
            return nj.pure(model)(state, x)[1]

        grads = jax.grad(loss_fn)(x)

        assert grads.shape == x.shape
        assert jnp.all(jnp.isfinite(grads))


class TestGRUIntegration:
    """Integration tests for GRU module"""

    def test_gru_forward_pass(self):
        """Test GRU forward pass"""

        class Model(nj.Module):
            def __call__(self, carry, inputs, resets):
                gru = self.sub("gru", nets.GRU, units=128)
                return gru(carry, inputs, resets)

        model = Model(name="model")
        B, T = 4, 8
        carry = jnp.zeros((B, 128), dtype=nets.COMPUTE_DTYPE)
        inputs = jnp.ones((B, T, 64), dtype=nets.COMPUTE_DTYPE)
        resets = jnp.zeros((B, T), dtype=bool)

        state = nj.init(model)({}, carry, inputs, resets, seed=0)
        state, (new_carry, outputs) = nj.pure(model)(state, carry, inputs, resets)

        assert new_carry.shape == (B, 128)
        assert outputs.shape == (B, T, 128)
        assert "model/gru/linear/kernel" in state
        assert "model/gru/norm/scale" in state

    def test_gru_single_step(self):
        """Test GRU single step mode"""

        class Model(nj.Module):
            def __call__(self, carry, inp, reset):
                gru = self.sub("gru", nets.GRU, units=64)
                return gru(carry, inp, reset, single=True)

        model = Model(name="model")
        B = 4
        carry = jnp.zeros((B, 64), dtype=nets.COMPUTE_DTYPE)
        inp = jnp.ones((B, 32), dtype=nets.COMPUTE_DTYPE)
        reset = jnp.zeros(B, dtype=bool)

        state = nj.init(model)({}, carry, inp, reset, seed=0)
        state, (new_carry, output) = nj.pure(model)(state, carry, inp, reset)

        assert new_carry.shape == (B, 64)
        assert output.shape == (B, 64)

    def test_gru_with_resets(self):
        """Test GRU properly handles reset signals"""

        class Model(nj.Module):
            def __call__(self, carry, inputs, resets):
                gru = self.sub("gru", nets.GRU, units=64)
                return gru(carry, inputs, resets)

        model = Model(name="model")
        B, T = 2, 4
        carry = jnp.zeros((B, 64), dtype=nets.COMPUTE_DTYPE)
        inputs = jnp.ones((B, T, 32), dtype=nets.COMPUTE_DTYPE)
        # Reset at timestep 2
        resets = jnp.array([[False, False, True, False], [False, False, False, False]])

        state = nj.init(model)({}, carry, inputs, resets, seed=0)
        state, (new_carry, outputs) = nj.pure(model)(state, carry, inputs, resets)

        assert outputs.shape == (B, T, 64)

    def test_gru_gradient_flow(self):
        """Test gradient computation through GRU"""

        class Model(nj.Module):
            def __call__(self, carry, inputs, resets):
                gru = self.sub("gru", nets.GRU, units=64)
                _, outputs = gru(carry, inputs, resets)
                return jnp.sum(outputs**2)

        model = Model(name="model")
        B, T = 2, 4
        carry = jnp.zeros((B, 64), dtype=nets.COMPUTE_DTYPE)
        inputs = jnp.ones((B, T, 32), dtype=nets.COMPUTE_DTYPE)
        resets = jnp.zeros((B, T), dtype=bool)

        state = nj.init(model)({}, carry, inputs, resets, seed=0)

        def loss_fn(inputs):
            return nj.pure(model)(state, carry, inputs, resets)[1]

        grads = jax.grad(loss_fn)(inputs)

        assert grads.shape == inputs.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_gru_initial_state(self):
        """Test GRU initial state generation"""
        # Test the static method without ninjax context
        batch_sizes = [1, 4, 16]
        units = 128
        for B in batch_sizes:
            # Create GRU directly and call initial method
            gru = nets.GRU(units=units, name="gru")
            carry = gru.initial(B)
            assert carry.shape == (B, units)
            assert carry.dtype == nets.COMPUTE_DTYPE
            assert jnp.all(carry == 0)


class TestDropoutIntegration:
    """Integration tests for dropout function"""

    def test_dropout_training_mode(self):
        """Test dropout in training mode"""

        class Model(nj.Module):
            def __call__(self, x):
                return nets.dropout(x, prob=0.5, training=True)

        model = Model(name="model")
        x = jnp.ones((100, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x, seed=1)

        # Some elements should be zeroed out
        assert jnp.sum(y == 0) > 0
        # Some elements should be scaled up
        assert jnp.any(y > 1.0)

    def test_dropout_eval_mode(self):
        """Test dropout in eval mode (no dropout)"""

        class Model(nj.Module):
            def __call__(self, x):
                return nets.dropout(x, prob=0.5, training=False)

        model = Model(name="model")
        x = jnp.ones((100, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        # No dropout in eval mode
        assert jnp.array_equal(y, x)

    def test_dropout_zero_prob(self):
        """Test dropout with zero probability"""

        class Model(nj.Module):
            def __call__(self, x):
                return nets.dropout(x, prob=0.0, training=True)

        model = Model(name="model")
        x = jnp.ones((100, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, y = nj.pure(model)(state, x)

        # No dropout when prob=0
        assert jnp.array_equal(y, x)


class TestInitializerIntegration:
    """Integration tests for Initializer with ninjax context"""

    def test_initializer_uniform(self):
        """Test uniform initialization"""

        class Model(nj.Module):
            def __call__(self):
                init = nets.Initializer("uniform", "in")
                return init((100, 50))

        model = Model(name="model")
        state = nj.init(model)({}, seed=0)
        state, weights = nj.pure(model)(state, seed=1)

        assert weights.shape == (100, 50)
        # Uniform initialization should have values in reasonable range
        assert jnp.abs(weights).max() < 1.0

    def test_initializer_normal(self):
        """Test normal initialization"""

        class Model(nj.Module):
            def __call__(self):
                init = nets.Initializer("normal", "in")
                return init((100, 50))

        model = Model(name="model")
        state = nj.init(model)({}, seed=0)
        state, weights = nj.pure(model)(state, seed=1)

        assert weights.shape == (100, 50)
        # Most values should be within 3 standard deviations
        assert jnp.abs(weights).max() < 1.0

    def test_initializer_trunc_normal(self):
        """Test truncated normal initialization"""

        class Model(nj.Module):
            def __call__(self):
                init = nets.Initializer("trunc_normal", "in")
                return init((100, 50))

        model = Model(name="model")
        state = nj.init(model)({}, seed=0)
        state, weights = nj.pure(model)(state, seed=1)

        assert weights.shape == (100, 50)
        # Truncated normal should have no extreme outliers
        assert jnp.abs(weights).max() < 1.0

    def test_initializer_normed(self):
        """Test normed initialization"""

        class Model(nj.Module):
            def __call__(self):
                init = nets.Initializer("normed", "in")
                return init((100, 50))

        model = Model(name="model")
        state = nj.init(model)({}, seed=0)
        state, weights = nj.pure(model)(state, seed=1)

        assert weights.shape == (100, 50)
        # Check column-wise normalization
        norms = jnp.linalg.norm(weights, axis=0)
        assert jnp.allclose(norms, 1.0, rtol=0.1)

    def test_initializer_fan_modes(self):
        """Test different fan modes"""
        for fan in ["in", "out", "avg"]:

            class Model(nj.Module):
                def __call__(self):
                    init = nets.Initializer("uniform", fan)
                    return init((100, 50))

            model = Model(name="model")
            state = nj.init(model)({}, seed=0)
            state, weights = nj.pure(model)(state, seed=1)

            assert weights.shape == (100, 50)

    def test_initializer_scale(self):
        """Test initialization with custom scale"""

        class Model(nj.Module):
            def __call__(self):
                init_normal = nets.Initializer("uniform", "in", scale=1.0)
                init_scaled = nets.Initializer("uniform", "in", scale=0.1)
                w1 = init_normal((100, 50))
                w2 = init_scaled((100, 50))
                return w1, w2

        model = Model(name="model")
        state = nj.init(model)({}, seed=0)
        state, (w1, w2) = nj.pure(model)(state, seed=1)

        # Scaled weights should have smaller magnitude
        assert jnp.abs(w2).mean() < jnp.abs(w1).mean()


class TestEncoderDecoderPattern:
    """Integration tests for encoder-decoder network patterns"""

    def test_encoder_decoder_architecture(self):
        """Test typical encoder-decoder architecture"""

        class Model(nj.Module):
            def __call__(self, x):
                z = self.sub("encoder", nets.MLP, layers=2, units=128)(x)
                x_recon = self.sub("decoder", nets.MLP, layers=2, units=64)(z)
                return x_recon

        model = Model(name="model")
        x = jnp.ones((4, 64), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, x_recon = nj.pure(model)(state, x)

        assert x_recon.shape == (4, 64)

    def test_conv_encoder_decoder(self):
        """Test convolutional encoder-decoder"""

        class Model(nj.Module):
            def __call__(self, x):
                # Encoder: downsampling
                h1 = self.sub("enc1", nets.Conv2D, depth=32, kernel=3, stride=2)(x)
                h2 = self.sub("enc2", nets.Conv2D, depth=64, kernel=3, stride=2)(h1)
                # Decoder: upsampling
                h3 = self.sub(
                    "dec1", nets.Conv2D, depth=32, kernel=3, stride=2, transp=True
                )(h2)
                x_recon = self.sub(
                    "dec2", nets.Conv2D, depth=3, kernel=3, stride=2, transp=True
                )(h3)
                return x_recon

        model = Model(name="model")
        x = jnp.ones((2, 32, 32, 3), dtype=nets.COMPUTE_DTYPE)
        state = nj.init(model)({}, x, seed=0)
        state, x_recon = nj.pure(model)(state, x)

        assert x_recon.shape == (2, 32, 32, 3)

    def test_transformer_sequence_processing(self):
        """Test Transformer for sequence processing"""

        class Model(nj.Module):
            def __call__(self, tokens):
                x = self.sub("embed", nets.Embed, classes=100, units=64)(tokens)
                x = self.sub(
                    "trans", nets.Transformer, units=64, layers=2, heads=4, rope=False
                )(x)
                logits = self.sub("output", nets.Linear, 10)(x)
                return logits

        model = Model(name="model")
        tokens = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)
        state = nj.init(model)({}, tokens, seed=0)
        state, logits = nj.pure(model)(state, tokens)

        assert logits.shape == (2, 4, 10)
