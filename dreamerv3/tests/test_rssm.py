"""
Tests for dreamerv3.rssm - World Model (RSSM, Encoder, Decoder)

Coverage goal: 90% (starting with ~20-30% foundational tests)
"""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest


class TestRSSM:
    """Tests for RSSM (Recurrent State-Space Model)"""

    @pytest.fixture
    def act_space(self):
        """Simple action space for testing"""
        return {
            "action": elements.Space(np.float32, (4,)),
        }

    @pytest.fixture
    def rssm_module(self, act_space):
        """Create a small RSSM module for testing"""
        from dreamerv3 import rssm

        return rssm.RSSM(
            act_space,
            deter=256,  # Smaller than default for faster tests
            hidden=128,
            stoch=8,
            classes=8,
            blocks=8,
            name="test_rssm",
        )

    def test_initialization(self, act_space):
        """Test RSSM can be initialized with valid configuration"""
        from dreamerv3 import rssm

        module = rssm.RSSM(
            act_space,
            deter=256,
            hidden=128,
            stoch=8,
            classes=8,
            blocks=8,
            name="test_rssm",
        )
        assert module.deter == 256
        assert module.hidden == 128
        assert module.stoch == 8
        assert module.classes == 8
        assert module.blocks == 8
        assert module.act_space == act_space

    def test_invalid_blocks_configuration(self, act_space):
        """Test RSSM raises error when deter is not divisible by blocks"""
        from dreamerv3 import rssm

        with pytest.raises(AssertionError):
            rssm.RSSM(
                act_space,
                deter=256,  # Not divisible by 7
                blocks=7,
                name="test_invalid_rssm",
            )

    def test_entry_space(self, rssm_module):
        """Test entry_space returns correct Space definitions"""
        space = rssm_module.entry_space
        assert "deter" in space
        assert "stoch" in space
        assert space["deter"].shape == (256,)
        assert space["stoch"].shape == (8, 8)
        assert space["deter"].dtype == np.float32
        assert space["stoch"].dtype == np.float32

    def test_initial_state(self, rssm_module):
        """Test initial state generation"""
        batch_size = 4
        carry = rssm_module.initial(batch_size)

        assert "deter" in carry
        assert "stoch" in carry
        assert carry["deter"].shape == (batch_size, 256)
        assert carry["stoch"].shape == (batch_size, 8, 8)
        # Default dtype is bfloat16 (from JAX config)
        assert carry["deter"].dtype == jnp.bfloat16
        assert carry["stoch"].dtype == jnp.bfloat16

        # Initial state should be zeros
        assert jnp.allclose(carry["deter"], 0.0)
        assert jnp.allclose(carry["stoch"], 0.0)

    def test_truncate(self, rssm_module):
        """Test truncate extracts last timestep"""
        batch_size = 4
        seq_length = 10

        # Create dummy entries with time dimension using numpy
        entries = {
            "deter": np.ones((batch_size, seq_length, 256), dtype=np.float32),
            "stoch": np.ones((batch_size, seq_length, 8, 8), dtype=np.float32),
        }

        carry = rssm_module.truncate(entries)

        assert carry["deter"].shape == (batch_size, 256)
        assert carry["stoch"].shape == (batch_size, 8, 8)

    def test_starts(self, rssm_module):
        """Test starts reshapes last n timesteps"""
        batch_size = 4
        seq_length = 10
        nlast = 3

        # Create dummy entries with time dimension using numpy
        entries = {
            "deter": np.ones((batch_size, seq_length, 256), dtype=np.float32),
            "stoch": np.ones((batch_size, seq_length, 8, 8), dtype=np.float32),
        }
        carry = rssm_module.initial(batch_size)

        result = rssm_module.starts(entries, carry, nlast)

        expected_batch = batch_size * nlast
        assert result["deter"].shape == (expected_batch, 256)
        assert result["stoch"].shape == (expected_batch, 8, 8)


class TestEncoder:
    """Tests for Encoder module"""

    @pytest.fixture
    def obs_space(self):
        """Simple observation space for testing"""
        return {
            "image": elements.Space(np.uint8, (64, 64, 3)),
            "vector": elements.Space(np.float32, (10,)),
        }

    def test_encoder_initialization(self, obs_space):
        """Test Encoder can be initialized"""
        from dreamerv3 import rssm

        encoder = rssm.Encoder(
            obs_space,
            depth=32,
            mults=(2, 2, 2),
            norm="layer",
            act="gelu",
            name="test_encoder",
        )
        assert encoder.obs_space == obs_space
        assert encoder.depth == 32


class TestDecoder:
    """Tests for Decoder module"""

    @pytest.fixture
    def obs_space(self):
        """Simple observation space for testing"""
        return {
            "image": elements.Space(np.uint8, (64, 64, 3)),
            "vector": elements.Space(np.float32, (10,)),
        }

    def test_decoder_initialization(self, obs_space):
        """Test Decoder can be initialized"""
        from dreamerv3 import rssm

        decoder = rssm.Decoder(
            obs_space,
            depth=32,
            mults=(2, 2, 2),
            norm="layer",
            act="gelu",
            name="test_decoder",
        )
        assert decoder.obs_space == obs_space
        assert decoder.depth == 32
