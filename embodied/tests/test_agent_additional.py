"""
Additional targeted tests for embodied.jax.agent - Edge cases and configuration paths

Coverage goal: Improve agent.py from 77.46% toward 85%+

Tests cover missing lines:
- use_shardmap configuration (lines 107-108, 244-245)
- Precompilation paths (lines 231-238)
- Additional initialization scenarios
- Edge cases in policy/train methods
"""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from embodied.jax import agent, nets


class MinimalModel:
    """Minimal model implementation for testing Agent wrapper

    Note: This does NOT inherit from nj.Module because it's the inner model
    that gets wrapped by embodied.jax.Agent. It just needs the right methods.
    """

    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config

    @property
    def policy_keys(self):
        """Regex pattern for policy parameter keys"""
        return "^(enc|pol)/"

    @property
    def ext_space(self):
        """Extra inputs for training (beyond obs/acts)"""
        spaces = {
            "consec": elements.Space(np.int32, (), 0, 100),
            "stepid": elements.Space(np.int32, (), 0, 1000000),
        }
        return spaces

    def init_policy(self, batch_size):
        """Initialize policy carry state"""
        return {"step": jnp.zeros((batch_size,), jnp.int32)}

    def init_train(self, batch_size):
        """Initialize training carry state"""
        return {
            "step": jnp.zeros((batch_size,), jnp.int32),
            "train_state": jnp.zeros((batch_size, 4), jnp.float32),
        }

    def init_report(self, batch_size):
        """Initialize report carry state"""
        return {
            "step": jnp.zeros((batch_size,), jnp.int32),
            "report_state": jnp.zeros((batch_size, 4), jnp.float32),
        }

    def policy(self, carry, obs, mode="train"):
        """Policy inference"""
        B = obs["obs"].shape[0]
        carry = {"step": carry["step"] + 1}

        # Create simple network (will be part of params)
        enc = nets.Linear(8, name="enc")
        pol = nets.Linear(2, name="pol")

        # Forward pass
        feat = enc(obs["obs"])
        logits = pol(feat)

        # Simple policy: output zeros
        acts = {"action": jnp.zeros((B, 2), jnp.float32)}

        # Outputs for logging
        outs = {
            "policy_value": jnp.ones((B,), jnp.float32),
            "finite": {"acts": jnp.isfinite(acts["action"])},
        }

        return carry, acts, outs

    def train(self, carry, data):
        """Training step"""
        B = data["obs"].shape[0]
        carry = {
            "step": carry["step"] + 1,
            "train_state": carry["train_state"] + 0.1,
        }

        # Create trainable parameters
        enc = nets.Linear(8, name="enc")
        pol = nets.Linear(2, name="pol")

        # Forward pass
        feat = enc(data["obs"][:, 0])  # [B, T, 4] -> use first timestep
        _ = pol(feat)

        # Training outputs
        outs = {
            "train_loss": jnp.ones((B,), jnp.float32),
        }

        # Metrics
        mets = {
            "loss": jnp.mean(jnp.ones((B,), jnp.float32)),
            "param_norm": jnp.array(1.0),
        }

        return carry, outs, mets

    def report(self, carry, data):
        """Report metrics"""
        B = data["obs"].shape[0]
        carry = {
            "step": carry["step"] + 1,
            "report_state": carry["report_state"] + 0.1,
        }

        mets = {
            "report_metric": jnp.array(1.0),
            "obs_mean": jnp.mean(data["obs"]),
        }

        return carry, mets


def make_test_config(**overrides):
    """Helper to create a complete config for testing with optional overrides"""
    config_dict = {
        "logdir": "/tmp/test_agent",
        "batch_size": 2,
        "batch_length": 4,
        "report_length": 4,
        "replay_context": 0,
        "seed": 42,
        "jax": {
            "platform": "cpu",
            "compute_dtype": "float32",
            "policy_devices": [0],
            "train_devices": [0],
            "policy_mesh": "-1,1,1",
            "train_mesh": "-1,1,1",
            "use_shardmap": False,
            "enable_policy": True,
            "precompile": False,
            "jit": True,
        },
    }

    # Apply overrides
    for key, value in overrides.items():
        if key == "jax" and isinstance(value, dict):
            config_dict["jax"].update(value)
        else:
            config_dict[key] = value

    return elements.Config(**config_dict)


def create_agent(obs_space, act_space, config):
    """Helper to create Agent with MinimalModel"""

    class MinimalAgent(agent.Agent):
        def __init__(self, obs_space, act_space, config):
            self.obs_space = obs_space
            self.act_space = act_space
            self.config = config
            self.model = MinimalModel(obs_space, act_space, config)

        @property
        def policy_keys(self):
            return self.model.policy_keys

        @property
        def ext_space(self):
            return self.model.ext_space

        def init_policy(self, batch_size):
            return self.model.init_policy(batch_size)

        def init_train(self, batch_size):
            return self.model.init_train(batch_size)

        def init_report(self, batch_size):
            return self.model.init_report(batch_size)

        def policy(self, carry, obs, mode="train"):
            return self.model.policy(carry, obs, mode)

        def train(self, carry, data):
            return self.model.train(carry, data)

        def report(self, carry, data):
            return self.model.report(carry, data)

    return MinimalAgent(obs_space, act_space, config)


class TestAgentShardMapConfiguration:
    """Test Agent with use_shardmap configuration"""

    def test_agent_with_shardmap_disabled(self):
        """Test Agent with use_shardmap=False (default)"""
        obs_space = {"obs": elements.Space(np.float32, (4,))}
        act_space = {"action": elements.Space(np.float32, (2,))}

        config = make_test_config(jax={"use_shardmap": False})

        ag = create_agent(obs_space, act_space, config)

        # Should initialize successfully
        assert ag.jaxcfg.use_shardmap is False
        assert hasattr(ag, "policy_mesh")
        assert hasattr(ag, "train_mesh")

    def test_agent_with_shardmap_enabled(self):
        """Test Agent with use_shardmap=True validates mesh shape"""
        obs_space = {"obs": elements.Space(np.float32, (4,))}
        act_space = {"action": elements.Space(np.float32, (2,))}

        # With default mesh "-1,1,1", shape["d"] should equal size
        config = make_test_config(
            jax={
                "use_shardmap": True,
                "policy_mesh": "1,1,1",  # d=1, f=1, t=1
                "train_mesh": "1,1,1",
            }
        )

        ag = create_agent(obs_space, act_space, config)

        # Should initialize successfully with proper mesh
        assert ag.jaxcfg.use_shardmap is True
        # Lines 107-108 should be covered
        assert ag.train_mesh.shape["d"] == ag.train_mesh.size
        assert ag.policy_mesh.shape["d"] == ag.policy_mesh.size


class TestAgentPrecompilation:
    """Test Agent precompilation paths"""

    def test_agent_with_precompile_enabled(self):
        """Test Agent with precompile=True triggers compilation"""
        obs_space = {"obs": elements.Space(np.float32, (4,))}
        act_space = {"action": elements.Space(np.float32, (2,))}

        config = make_test_config(jax={"precompile": True, "enable_policy": True})

        # This should trigger precompile (lines 231-238)
        ag = create_agent(obs_space, act_space, config)

        # Verify agent was created successfully
        assert hasattr(ag, "_train")
        assert hasattr(ag, "_report")
        assert ag.jaxcfg.precompile is True

    def test_agent_with_precompile_disabled(self):
        """Test Agent with precompile=False skips compilation"""
        obs_space = {"obs": elements.Space(np.float32, (4,))}
        act_space = {"action": elements.Space(np.float32, (2,))}

        config = make_test_config(jax={"precompile": False})

        ag = create_agent(obs_space, act_space, config)

        # Should skip precompilation
        assert ag.jaxcfg.precompile is False


class TestAgentInitPolicyShardMap:
    """Test init_policy with use_shardmap configuration"""

    def test_init_policy_with_shardmap(self):
        """Test init_policy batch size calculation with use_shardmap"""
        obs_space = {"obs": elements.Space(np.float32, (4,))}
        act_space = {"action": elements.Space(np.float32, (2,))}

        config = make_test_config(jax={"use_shardmap": True, "policy_mesh": "1,1,1"})

        ag = create_agent(obs_space, act_space, config)

        # init_policy with use_shardmap should divide batch_size (line 245)
        carry = ag.init_policy(batch_size=4)

        # Should return carry structure
        assert isinstance(carry, dict)
        assert "step" in carry

    def test_init_policy_without_shardmap(self):
        """Test init_policy without use_shardmap"""
        obs_space = {"obs": elements.Space(np.float32, (4,))}
        act_space = {"action": elements.Space(np.float32, (2,))}

        config = make_test_config(jax={"use_shardmap": False})

        ag = create_agent(obs_space, act_space, config)

        carry = ag.init_policy(batch_size=4)

        assert isinstance(carry, dict)


class TestAgentInitTrainShardMap:
    """Test init_train with use_shardmap configuration"""

    def test_init_train_with_shardmap(self):
        """Test init_train batch size calculation with use_shardmap"""
        obs_space = {"obs": elements.Space(np.float32, (4,))}
        act_space = {"action": elements.Space(np.float32, (2,))}

        config = make_test_config(jax={"use_shardmap": True, "train_mesh": "1,1,1"})

        ag = create_agent(obs_space, act_space, config)

        # init_train with use_shardmap should divide batch_size (similar to init_policy)
        carry = ag.init_train(batch_size=4)

        assert isinstance(carry, dict)

    def test_init_report_with_shardmap(self):
        """Test init_report batch size calculation with use_shardmap"""
        obs_space = {"obs": elements.Space(np.float32, (4,))}
        act_space = {"action": elements.Space(np.float32, (2,))}

        config = make_test_config(jax={"use_shardmap": True, "train_mesh": "1,1,1"})

        ag = create_agent(obs_space, act_space, config)

        carry = ag.init_report(batch_size=4)

        assert isinstance(carry, dict)


class TestAgentMeshValidation:
    """Test Agent mesh validation"""

    def test_invalid_mesh_shape_raises_assertion(self):
        """Test Agent raises AssertionError for invalid mesh shapes"""
        obs_space = {"obs": elements.Space(np.float32, (4,))}
        act_space = {"action": elements.Space(np.float32, (2,))}

        # Try to create mesh with TP dimension larger than devices
        # With 1 device, train_devices=(0,), mesh "1,1,2" would require 2 devices
        config = make_test_config(jax={"train_mesh": "1,1,2"})

        # Should raise AssertionError in mesh creation
        with pytest.raises(AssertionError):
            create_agent(obs_space, act_space, config)

    def test_valid_mesh_shape_succeeds(self):
        """Test Agent succeeds with valid mesh shape"""
        obs_space = {"obs": elements.Space(np.float32, (4,))}
        act_space = {"action": elements.Space(np.float32, (2,))}

        config = make_test_config(jax={"policy_mesh": "1,1,1", "train_mesh": "1,1,1"})

        # Should create successfully
        ag = create_agent(obs_space, act_space, config)
        assert ag is not None
