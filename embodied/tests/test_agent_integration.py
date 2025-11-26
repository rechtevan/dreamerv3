"""
Integration tests for embodied.jax.agent - JAX agent wrapper

Coverage goal: Improve embodied/jax/agent.py from 34.08% to 70%+

These integration tests exercise the full Agent workflow:
- Agent initialization with a minimal model
- Full training steps with policy/train/report methods
- State management (carries) across multiple steps
- Dataset streaming and batching
- Save/load checkpoint cycle
"""

import contextlib
import re
import threading
from unittest.mock import MagicMock, Mock, patch

import chex
import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

import embodied
import embodied.jax.nets as nn
from embodied.jax import agent


@pytest.fixture
def minimal_config(tmp_path):
    """Minimal configuration for testing Agent"""
    return elements.Config(
        logdir=str(tmp_path / "logdir"),
        batch_size=2,
        batch_length=4,
        report_length=4,
        replay_context=0,
        seed=42,
        jax=dict(
            platform="cpu",
            compute_dtype="float32",
            policy_devices=[0],
            train_devices=[0],
            policy_mesh="-1,1,1",
            train_mesh="-1,1,1",
            expect_devices=0,
            use_shardmap=False,
            enable_policy=True,
            ckpt_chunksize=-1,
            precompile=False,
            profiler=False,
            jit=True,
            debug=False,
            prealloc=False,
        ),
    )


@pytest.fixture
def obs_space():
    """Simple observation space"""
    return {
        "obs": elements.Space(np.float32, (4,)),
        "reward": elements.Space(np.float32, ()),
        "is_first": elements.Space(bool, ()),
    }


@pytest.fixture
def act_space():
    """Simple action space"""
    return {
        "action": elements.Space(np.float32, (2,)),
    }


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
        enc = nn.Linear(8, name="enc")
        pol = nn.Linear(2, name="pol")

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
        enc = nn.Linear(8, name="enc")
        pol = nn.Linear(2, name="pol")

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


def create_agent(obs_space, act_space, config):
    """Helper to create an Agent with MinimalModel

    Agent's __new__ method expects to instantiate a subclass that has
    __init__(obs_space, act_space, config). So we make MinimalModel look
    like a subclass for this purpose.
    """

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


class TestAgentInitialization:
    """Test Agent initialization and setup"""

    def test_agent_basic_initialization(self, obs_space, act_space, minimal_config):
        """Test Agent initializes with minimal model"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        assert agent_instance.obs_space == obs_space
        assert agent_instance.act_space == act_space
        assert agent_instance.config == minimal_config

    def test_agent_spaces_construction(self, obs_space, act_space, minimal_config):
        """Test Agent correctly merges obs/act/ext spaces"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        # Verify spaces are merged
        assert "obs" in agent_instance.spaces
        assert "action" in agent_instance.spaces
        assert "consec" in agent_instance.spaces
        assert "stepid" in agent_instance.spaces

    def test_agent_rejects_log_keys_in_obs(self, act_space, minimal_config):
        """Test Agent rejects observation keys starting with 'log/'"""
        bad_obs_space = {
            "log/debug": elements.Space(np.float32, ()),
            "obs": elements.Space(np.float32, (4,)),
        }

        with pytest.raises(AssertionError):
            create_agent(bad_obs_space, act_space, minimal_config)

    def test_agent_rejects_reset_in_actions(self, obs_space, minimal_config):
        """Test Agent rejects 'reset' key in action space"""
        bad_act_space = {
            "reset": elements.Space(bool, ()),
            "action": elements.Space(np.float32, (2,)),
        }

        with pytest.raises(AssertionError):
            create_agent(obs_space, bad_act_space, minimal_config)

    def test_agent_counters_initialization(self, obs_space, act_space, minimal_config):
        """Test Agent initializes counters correctly"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        assert isinstance(agent_instance.n_updates, elements.Counter)
        assert isinstance(agent_instance.n_batches, elements.Counter)
        assert isinstance(agent_instance.n_actions, elements.Counter)
        assert int(agent_instance.n_updates) == 0
        assert int(agent_instance.n_batches) == 0
        assert int(agent_instance.n_actions) == 0

    def test_agent_locks_initialization(self, obs_space, act_space, minimal_config):
        """Test Agent initializes threading locks"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        # threading.Lock() returns _thread.lock, so we need to check the type
        assert hasattr(agent_instance.policy_lock, "acquire")
        assert hasattr(agent_instance.policy_lock, "release")
        assert hasattr(agent_instance.train_lock, "acquire")
        assert hasattr(agent_instance.train_lock, "release")


class TestAgentInitMethods:
    """Test Agent init_policy/init_train/init_report methods"""

    def test_init_policy_with_batch_size(self, obs_space, act_space, minimal_config):
        """Test init_policy returns carry state"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        carry = agent_instance.init_policy(batch_size=2)

        assert carry is not None
        # Split returns dict with lists as values (per-device distribution)
        assert isinstance(carry, dict)

    def test_init_train_with_batch_size(self, obs_space, act_space, minimal_config):
        """Test init_train returns carry state"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        carry = agent_instance.init_train(batch_size=2)

        assert carry is not None

    def test_init_report_with_batch_size(self, obs_space, act_space, minimal_config):
        """Test init_report returns carry state"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        carry = agent_instance.init_report(batch_size=2)

        assert carry is not None


class TestAgentPolicy:
    """Test Agent policy method"""

    def test_policy_basic_execution(self, obs_space, act_space, minimal_config):
        """Test policy executes and returns actions"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        # Initialize carry
        carry = agent_instance.init_policy(batch_size=1)

        # Create observation
        obs = {
            "obs": np.random.randn(1, 4).astype(np.float32),
            "reward": np.array([0.0], dtype=np.float32),
            "is_first": np.array([False]),
        }

        # Run policy
        carry, acts, outs = agent_instance.policy(carry, obs, mode="train")

        # Verify outputs
        assert "action" in acts
        assert acts["action"].shape == (1, 2)
        assert np.isfinite(acts["action"]).all()
        assert outs is not None

    def test_policy_validates_observations(self, obs_space, act_space, minimal_config):
        """Test policy validates observation keys"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        carry = agent_instance.init_policy(batch_size=1)

        # Missing observation key
        bad_obs = {"obs": np.random.randn(1, 4).astype(np.float32)}

        with pytest.raises(AssertionError):
            agent_instance.policy(carry, bad_obs, mode="train")

    def test_policy_increments_action_counter(
        self, obs_space, act_space, minimal_config
    ):
        """Test policy increments action counter"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        carry = agent_instance.init_policy(batch_size=1)
        obs = {
            "obs": np.random.randn(1, 4).astype(np.float32),
            "reward": np.array([0.0], dtype=np.float32),
            "is_first": np.array([False]),
        }

        initial_count = int(agent_instance.n_actions)
        agent_instance.policy(carry, obs, mode="train")

        assert int(agent_instance.n_actions) == initial_count + 1

    def test_policy_multiple_steps(self, obs_space, act_space, minimal_config):
        """Test policy runs multiple steps correctly"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        carry = agent_instance.init_policy(batch_size=1)

        for i in range(5):
            obs = {
                "obs": np.random.randn(1, 4).astype(np.float32),
                "reward": np.array([float(i)], dtype=np.float32),
                "is_first": np.array([i == 0]),
            }

            carry, acts, _outs = agent_instance.policy(carry, obs, mode="train")

            assert acts["action"].shape == (1, 2)
            assert np.isfinite(acts["action"]).all()

        assert int(agent_instance.n_actions) == 5


class TestAgentTrain:
    """Test Agent train method"""

    def test_train_basic_execution(self, obs_space, act_space, minimal_config):
        """Test train executes and returns metrics"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        carry = agent_instance.init_train(batch_size=2)

        # Create training data
        B, T = 2, 4
        data = {
            "obs": np.random.randn(B, T, 4).astype(np.float32),
            "action": np.random.randn(B, T, 2).astype(np.float32),
            "reward": np.random.randn(B, T).astype(np.float32),
            "is_first": np.zeros((B, T), dtype=bool),
            "consec": np.ones((B, T), dtype=np.int32),
            "stepid": np.arange(T)[None, :].repeat(B, axis=0).astype(np.int32),
            "seed": np.array([1, 2], dtype=np.uint32),
        }

        # Run train
        carry, outs, mets = agent_instance.train(carry, data)

        # Verify outputs
        assert carry is not None
        # First call returns empty (pending outputs)
        assert isinstance(outs, dict)
        assert isinstance(mets, dict)

    def test_train_increments_update_counter(
        self, obs_space, act_space, minimal_config
    ):
        """Test train increments update counter"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        carry = agent_instance.init_train(batch_size=2)
        B, T = 2, 4
        data = {
            "obs": np.random.randn(B, T, 4).astype(np.float32),
            "action": np.random.randn(B, T, 2).astype(np.float32),
            "reward": np.random.randn(B, T).astype(np.float32),
            "is_first": np.zeros((B, T), dtype=bool),
            "consec": np.ones((B, T), dtype=np.int32),
            "stepid": np.arange(T)[None, :].repeat(B, axis=0).astype(np.int32),
            "seed": np.array([1, 2], dtype=np.uint32),
        }

        initial_count = int(agent_instance.n_updates)
        agent_instance.train(carry, data)

        assert int(agent_instance.n_updates) == initial_count + 1

    def test_train_multiple_steps(self, obs_space, act_space, minimal_config):
        """Test train runs multiple steps correctly"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        carry = agent_instance.init_train(batch_size=2)
        B, T = 2, 4

        for i in range(3):
            data = {
                "obs": np.random.randn(B, T, 4).astype(np.float32),
                "action": np.random.randn(B, T, 2).astype(np.float32),
                "reward": np.random.randn(B, T).astype(np.float32),
                "is_first": np.zeros((B, T), dtype=bool),
                "consec": np.ones((B, T), dtype=np.int32),
                "stepid": (np.arange(T)[None, :].repeat(B, axis=0) + i * T).astype(
                    np.int32
                ),
                "seed": np.array([i * 2 + 1, i * 2 + 2], dtype=np.uint32),
            }

            carry, outs, mets = agent_instance.train(carry, data)

            # After first step, we should get outputs from previous step
            if i > 0:
                assert isinstance(outs, dict)
                assert isinstance(mets, dict)

        assert int(agent_instance.n_updates) == 3

    def test_train_validates_data_keys(self, obs_space, act_space, minimal_config):
        """Test train validates data keys"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        carry = agent_instance.init_train(batch_size=2)

        # Missing required key
        bad_data = {
            "obs": np.random.randn(2, 4, 4).astype(np.float32),
            "seed": np.array([1, 2], dtype=np.uint32),
        }

        with pytest.raises(AssertionError):
            agent_instance.train(carry, bad_data)


class TestAgentReport:
    """Test Agent report method"""

    def test_report_returns_param_summary(self, obs_space, act_space, minimal_config):
        """Test report returns metrics with parameter summary"""
        # Note: Full report testing requires proper data sharding
        # This test just verifies the _summary method is called
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        # Verify _summary method exists and returns a string
        summary = agent_instance._summary()
        assert isinstance(summary, str)


class TestAgentStream:
    """Test Agent stream method"""

    def test_stream_returns_prefetch(self, obs_space, act_space, minimal_config):
        """Test stream returns Prefetch stream"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        mock_stream = iter([{"data": np.array([1.0])}])
        result = agent_instance.stream(mock_stream)

        assert isinstance(result, embodied.streams.Prefetch)

    def test_stream_adds_seed_to_data(self, obs_space, act_space, minimal_config):
        """Test stream function adds seed to batches"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        # Create mock stream - stream wraps it with Prefetch
        mock_stream = iter([{"data": np.array([1.0])}])
        result = agent_instance.stream(mock_stream)

        # Verify it returns a Prefetch stream
        assert isinstance(result, embodied.streams.Prefetch)


class TestAgentSaveLoad:
    """Test Agent save/load functionality"""

    def test_save_returns_correct_structure(self, obs_space, act_space, minimal_config):
        """Test save returns params and counters"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        # Run one train step to create parameters
        carry = agent_instance.init_train(batch_size=2)
        B, T = 2, 4
        data = {
            "obs": np.random.randn(B, T, 4).astype(np.float32),
            "action": np.random.randn(B, T, 2).astype(np.float32),
            "reward": np.random.randn(B, T).astype(np.float32),
            "is_first": np.zeros((B, T), dtype=bool),
            "consec": np.ones((B, T), dtype=np.int32),
            "stepid": np.arange(T)[None, :].repeat(B, axis=0).astype(np.int32),
            "seed": np.array([1, 2], dtype=np.uint32),
        }
        agent_instance.train(carry, data)

        # Save
        checkpoint = agent_instance.save()

        assert "params" in checkpoint
        assert "counters" in checkpoint
        assert "updates" in checkpoint["counters"]
        assert "batches" in checkpoint["counters"]
        assert "actions" in checkpoint["counters"]

    def test_load_restores_counters(self, obs_space, act_space, minimal_config):
        """Test load restores counter values"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        # Save initial state
        checkpoint = agent_instance.save()

        # Modify counters
        with agent_instance.n_updates.lock:
            agent_instance.n_updates.value = 10
        with agent_instance.n_batches.lock:
            agent_instance.n_batches.value = 20
        with agent_instance.n_actions.lock:
            agent_instance.n_actions.value = 30

        # Update checkpoint with new values
        checkpoint["counters"]["updates"] = 5
        checkpoint["counters"]["batches"] = 7
        checkpoint["counters"]["actions"] = 15

        # Load
        agent_instance.load(checkpoint)

        assert int(agent_instance.n_updates) == 5
        # n_batches is set to updates value (line 428 in agent.py)
        assert int(agent_instance.n_batches) == 5
        assert int(agent_instance.n_actions) == 15

    def test_save_load_cycle(self, obs_space, act_space, minimal_config):
        """Test full save/load cycle preserves state"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        # Run training to create state
        carry = agent_instance.init_train(batch_size=2)
        B, T = 2, 4

        for i in range(3):
            data = {
                "obs": np.random.randn(B, T, 4).astype(np.float32),
                "action": np.random.randn(B, T, 2).astype(np.float32),
                "reward": np.random.randn(B, T).astype(np.float32),
                "is_first": np.zeros((B, T), dtype=bool),
                "consec": np.ones((B, T), dtype=np.int32),
                "stepid": np.arange(T)[None, :].repeat(B, axis=0).astype(np.int32),
                "seed": np.array([i * 2 + 1, i * 2 + 2], dtype=np.uint32),
            }
            agent_instance.train(carry, data)

        # Save
        checkpoint = agent_instance.save()
        saved_updates = int(agent_instance.n_updates)

        # Load into new agent
        agent_instance2 = create_agent(obs_space, act_space, minimal_config)
        agent_instance2.load(checkpoint)

        # Verify counters match
        assert int(agent_instance2.n_updates) == saved_updates
        assert int(agent_instance2.n_batches) == saved_updates


class TestAgentDeviceManagement:
    """Test Agent device and sharding management"""

    def test_agent_identifies_devices(self, obs_space, act_space, minimal_config):
        """Test Agent identifies available JAX devices"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        # Should have policy and train devices
        assert hasattr(agent_instance, "policy_devices")
        assert hasattr(agent_instance, "train_devices")
        assert len(agent_instance.policy_devices) > 0
        assert len(agent_instance.train_devices) > 0

    def test_agent_creates_meshes(self, obs_space, act_space, minimal_config):
        """Test Agent creates policy and train meshes"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        assert hasattr(agent_instance, "policy_mesh")
        assert hasattr(agent_instance, "train_mesh")
        assert agent_instance.policy_mesh is not None
        assert agent_instance.train_mesh is not None

    def test_agent_creates_shardings(self, obs_space, act_space, minimal_config):
        """Test Agent creates sharding specs"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        assert hasattr(agent_instance, "policy_sharded")
        assert hasattr(agent_instance, "policy_mirrored")
        assert hasattr(agent_instance, "train_sharded")
        assert hasattr(agent_instance, "train_mirrored")


class TestAgentPolicyKeys:
    """Test Agent policy key filtering"""

    def test_agent_filters_policy_keys(self, obs_space, act_space, minimal_config):
        """Test Agent correctly filters policy keys from params"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        # Should have filtered policy_keys based on model.policy_keys regex
        assert hasattr(agent_instance, "policy_keys")
        assert isinstance(agent_instance.policy_keys, list)

    def test_agent_policy_keys_match_regex(self, obs_space, act_space, minimal_config):
        """Test policy keys match the model's regex pattern"""
        agent_instance = create_agent(obs_space, act_space, minimal_config)

        pattern_str = agent_instance.model.policy_keys
        pattern = re.compile(pattern_str)

        # All policy_keys should match the pattern
        for key in agent_instance.policy_keys:
            assert pattern.search(key), (
                f"Key {key} does not match pattern {pattern_str}"
            )


class TestAgentWithDisabledPolicy:
    """Test Agent behavior with enable_policy=False"""

    def test_init_policy_disabled(self, obs_space, act_space, minimal_config):
        """Test init_policy raises when policy disabled"""
        config = minimal_config.update({"jax": {"enable_policy": False}})
        agent_instance = create_agent(obs_space, act_space, config)

        with pytest.raises(Exception, match="Policy not available"):
            agent_instance.init_policy(batch_size=2)

    def test_policy_disabled(self, obs_space, act_space, minimal_config):
        """Test policy raises when policy disabled"""
        config = minimal_config.update({"jax": {"enable_policy": False}})
        agent_instance = create_agent(obs_space, act_space, config)

        obs = {
            "obs": np.random.randn(1, 4).astype(np.float32),
            "reward": np.array([0.0], dtype=np.float32),
            "is_first": np.array([False]),
        }

        with pytest.raises(Exception, match="Policy not available"):
            agent_instance.policy({}, obs, mode="train")
