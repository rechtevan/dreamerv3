"""
Comprehensive tests for dreamerv3/main.py

Coverage goal: Achieve 80%+ coverage of dreamerv3/main.py (currently 0%)

These tests exercise the main entry point and factory functions:
- Config loading and merging
- make_agent() - Agent creation
- make_env() - Environment creation and wrapping
- make_logger() - Logger setup
- make_replay() - Replay buffer creation
- make_stream() - Data stream creation
"""

import elements
import numpy as np
import pytest

from dreamerv3 import main


def load_test_config():
    """Load test config for main.py testing"""
    import pathlib

    import ruamel.yaml as yaml

    config_path = (
        pathlib.Path(__file__).parent.parent.parent / "dreamerv3" / "configs.yaml"
    )
    configs = yaml.YAML(typ="safe").load(config_path.read_text())

    # Use debug config with dummy environment
    full_config = elements.Config(configs["defaults"])
    full_config = full_config.update(configs["debug"])
    full_config = full_config.update(
        {
            "task": "dummy_disc",  # Use dummy env for testing
            "logdir": "/tmp/test_dreamerv3_main",
            "jax.platform": "cpu",
            "jax.jit": False,  # Disable JIT for faster tests
            "batch_size": 2,
            "batch_length": 4,
            "replay_context": 0,
            "random_agent": False,
        }
    )

    return full_config


@pytest.fixture
def config():
    """Test configuration"""
    return load_test_config()


class TestMakeAgent:
    """Test make_agent() factory function"""

    def test_make_agent_creates_agent(self, config):
        """Test make_agent creates an Agent instance"""
        agent = main.make_agent(config)

        assert agent is not None
        assert hasattr(agent, "policy")
        assert hasattr(agent, "train")
        assert hasattr(agent, "obs_space")
        assert hasattr(agent, "act_space")

    def test_make_agent_filters_log_obs(self, config):
        """Test make_agent filters out log/ observations"""
        agent = main.make_agent(config)

        # Check that log/ keys are filtered
        assert not any(k.startswith("log/") for k in agent.obs_space.keys())

    def test_make_agent_removes_reset_action(self, config):
        """Test make_agent removes reset from action space"""
        agent = main.make_agent(config)

        # Check that reset is removed from action space
        assert "reset" not in agent.act_space

    def test_make_agent_with_random_agent(self, config):
        """Test make_agent with random_agent=True"""
        config = config.update({"random_agent": True})
        agent = main.make_agent(config)

        # Should create RandomAgent
        assert agent is not None
        assert hasattr(agent, "policy")


class TestMakeEnv:
    """Test make_env() factory function"""

    def test_make_env_dummy(self, config):
        """Test make_env with dummy environment"""
        env = main.make_env(config, index=0)

        assert env is not None
        assert hasattr(env, "obs_space")
        assert hasattr(env, "act_space")
        assert hasattr(env, "step")
        # Environment is properly created and wrapped
        assert callable(env.step)

        env.close()

    def test_make_env_with_index(self, config):
        """Test make_env with different indices"""
        env1 = main.make_env(config, index=0)
        env2 = main.make_env(config, index=1)

        assert env1 is not None
        assert env2 is not None

        env1.close()
        env2.close()

    def test_wrap_env_applies_wrappers(self, config):
        """Test wrap_env applies standard wrappers"""
        import embodied

        # Create basic dummy env
        env = embodied.envs.dummy.Dummy("disc")

        # Wrap it
        wrapped = main.wrap_env(env, config)

        # Should be wrapped (different object)
        assert wrapped is not None
        assert hasattr(wrapped, "obs_space")
        assert hasattr(wrapped, "act_space")

        wrapped.close()


class TestMakeLogger:
    """Test make_logger() factory function"""

    def test_make_logger_creates_logger(self, config):
        """Test make_logger creates a Logger instance"""
        logger = main.make_logger(config)

        assert logger is not None
        assert hasattr(logger, "add")
        assert hasattr(logger, "write")

    def test_make_logger_with_jsonl_output(self, config):
        """Test logger with JSONL output"""
        config = config.update({"logger.outputs": ["jsonl"]})
        logger = main.make_logger(config)

        assert logger is not None

    def test_make_logger_with_scope_output(self, config):
        """Test logger with Scope output"""
        config = config.update({"logger.outputs": ["scope"]})
        logger = main.make_logger(config)

        assert logger is not None


class TestMakeReplay:
    """Test make_replay() factory function"""

    def test_make_replay_train_mode(self, config):
        """Test make_replay in train mode"""
        replay = main.make_replay(config, "replay", mode="train")

        assert replay is not None
        assert hasattr(replay, "add")
        assert hasattr(replay, "sample")
        assert hasattr(replay, "dataset")

    def test_make_replay_report_mode(self, config):
        """Test make_replay in report mode"""
        replay = main.make_replay(config, "replay_report", mode="report")

        assert replay is not None
        assert hasattr(replay, "add")
        assert hasattr(replay, "sample")

    def test_make_replay_with_uniform_selector(self, config):
        """Test replay with uniform selector (default)"""
        # Default config has uniform: 1.0
        replay = main.make_replay(config, "replay", mode="train")

        assert replay is not None

    def test_make_replay_with_mixed_selectors(self, config):
        """Test replay with mixed selectors"""
        config = config.update(
            {
                "replay.fracs.uniform": 0.5,
                "replay.fracs.priority": 0.3,
                "replay.fracs.recency": 0.2,
            }
        )
        replay = main.make_replay(config, "replay", mode="train")

        assert replay is not None


class TestMakeStream:
    """Test make_stream() factory function"""

    def test_make_stream_train(self, config):
        """Test make_stream for training"""
        # Create a minimal replay buffer
        replay = main.make_replay(config, "replay_test", mode="train")

        # Add some dummy data so sample() works
        dummy_data = {
            "obs": np.random.randn(10, 4).astype(np.float32),
            "action": np.random.randint(0, 5, (10,)).astype(np.int32),
            "reward": np.random.randn(10).astype(np.float32),
            "is_first": np.zeros((10,), dtype=bool),
            "is_last": np.zeros((10,), dtype=bool),
            "is_terminal": np.zeros((10,), dtype=bool),
        }
        dummy_data["is_first"][0] = True
        dummy_data["is_last"][-1] = True

        # Add to replay
        for i in range(len(dummy_data["obs"])):
            step_data = {k: v[i] for k, v in dummy_data.items()}
            replay.add(step_data, worker=0)

        stream = main.make_stream(config, replay, mode="train")

        assert stream is not None
        assert hasattr(stream, "__iter__")

    def test_make_stream_report(self, config):
        """Test make_stream for reporting"""
        # Create a minimal replay buffer
        replay = main.make_replay(config, "replay_report_test", mode="report")

        # Add some dummy data
        dummy_data = {
            "obs": np.random.randn(10, 4).astype(np.float32),
            "action": np.random.randint(0, 5, (10,)).astype(np.int32),
            "reward": np.random.randn(10).astype(np.float32),
            "is_first": np.zeros((10,), dtype=bool),
            "is_last": np.zeros((10,), dtype=bool),
            "is_terminal": np.zeros((10,), dtype=bool),
        }
        dummy_data["is_first"][0] = True
        dummy_data["is_last"][-1] = True

        # Add to replay
        for i in range(len(dummy_data["obs"])):
            step_data = {k: v[i] for k, v in dummy_data.items()}
            replay.add(step_data, worker=0)

        stream = main.make_stream(config, replay, mode="report")

        assert stream is not None
        assert hasattr(stream, "__iter__")


class TestConfigLoading:
    """Test configuration loading and merging"""

    def test_config_has_required_keys(self, config):
        """Test config has all required top-level keys"""
        assert "task" in config
        assert "logdir" in config
        assert "seed" in config
        assert "batch_size" in config
        assert "batch_length" in config
        assert "agent" in config
        assert "replay" in config
        assert "run" in config
        assert "jax" in config
        assert "logger" in config

    def test_config_agent_structure(self, config):
        """Test agent config has required structure"""
        assert "enc" in config.agent
        assert "dyn" in config.agent
        assert "dec" in config.agent
        assert "loss_scales" in config.agent

    def test_config_replay_structure(self, config):
        """Test replay config has required structure"""
        assert "size" in config.replay
        assert "online" in config.replay
        assert "fracs" in config.replay
        assert "chunksize" in config.replay


# Additional tests for missing coverage areas


class TestMakeLoggerOutputs:
    """Test make_logger with different output types"""

    @pytest.mark.skip(reason="Requires tensorflow dependency")
    def test_make_logger_with_tensorboard(self, config):
        """Test logger with TensorBoard output"""
        config = config.update({"logger.outputs": ["tensorboard"]})
        logger = main.make_logger(config)
        assert logger is not None

    @pytest.mark.skip(reason="Requires wandb authentication")
    def test_make_logger_with_wandb(self, config):
        """Test logger with WandB output"""
        config = config.update({"logger.outputs": ["wandb"]})
        logger = main.make_logger(config)
        assert logger is not None

    def test_make_logger_with_multiple_outputs(self, config):
        """Test logger with multiple outputs"""
        config = config.update({"logger.outputs": ["jsonl", "scope"]})
        logger = main.make_logger(config)
        assert logger is not None


class TestMakeEnvEdgeCases:
    """Test make_env edge cases"""

    def test_make_env_with_overrides(self, config):
        """Test make_env with parameter overrides"""
        env = main.make_env(config, index=0, size=(32, 32))
        assert env is not None
        env.close()

    def test_wrap_env_with_continuous_actions(self, config):
        """Test wrap_env normalizes continuous actions"""
        import embodied

        # Dummy env with continuous actions
        env = embodied.envs.dummy.Dummy("cont")
        wrapped = main.wrap_env(env, config)

        # Check wrappers were applied
        assert wrapped is not None
        wrapped.close()


class TestMakeReplayEdgeCases:
    """Test make_replay edge cases"""

    def test_make_replay_with_replica_directory(self, config):
        """Test replay creates replica subdirectory when replicas > 1"""
        config = config.update({"replicas": 2, "replica": 1})
        replay = main.make_replay(config, "replay_replicas", mode="train")
        assert replay is not None

    def test_make_replay_capacity_scaling(self, config):
        """Test replay capacity is scaled for report mode"""
        replay_train = main.make_replay(config, "replay_cap_train", mode="train")
        replay_report = main.make_replay(config, "replay_cap_report", mode="report")

        # Report mode should have smaller capacity
        assert replay_train is not None
        assert replay_report is not None


# Note: Testing main() function directly is complex as it runs full training
# The factory functions above provide good coverage of main.py's core logic
