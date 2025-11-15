"""Tests for embodied.core.base - Base classes for Agent, Env, and Stream.

Coverage goal: 100% (from 89.29%)

Tests cover:
- Agent base class initialization
- Env.__repr__ for string representation
- Stream.__iter__ return value
- NotImplementedError assertions for abstract methods
"""

import pytest

from embodied.core import base


class TestAgent:
    """Tests for Agent base class."""

    def test_init_creates_instance(self):
        """Test Agent can be instantiated with spaces and config."""
        obs_space = {"image": None, "reward": None}
        act_space = {"action": None}
        config = {"learning_rate": 0.001}

        agent = base.Agent(obs_space, act_space, config)

        # Should create instance without error
        assert isinstance(agent, base.Agent)

    def test_init_train_not_implemented(self):
        """Test init_train raises NotImplementedError."""
        agent = base.Agent({}, {}, {})

        with pytest.raises(NotImplementedError):
            agent.init_train(batch_size=1)

    def test_init_report_not_implemented(self):
        """Test init_report raises NotImplementedError."""
        agent = base.Agent({}, {}, {})

        with pytest.raises(NotImplementedError):
            agent.init_report(batch_size=1)

    def test_init_policy_not_implemented(self):
        """Test init_policy raises NotImplementedError."""
        agent = base.Agent({}, {}, {})

        with pytest.raises(NotImplementedError):
            agent.init_policy(batch_size=1)

    def test_train_not_implemented(self):
        """Test train raises NotImplementedError."""
        agent = base.Agent({}, {}, {})

        with pytest.raises(NotImplementedError):
            agent.train(carry=None, data={})

    def test_report_not_implemented(self):
        """Test report raises NotImplementedError."""
        agent = base.Agent({}, {}, {})

        with pytest.raises(NotImplementedError):
            agent.report(carry=None, data={})

    def test_policy_not_implemented(self):
        """Test policy raises NotImplementedError."""
        agent = base.Agent({}, {}, {})

        with pytest.raises(NotImplementedError):
            agent.policy(carry=None, obs={}, mode="train")

    def test_stream_not_implemented(self):
        """Test stream raises NotImplementedError."""
        agent = base.Agent({}, {}, {})

        with pytest.raises(NotImplementedError):
            agent.stream(st=None)

    def test_save_not_implemented(self):
        """Test save raises NotImplementedError."""
        agent = base.Agent({}, {}, {})

        with pytest.raises(NotImplementedError):
            agent.save()

    def test_load_not_implemented(self):
        """Test load raises NotImplementedError."""
        agent = base.Agent({}, {}, {})

        with pytest.raises(NotImplementedError):
            agent.load(data={})


class TestEnv:
    """Tests for Env base class."""

    def test_repr_format(self):
        """Test Env.__repr__ returns formatted string."""

        class DummyEnv(base.Env):
            @property
            def obs_space(self):
                return {"image": "space1", "reward": "space2"}

            @property
            def act_space(self):
                return {"action": "space3"}

            def step(self, action):
                return {}

        env = DummyEnv()
        repr_str = repr(env)

        # Should contain class name and spaces
        assert "DummyEnv" in repr_str
        assert "obs_space=" in repr_str
        assert "act_space=" in repr_str

    def test_obs_space_not_implemented(self):
        """Test obs_space raises NotImplementedError."""
        env = base.Env()

        with pytest.raises(NotImplementedError):
            _ = env.obs_space

    def test_act_space_not_implemented(self):
        """Test act_space raises NotImplementedError."""
        env = base.Env()

        with pytest.raises(NotImplementedError):
            _ = env.act_space

    def test_step_not_implemented(self):
        """Test step raises NotImplementedError."""
        env = base.Env()

        with pytest.raises(NotImplementedError):
            env.step(action={})

    def test_close_does_nothing(self):
        """Test close method exists and does nothing by default."""
        env = base.Env()

        # Should not raise any exception
        env.close()


class TestStream:
    """Tests for Stream base class."""

    def test_iter_returns_self(self):
        """Test __iter__ returns self."""
        stream = base.Stream()

        result = iter(stream)

        assert result is stream

    def test_next_not_implemented(self):
        """Test __next__ raises NotImplementedError."""
        stream = base.Stream()

        with pytest.raises(NotImplementedError):
            next(stream)

    def test_save_not_implemented(self):
        """Test save raises NotImplementedError."""
        stream = base.Stream()

        with pytest.raises(NotImplementedError):
            stream.save()

    def test_load_not_implemented(self):
        """Test load raises NotImplementedError."""
        stream = base.Stream()

        with pytest.raises(NotImplementedError):
            stream.load(state={})
