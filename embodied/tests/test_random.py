"""Tests for embodied.core.random - Random agent implementation.

Coverage goal: 100% (from 92%)

Tests cover:
- RandomAgent initialization
- Policy sampling
- Training interface (no-op)
- Report interface (no-op)
- Stream passthrough
- Save/load operations (missing lines 37, 40)
"""

import elements
import numpy as np
import pytest

from embodied.core import random


class TestRandomAgent:
    """Test RandomAgent class"""

    def test_init(self):
        """Test RandomAgent initialization"""
        obs_space = {"image": elements.Space(np.uint8, (64, 64, 3))}
        act_space = {"action": elements.Space(np.int32, (), 0, 4)}

        agent = random.RandomAgent(obs_space, act_space)

        assert agent.obs_space == obs_space
        assert agent.act_space == act_space

    def test_init_policy(self):
        """Test init_policy returns empty tuple"""
        agent = random.RandomAgent({}, {})

        carry = agent.init_policy(batch_size=4)

        assert carry == ()

    def test_init_train(self):
        """Test init_train returns empty tuple"""
        agent = random.RandomAgent({}, {})

        carry = agent.init_train(batch_size=4)

        assert carry == ()

    def test_init_report(self):
        """Test init_report returns empty tuple"""
        agent = random.RandomAgent({}, {})

        carry = agent.init_report(batch_size=4)

        assert carry == ()

    def test_policy_samples_actions(self):
        """Test policy samples random actions from action space"""
        obs_space = {"image": elements.Space(np.uint8, (64, 64, 3))}
        act_space = {
            "move": elements.Space(np.int32, (), 0, 4),
            "jump": elements.Space(np.int32, (), 0, 1),
        }
        agent = random.RandomAgent(obs_space, act_space)

        obs = {"is_first": np.array([True, False, True])}
        carry, actions, outs = agent.policy((), obs, mode="train")

        assert carry == ()
        assert outs == {}
        assert "move" in actions
        assert "jump" in actions
        assert actions["move"].shape == (3,)
        assert actions["jump"].shape == (3,)

    def test_policy_excludes_reset(self):
        """Test policy excludes 'reset' from sampled actions"""
        obs_space = {}
        act_space = {
            "action": elements.Space(np.int32, (), 0, 4),
            "reset": elements.Space(np.bool_, ()),  # Should be excluded
        }
        agent = random.RandomAgent(obs_space, act_space)

        obs = {"is_first": np.array([True])}
        _, actions, _ = agent.policy((), obs, mode="train")

        assert "action" in actions
        assert "reset" not in actions

    def test_train_noop(self):
        """Test train is a no-op returning empty dicts"""
        agent = random.RandomAgent({}, {})

        carry, outs, mets = agent.train((), {})

        assert carry == ()
        assert outs == {}
        assert mets == {}

    def test_report_noop(self):
        """Test report is a no-op returning empty dict"""
        agent = random.RandomAgent({}, {})

        carry, mets = agent.report((), {})

        assert carry == ()
        assert mets == {}

    def test_stream_passthrough(self):
        """Test stream returns input stream unchanged"""
        agent = random.RandomAgent({}, {})
        mock_stream = [1, 2, 3]

        result = agent.stream(mock_stream)

        assert result is mock_stream

    def test_save_returns_none(self):
        """Test save returns None (line 37)"""
        agent = random.RandomAgent({}, {})

        result = agent.save()

        assert result is None

    def test_load_accepts_none(self):
        """Test load accepts None and does nothing (line 40)"""
        agent = random.RandomAgent({}, {})

        # Should not raise any exception
        agent.load(data=None)

    def test_load_accepts_data(self):
        """Test load accepts data and does nothing (line 40)"""
        agent = random.RandomAgent({}, {})

        # Should not raise any exception with actual data
        agent.load(data={"some": "data"})
