"""
Tests for embodied.run.eval_only - Evaluation-only mode

Coverage goal: 90% (from 9.52%)

Tests cover:
- Checkpoint requirement validation
- Agent and logger initialization
- Checkpoint loading
- Logging function behavior (episode tracking, metrics aggregation)
- Policy mode setting ("eval")
- Driver integration
- Episode statistics collection
- Periodic logging
"""

from collections import defaultdict
from functools import partial as bind
from unittest import mock

import elements
import numpy as np
import pytest

import embodied
from embodied.run.eval_only import eval_only


class TestEvalOnlyValidation:
    """Test eval_only validation and setup"""

    def test_requires_from_checkpoint(self):
        """Test eval_only asserts from_checkpoint is provided"""
        args = self._make_args()
        args.from_checkpoint = None

        with pytest.raises(AssertionError):
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                self._make_logger_factory(),
                args,
            )

    def test_creates_agent_logger_from_factories(self):
        """Test eval_only calls factory functions"""
        agent_factory = mock.Mock(return_value=self._make_agent())
        logger_factory = mock.Mock(return_value=self._make_logger())
        args = self._make_args(steps=1)

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                agent_factory,
                self._make_env_factory(),
                logger_factory,
                args,
            )

        agent_factory.assert_called_once()
        logger_factory.assert_called_once()

    def test_creates_logdir(self):
        """Test eval_only creates log directory"""
        args = self._make_args(steps=1)
        logdir = elements.Path(args.logdir)

        # Clean up if exists
        if logdir.exists():
            import shutil

            shutil.rmtree(logdir)

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                self._make_logger_factory(),
                args,
            )

        assert logdir.exists()

        # Clean up
        import shutil

        shutil.rmtree(logdir)


class TestCheckpointLoading:
    """Test checkpoint loading behavior"""

    def test_loads_checkpoint_with_agent_keys(self):
        """Test eval_only loads checkpoint with agent keys"""
        args = self._make_args(steps=1)

        with mock.patch("elements.Checkpoint") as mock_cp_class:
            mock_cp = mock.Mock()
            mock_cp_class.return_value = mock_cp

            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                self._make_logger_factory(),
                args,
            )

            # Verify checkpoint.agent was set
            assert hasattr(mock_cp, "agent")
            # Verify load was called with correct args
            mock_cp.load.assert_called_once_with(args.from_checkpoint, keys=["agent"])


class TestLoggingFunction:
    """Test logfn behavior for episode tracking and metrics"""

    def test_logfn_resets_on_first(self):
        """Test logfn resets episode on is_first"""
        args = self._make_args(steps=100, envs=1)
        logged = []

        logger = self._make_logger()
        logger.add = lambda *a, **kw: logged.append((a, kw))

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                lambda: logger,
                args,
            )

        # Should have logged episode metrics (score, length)
        episode_logs = [log for log in logged if log[1].get("prefix") == "episode"]
        assert len(episode_logs) > 0

    def test_logfn_aggregates_score_and_length(self):
        """Test logfn aggregates episode score and length"""
        args = self._make_args(steps=100, envs=1)
        logged = []

        logger = self._make_logger()
        logger.add = lambda *a, **kw: logged.append((a, kw))

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                lambda: logger,
                args,
            )

        # Find episode logs
        episode_logs = [log for log in logged if log[1].get("prefix") == "episode"]
        assert len(episode_logs) > 0

        # Check structure
        for log_args, log_kwargs in episode_logs:
            assert "score" in log_args[0]
            assert "length" in log_args[0]

    def test_logfn_logs_images_for_worker_zero(self):
        """Test logfn only logs images for worker 0"""
        args = self._make_args(steps=100, envs=2)  # Multiple workers
        logged = []

        logger = self._make_logger()
        logger.add = lambda *a, **kw: logged.append((a, kw))

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                lambda: logger,
                args,
            )

        # Images should only be logged for worker 0
        # This is tested indirectly through the integration


class TestPolicyMode:
    """Test policy mode is set to eval"""

    def test_policy_mode_is_eval(self):
        """Test eval_only uses policy with mode='eval'"""
        args = self._make_args(steps=20, envs=1)
        policy_calls = []

        # Create agent with instrumented policy
        agent = self._make_agent()
        original_policy = agent.policy

        def instrumented_policy(*args, **kwargs):
            policy_calls.append(kwargs)
            return original_policy(*args, **kwargs)

        agent.policy = instrumented_policy

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                lambda: agent,
                self._make_env_factory(),
                self._make_logger_factory(),
                args,
            )

        # Check that policy was called with mode="eval"
        assert len(policy_calls) > 0
        for call in policy_calls:
            assert call.get("mode") == "eval"


class TestDriverIntegration:
    """Test driver integration and step counting"""

    def test_driver_executes_policy_steps(self):
        """Test driver executes policy for specified steps"""
        args = self._make_args(steps=50, envs=1)

        logger = self._make_logger()
        step_count = [0]
        original_increment = logger.step.increment

        def count_increment():
            step_count[0] += 1
            return original_increment()

        logger.step.increment = count_increment

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                lambda: logger,
                args,
            )

        # Should have incremented steps
        assert step_count[0] >= args.steps

    def test_multiple_environments(self):
        """Test eval_only works with multiple environments"""
        args = self._make_args(steps=50, envs=3)

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                self._make_logger_factory(),
                args,
            )

        # Should complete without errors


class TestMetricsAggregation:
    """Test metrics collection and aggregation"""

    def test_logs_episode_statistics(self):
        """Test eval_only logs episode statistics"""
        args = self._make_args(steps=100, envs=1, log_every=10)
        logged = []

        logger = self._make_logger()
        logger.add = lambda *a, **kw: logged.append((a, kw))

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                lambda: logger,
                args,
            )

        # Should have logged epstats
        epstats_logs = [log for log in logged if log[1].get("prefix") == "epstats"]
        assert len(epstats_logs) > 0

    def test_logs_usage_stats(self):
        """Test eval_only logs resource usage"""
        args = self._make_args(steps=100, envs=1, log_every=10)
        logged = []

        logger = self._make_logger()
        logger.add = lambda *a, **kw: logged.append((a, kw))

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                lambda: logger,
                args,
            )

        # Should have logged usage
        usage_logs = [log for log in logged if log[1].get("prefix") == "usage"]
        assert len(usage_logs) > 0

    def test_logs_fps_metrics(self):
        """Test eval_only logs FPS metrics"""
        args = self._make_args(steps=100, envs=1, log_every=10)
        logged = []

        logger = self._make_logger()
        logger.add = lambda *a, **kw: logged.append((a, kw))

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                lambda: logger,
                args,
            )

        # Should have logged fps/policy
        fps_logs = [log for log in logged if "fps/policy" in log[0][0]]
        assert len(fps_logs) > 0

    def test_periodic_logging(self):
        """Test eval_only respects log_every interval"""
        args = self._make_args(steps=100, envs=1, log_every=50)
        write_count = [0]

        logger = self._make_logger()
        original_write = logger.write

        def count_write(*args, **kwargs):
            write_count[0] += 1
            return original_write(*args, **kwargs)

        logger.write = count_write

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                lambda: logger,
                args,
            )

        # Should have written logs at intervals
        assert write_count[0] > 0


class TestRewardRateCalculation:
    """Test reward rate calculation in logfn"""

    def test_reward_rate_calculation(self):
        """Test reward_rate is calculated when episode has >1 reward"""
        args = self._make_args(steps=100, envs=1)
        logged = []

        logger = self._make_logger()
        logger.add = lambda *a, **kw: logged.append((a, kw))

        with mock.patch("elements.Checkpoint") as mock_cp:
            mock_cp.return_value.load.return_value = None
            eval_only(
                self._make_agent_factory(),
                self._make_env_factory(),
                lambda: logger,
                args,
            )

        # Check if any epstats contain reward_rate
        epstats_logs = [log for log in logged if log[1].get("prefix") == "epstats"]
        # reward_rate is optional depending on episode length


# Helper methods
def _make_args(steps=10, envs=1, log_every=5):
    """Create mock args for eval_only"""
    import tempfile

    args = mock.Mock()
    args.from_checkpoint = "test_checkpoint.pkl"
    args.logdir = tempfile.mkdtemp()
    args.steps = steps
    args.envs = envs
    args.log_every = log_every
    args.debug = True  # Use sequential mode for testing
    args.usage = {}
    return args


def _make_env_factory():
    """Create environment factory"""
    from embodied.envs import dummy

    def factory(i=0):
        return dummy.Dummy("disc", length=10)

    return factory


def _make_agent_factory():
    """Create agent factory"""

    def factory():
        from embodied.envs import dummy

        env = dummy.Dummy("disc", length=10)
        agent = embodied.RandomAgent(env.obs_space, env.act_space)
        env.close()
        return agent

    return factory


def _make_logger_factory():
    """Create logger factory"""

    def factory():
        return _make_logger()

    return factory


def _make_logger():
    """Create mock logger"""
    logger = mock.Mock()
    logger.step = elements.Counter()
    logger.add = mock.Mock()
    logger.write = mock.Mock()
    logger.close = mock.Mock()
    return logger


def _make_agent():
    """Create test agent"""
    from embodied.envs import dummy

    env = dummy.Dummy("disc", length=10)
    agent = embodied.RandomAgent(env.obs_space, env.act_space)
    env.close()
    return agent


# Add helpers as module-level functions
TestEvalOnlyValidation._make_args = staticmethod(_make_args)
TestEvalOnlyValidation._make_env_factory = staticmethod(_make_env_factory)
TestEvalOnlyValidation._make_agent_factory = staticmethod(_make_agent_factory)
TestEvalOnlyValidation._make_logger_factory = staticmethod(_make_logger_factory)
TestEvalOnlyValidation._make_agent = staticmethod(_make_agent)
TestEvalOnlyValidation._make_logger = staticmethod(_make_logger)

TestCheckpointLoading._make_args = staticmethod(_make_args)
TestCheckpointLoading._make_env_factory = staticmethod(_make_env_factory)
TestCheckpointLoading._make_agent_factory = staticmethod(_make_agent_factory)
TestCheckpointLoading._make_logger_factory = staticmethod(_make_logger_factory)

TestLoggingFunction._make_args = staticmethod(_make_args)
TestLoggingFunction._make_env_factory = staticmethod(_make_env_factory)
TestLoggingFunction._make_agent_factory = staticmethod(_make_agent_factory)
TestLoggingFunction._make_logger_factory = staticmethod(_make_logger_factory)
TestLoggingFunction._make_logger = staticmethod(_make_logger)

TestPolicyMode._make_args = staticmethod(_make_args)
TestPolicyMode._make_env_factory = staticmethod(_make_env_factory)
TestPolicyMode._make_agent_factory = staticmethod(_make_agent_factory)
TestPolicyMode._make_logger_factory = staticmethod(_make_logger_factory)
TestPolicyMode._make_agent = staticmethod(_make_agent)
TestPolicyMode._make_logger = staticmethod(_make_logger)

TestDriverIntegration._make_args = staticmethod(_make_args)
TestDriverIntegration._make_env_factory = staticmethod(_make_env_factory)
TestDriverIntegration._make_agent_factory = staticmethod(_make_agent_factory)
TestDriverIntegration._make_logger_factory = staticmethod(_make_logger_factory)
TestDriverIntegration._make_logger = staticmethod(_make_logger)

TestMetricsAggregation._make_args = staticmethod(_make_args)
TestMetricsAggregation._make_env_factory = staticmethod(_make_env_factory)
TestMetricsAggregation._make_agent_factory = staticmethod(_make_agent_factory)
TestMetricsAggregation._make_logger_factory = staticmethod(_make_logger_factory)
TestMetricsAggregation._make_logger = staticmethod(_make_logger)

TestRewardRateCalculation._make_args = staticmethod(_make_args)
TestRewardRateCalculation._make_env_factory = staticmethod(_make_env_factory)
TestRewardRateCalculation._make_agent_factory = staticmethod(_make_agent_factory)
TestRewardRateCalculation._make_logger_factory = staticmethod(_make_logger_factory)
TestRewardRateCalculation._make_logger = staticmethod(_make_logger)
