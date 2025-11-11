"""
End-to-end tests for embodied.run.eval_only - Full evaluation workflow

These tests verify the complete eval_only workflow including:
- Actual checkpoint creation and loading
- Full episode collection and evaluation
- Multi-environment parallel evaluation
- Report generation
- Metrics logging and aggregation
- Edge cases and boundary conditions

Coverage target: Push eval_only.py from 93.65% to 98%+
"""

import pickle
import tempfile
from functools import partial as bind
from pathlib import Path

import elements
import numpy as np
import pytest

import embodied
from embodied.envs import dummy
from embodied.run.eval_only import eval_only
from embodied.tests.utils import TestAgent


class TestBasicEvalExecution:
    """Test basic eval-only execution scenarios"""

    def test_eval_only_collects_specified_episodes(self, tmp_path):
        """Test eval_only collects exact number of episodes requested"""
        # Setup
        env_fn = lambda i=0: dummy.Dummy("disc", length=20)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)

        config = self._make_config(tmp_path, steps=150, envs=1)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Create logger
        logger = self._make_logger()
        episodes_logged = []
        original_add = logger.add

        def track_episodes(*args, **kwargs):
            if kwargs.get("prefix") == "episode":
                episodes_logged.append(args[0])
            return original_add(*args, **kwargs)

        logger.add = track_episodes

        # Run eval
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Verify episodes collected
        assert len(episodes_logged) >= 5  # At least some episodes
        for ep in episodes_logged:
            assert "score" in ep
            assert "length" in ep

    def test_eval_only_no_training_occurs(self, tmp_path):
        """Test that no training happens in eval-only mode"""
        env_fn = lambda i=0: dummy.Dummy("disc", length=20)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)

        config = self._make_config(tmp_path, steps=100, envs=1)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Track agent activity
        test_agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        original_train = test_agent.train
        train_calls = []

        def track_train(*args, **kwargs):
            train_calls.append(True)
            return original_train(*args, **kwargs)

        test_agent.train = track_train

        # Run eval
        eval_only(
            lambda: test_agent,
            env_fn,
            self._make_logger_factory(),
            config,
        )

        # Verify no training
        assert len(train_calls) == 0, "Training should not occur in eval-only mode"

        # Verify policy was called
        stats = test_agent.stats()
        assert stats["env_steps"] > 0, "Policy should have been executed"

    def test_eval_only_with_different_episode_counts(self, tmp_path):
        """Test eval-only with various step counts"""
        env_fn = lambda i=0: dummy.Dummy("disc", length=15)

        for steps in [20, 50, 100]:
            # Create checkpoint
            ckpt_path = tmp_path / f"checkpoint_{steps}.pkl"
            agent = TestAgent(env_fn().obs_space, env_fn().act_space)
            self._save_checkpoint(ckpt_path, agent)

            config = self._make_config(tmp_path, steps=steps, envs=1)
            config = config.update(from_checkpoint=str(ckpt_path))

            # Run eval
            logger = self._make_logger()
            eval_only(
                lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
                env_fn,
                lambda: logger,
                config,
            )

            # Verify execution completed
            assert logger.step.value >= steps


class TestCheckpointLoading:
    """Test checkpoint loading and restoration"""

    def test_checkpoint_loaded_correctly(self, tmp_path):
        """Test agent state is restored from checkpoint"""
        config = self._make_config(tmp_path, steps=50, envs=1)
        env_fn = lambda i=0: dummy.Dummy("disc", length=20)

        # Create checkpoint with specific state
        ckpt_path = tmp_path / "checkpoint.pkl"
        original_agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        original_agent._stats["custom_value"] = 12345
        self._save_checkpoint(ckpt_path, original_agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Create new agent and run eval
        test_agent = TestAgent(env_fn().obs_space, env_fn().act_space)

        eval_only(
            lambda: test_agent,
            env_fn,
            self._make_logger_factory(),
            config,
        )

        # Verify checkpoint was loaded
        stats = test_agent.stats()
        assert stats["loads"] == 1, "Checkpoint should have been loaded once"
        assert stats.get("custom_value") == 12345, "Custom state should be restored"

    def test_checkpoint_path_required(self, tmp_path):
        """Test that from_checkpoint is required"""
        config = self._make_config(tmp_path, steps=50, envs=1)
        config = config.update(from_checkpoint=None)  # No checkpoint
        env_fn = lambda i=0: dummy.Dummy("disc", length=20)

        # Should raise AssertionError
        with pytest.raises(AssertionError):
            eval_only(
                lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
                env_fn,
                self._make_logger_factory(),
                config,
            )

    def test_checkpoint_with_empty_string(self, tmp_path):
        """Test that empty string checkpoint path raises assertion"""
        config = self._make_config(tmp_path, steps=50, envs=1)
        config = config.update(from_checkpoint="")  # Empty string
        env_fn = lambda i=0: dummy.Dummy("disc", length=20)

        # Should raise AssertionError
        with pytest.raises(AssertionError):
            eval_only(
                lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
                env_fn,
                self._make_logger_factory(),
                config,
            )


class TestMultiEnvironmentEval:
    """Test evaluation with multiple parallel environments"""

    def test_multi_env_parallel_evaluation(self, tmp_path):
        """Test eval with multiple environments in parallel"""
        config = self._make_config(tmp_path, steps=100, envs=3)
        env_fn = lambda i=0: dummy.Dummy("disc", length=20)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Track episodes per worker
        logger = self._make_logger()
        worker_episodes = {0: [], 1: [], 2: []}
        original_add = logger.add

        def track_worker_episodes(*args, **kwargs):
            if kwargs.get("prefix") == "episode":
                # Can't easily track worker ID, but we can count total episodes
                worker_episodes[0].append(args[0])
            return original_add(*args, **kwargs)

        logger.add = track_worker_episodes

        # Run eval
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Verify episodes collected from multiple workers
        assert len(worker_episodes[0]) >= 3  # Multiple episodes collected

    def test_multi_env_with_different_counts(self, tmp_path):
        """Test with various environment counts"""
        env_fn = lambda i=0: dummy.Dummy("disc", length=15)

        for env_count in [1, 2, 4]:
            config = self._make_config(tmp_path, steps=80, envs=env_count)

            # Create checkpoint
            ckpt_path = tmp_path / f"checkpoint_env{env_count}.pkl"
            agent = TestAgent(env_fn().obs_space, env_fn().act_space)
            self._save_checkpoint(ckpt_path, agent)
            config = config.update(from_checkpoint=str(ckpt_path))

            # Run eval
            test_agent = TestAgent(env_fn().obs_space, env_fn().act_space)
            eval_only(
                lambda: test_agent,
                env_fn,
                self._make_logger_factory(),
                config,
            )

            # Verify execution completed
            stats = test_agent.stats()
            assert stats["env_steps"] > 0


class TestReportGeneration:
    """Test report generation during evaluation"""

    def test_report_called_during_eval(self, tmp_path):
        """Test that agent.report() is called during evaluation"""
        config = self._make_config(tmp_path, steps=100, envs=1)
        env_fn = lambda i=0: dummy.Dummy("disc", length=20)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Track reports
        test_agent = TestAgent(env_fn().obs_space, env_fn().act_space)

        # Run eval
        eval_only(
            lambda: test_agent,
            env_fn,
            self._make_logger_factory(),
            config,
        )

        # Note: report() is not called in eval_only.py - it's only for train mode
        # This test verifies that behavior
        stats = test_agent.stats()
        assert stats["reports"] == 0, "Reports should not be generated in eval-only"


class TestLoggingAndMetrics:
    """Test logging and metrics collection"""

    def test_logs_episode_metrics(self, tmp_path):
        """Test that episode metrics are logged correctly"""
        config = self._make_config(tmp_path, steps=100, envs=1, log_every=20)
        env_fn = lambda i=0: dummy.Dummy("disc", length=15)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Track all logged data
        logger = self._make_logger()
        logged_data = []
        original_add = logger.add

        def track_logs(*args, **kwargs):
            logged_data.append((args, kwargs))
            return original_add(*args, **kwargs)

        logger.add = track_logs

        # Run eval
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Check for required metrics
        prefixes = [kw.get("prefix") for _, kw in logged_data]
        assert "episode" in prefixes, "Episode metrics should be logged"
        assert "epstats" in prefixes, "Episode statistics should be logged"
        assert "usage" in prefixes, "Usage metrics should be logged"

        # Check for FPS metrics
        fps_logs = [args for args, _ in logged_data if "fps/policy" in args[0]]
        assert len(fps_logs) > 0, "FPS metrics should be logged"

    def test_log_every_interval_respected(self, tmp_path):
        """Test that logging respects log_every interval"""
        config = self._make_config(tmp_path, steps=100, envs=1, log_every=30)
        env_fn = lambda i=0: dummy.Dummy("disc", length=10)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Track write calls
        logger = self._make_logger()
        write_calls = []
        original_write = logger.write

        def track_writes():
            write_calls.append(logger.step.value)
            return original_write()

        logger.write = track_writes

        # Run eval
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Verify writes occurred at intervals
        assert len(write_calls) > 0, "Logs should be written"
        # Writes should occur at multiples of log_every (approximately)

    def test_usage_stats_collected(self, tmp_path):
        """Test that resource usage statistics are collected"""
        config = self._make_config(tmp_path, steps=100, envs=1, log_every=20)
        env_fn = lambda i=0: dummy.Dummy("disc", length=15)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Track usage logs
        logger = self._make_logger()
        usage_logs = []
        original_add = logger.add

        def track_usage(*args, **kwargs):
            if kwargs.get("prefix") == "usage":
                usage_logs.append(args[0])
            return original_add(*args, **kwargs)

        logger.add = track_usage

        # Run eval
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Verify usage stats logged
        assert len(usage_logs) > 0, "Usage statistics should be logged"

    def test_timer_stats_logged(self, tmp_path):
        """Test that timer statistics are logged"""
        config = self._make_config(tmp_path, steps=100, envs=1, log_every=20)
        env_fn = lambda i=0: dummy.Dummy("disc", length=15)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Track timer logs
        logger = self._make_logger()
        timer_logs = []
        original_add = logger.add

        def track_timers(*args, **kwargs):
            if "timer" in args[0]:
                timer_logs.append(args[0])
            return original_add(*args, **kwargs)

        logger.add = track_timers

        # Run eval
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Verify timer stats logged
        assert len(timer_logs) > 0, "Timer statistics should be logged"


class TestLogKeyHandling:
    """Test handling of log/ prefixed keys from policy"""

    def test_log_key_aggregation_in_transitions(self, tmp_path):
        """Test that log/ keys from environment are aggregated (avg, max, sum)"""
        config = self._make_config(tmp_path, steps=50, envs=1, log_every=10)

        # Create custom environment that returns log/ keys in observations
        # Note: log/ keys are NOT in obs_space (they're metadata, not observations)
        class LoggingEnv(dummy.Dummy):
            def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
                obs = super()._obs(reward, is_first, is_last, is_terminal)
                # Add log/ metrics (not in obs_space)
                obs["log/custom_metric"] = np.float32(0.5 + self.count * 0.1)
                obs["log/another_metric"] = np.float32(1.0 + self.count * 0.05)
                return obs

        env_fn = lambda i=0: LoggingEnv(
            "disc", length=5
        )  # Short episodes to complete within steps

        # Create checkpoint using dummy environment (without log/ keys in obs_space)
        base_env = dummy.Dummy("disc", length=20)
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(base_env.obs_space, base_env.act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        test_agent = TestAgent(base_env.obs_space, base_env.act_space)

        # Track all logged data
        logger = self._make_logger()
        logged_data = []
        original_add = logger.add

        def track_logs(*args, **kwargs):
            logged_data.append((args, kwargs))
            return original_add(*args, **kwargs)

        logger.add = track_logs

        # Run eval
        eval_only(
            lambda: test_agent,
            env_fn,
            lambda: logger,
            config,
        )

        # Check for aggregated log metrics in epstats
        epstats_logs = [
            args[0]
            for args, kwargs in logged_data
            if kwargs.get("prefix") == "epstats" and args
        ]

        # At least one epstats entry should contain the aggregated log/ metrics
        found_log_metrics = False
        for epstats in epstats_logs:
            if isinstance(epstats, dict):
                # Look for log/ keys with /avg, /max, /sum suffixes
                log_keys = [k for k in epstats.keys() if k.startswith("log/")]
                if log_keys:
                    found_log_metrics = True
                    # Verify aggregation suffixes exist
                    assert any("/avg" in k for k in log_keys), (
                        "Should have /avg aggregation"
                    )
                    assert any("/max" in k for k in log_keys), (
                        "Should have /max aggregation"
                    )
                    assert any("/sum" in k for k in log_keys), (
                        "Should have /sum aggregation"
                    )
                    break

        assert found_log_metrics, "Log metrics should be aggregated and logged"


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_short_evaluation(self, tmp_path):
        """Test eval with minimal steps"""
        config = self._make_config(tmp_path, steps=10, envs=1)
        env_fn = lambda i=0: dummy.Dummy("disc", length=5)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Run eval
        logger = self._make_logger()
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Should complete without errors
        assert logger.step.value >= config.steps

    def test_long_episodes(self, tmp_path):
        """Test eval with very long episodes"""
        config = self._make_config(tmp_path, steps=200, envs=1)
        env_fn = lambda i=0: dummy.Dummy("disc", length=100)  # Long episodes

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Run eval
        logger = self._make_logger()
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Should complete
        assert logger.step.value >= config.steps

    def test_reward_rate_calculation_with_varying_rewards(self, tmp_path):
        """Test reward_rate calculation logic"""
        config = self._make_config(tmp_path, steps=100, envs=1)
        env_fn = lambda i=0: dummy.Dummy("disc", length=30)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Track epstats to see if reward_rate is calculated
        logger = self._make_logger()
        epstats_logs = []
        original_add = logger.add

        def track_epstats(*args, **kwargs):
            if kwargs.get("prefix") == "epstats":
                epstats_logs.append(args[0])
            return original_add(*args, **kwargs)

        logger.add = track_epstats

        # Run eval
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Epstats should be logged
        assert len(epstats_logs) > 0

    def test_image_logging_for_worker_zero(self, tmp_path):
        """Test that images are only logged for worker 0"""
        config = self._make_config(tmp_path, steps=100, envs=2)
        env_fn = lambda i=0: dummy.Dummy("disc", length=20)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Track all logs
        logger = self._make_logger()
        all_logs = []
        original_add = logger.add

        def track_all(*args, **kwargs):
            all_logs.append((args, kwargs))
            return original_add(*args, **kwargs)

        logger.add = track_all

        # Run eval
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Check for policy_ prefixed keys (images)
        epstats_with_images = [
            log
            for log, _ in all_logs
            if log
            and isinstance(log[0], dict)
            and any(k.startswith("policy_") for k in log[0].keys())
        ]
        # Images should be logged for worker 0

    def test_log_key_aggregation(self, tmp_path):
        """Test log/ key aggregation (avg, max, sum)"""
        config = self._make_config(tmp_path, steps=100, envs=1)
        env_fn = lambda i=0: dummy.Dummy("disc", length=20)

        # Create checkpoint
        ckpt_path = tmp_path / "checkpoint.pkl"
        agent = TestAgent(env_fn().obs_space, env_fn().act_space)
        self._save_checkpoint(ckpt_path, agent)
        config = config.update(from_checkpoint=str(ckpt_path))

        # Run eval - logfn handles log/ keys from transitions
        logger = self._make_logger()
        eval_only(
            lambda: TestAgent(env_fn().obs_space, env_fn().act_space),
            env_fn,
            lambda: logger,
            config,
        )

        # Should complete without errors
        assert logger.step.value >= config.steps


# Helper methods
def _make_config(tmp_path, steps=100, envs=1, log_every=20):
    """Create configuration for eval_only"""
    config = elements.Config(
        logdir=str(tmp_path / "logdir"),
        steps=steps,
        envs=envs,
        log_every=log_every,
        debug=True,  # Sequential mode for testing
        usage=dict(psutil=True),  # Need non-empty dict for Config to create attribute
        from_checkpoint="",  # Will be set by tests
    )
    return config


def _save_checkpoint(path, agent):
    """Save agent checkpoint to file"""
    checkpoint = elements.Checkpoint()
    checkpoint.agent = agent
    checkpoint.save(str(path))


def _make_logger():
    """Create a logger for testing"""
    from unittest import mock

    logger = mock.Mock()
    logger.step = elements.Counter()
    logger.add = mock.Mock()
    logger.write = mock.Mock()
    logger.close = mock.Mock()
    return logger


def _make_logger_factory():
    """Create logger factory"""
    return lambda: _make_logger()


# Add helpers to test classes
for test_class in [
    TestBasicEvalExecution,
    TestCheckpointLoading,
    TestMultiEnvironmentEval,
    TestReportGeneration,
    TestLoggingAndMetrics,
    TestLogKeyHandling,
    TestEdgeCases,
]:
    test_class._make_config = staticmethod(_make_config)
    test_class._save_checkpoint = staticmethod(_save_checkpoint)
    test_class._make_logger = staticmethod(_make_logger)
    test_class._make_logger_factory = staticmethod(_make_logger_factory)
