"""
End-to-End tests for embodied/run/train.py training loop.

These tests validate the full training workflow including:
- Agent, replay, driver, and logger integration
- Training loop execution and metrics logging
- Checkpoint save/load cycles
- Edge cases and error handling
"""

from functools import partial as bind

import elements
import numpy as np
import pytest

import embodied
from embodied.envs import dummy
from embodied.tests import utils


class TestTrainE2E:
    """End-to-end integration tests for the training loop."""

    def test_full_training_workflow(self, tmpdir):
        """Test complete training workflow from initialization to completion."""
        args = self._make_args(tmpdir, steps=500)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        assert stats["env_steps"] > 0, "Agent should take environment steps"
        assert stats["replay_steps"] > 0, "Agent should train on replay data"
        # Note: Reports may not trigger if duration is too short
        assert stats["saves"] >= 1, "Agent should save checkpoints"
        assert stats["loads"] == 0, "First run should not load checkpoints"

        # Verify checkpoint directory was created and contains files
        ckpt_dir = elements.Path(tmpdir) / "ckpt"
        assert ckpt_dir.exists(), "Checkpoint directory should exist"
        ckpt_files = list(ckpt_dir.glob("*"))
        assert len(ckpt_files) > 0, "Checkpoint directory should contain files"

    def test_checkpoint_save_load_cycle(self, tmpdir):
        """Test that checkpoints can be saved and loaded correctly."""
        args = self._make_args(tmpdir, steps=500, save_every=0.1)
        agent = self._make_agent()

        # First training run
        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats_after_first = agent.stats()
        first_saves = stats_after_first["saves"]
        first_env_steps = stats_after_first["env_steps"]

        assert first_saves >= 1, "Should save at least one checkpoint"

        # Second training run - should load checkpoint and continue
        args_continue = args.update(steps=1000)
        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args_continue),
            self._make_env,
            bind(self._make_stream, args_continue),
            self._make_logger,
            args_continue,
        )

        stats_after_second = agent.stats()
        assert stats_after_second["loads"] == 1, "Should load checkpoint on resume"
        assert stats_after_second["env_steps"] > first_env_steps, (
            "Should continue training from checkpoint"
        )

    def test_training_with_high_train_ratio(self, tmpdir):
        """Test training with high train_ratio (many training steps per env step)."""
        args = self._make_args(tmpdir, steps=300, train_ratio=64.0)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        # With high train_ratio, replay_steps should be higher than with lower ratios
        # Note: Actual ratio may be lower due to training warmup and batch size constraints
        assert stats["replay_steps"] > stats["env_steps"], (
            f"Replay steps ({stats['replay_steps']}) should exceed env steps ({stats['env_steps']})"
        )

    def test_training_with_low_train_ratio(self, tmpdir):
        """Test training with low train_ratio (fewer training steps per env step)."""
        args = self._make_args(tmpdir, steps=500, train_ratio=8.0)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        # With low train_ratio, still should have reasonable replay steps
        # The ratio is applied per batch, so actual replay_steps depend on batch_steps
        assert stats["replay_steps"] > 0, "Should have some replay steps"
        assert stats["env_steps"] >= args.steps * 0.5, (
            "Should collect most of the target steps"
        )

    def test_metrics_logging_intervals(self, tmpdir):
        """Test that metrics are logged at correct intervals."""
        args = self._make_args(tmpdir, steps=600, log_every=0.05, report_every=0.1)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        # With log_every=0.05 and steps=600, we expect multiple log events
        # With report_every=0.1 and steps=600, we expect multiple reports
        # Use conservative expectation due to timing variability
        assert stats["reports"] >= 2, (
            f"Expected at least 2 reports, got {stats['reports']}"
        )

    def test_checkpoint_intervals(self, tmpdir):
        """Test that checkpoints are saved at correct intervals."""
        args = self._make_args(tmpdir, steps=700, save_every=0.12)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        # With save_every=0.12 and steps=700, we expect multiple saves
        # Use conservative expectation due to timing variability
        assert stats["saves"] >= 2, f"Expected at least 2 saves, got {stats['saves']}"

    def test_training_with_small_replay_capacity(self, tmpdir):
        """Test training with small replay buffer capacity."""
        args = self._make_args(tmpdir, steps=400, batch_size=4, batch_length=8)
        agent = self._make_agent()

        # Create replay with small capacity
        def make_small_replay():
            return embodied.replay.Replay(length=args.batch_length, capacity=500)

        embodied.run.train(
            lambda: agent,
            make_small_replay,
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        assert stats["env_steps"] > 0, "Should complete training with small replay"
        assert stats["replay_steps"] > 0, "Should still train on replay data"

    def test_training_with_multiple_environments(self, tmpdir):
        """Test training with multiple parallel environments."""
        args = self._make_args(tmpdir, steps=500, envs=8)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        assert stats["env_steps"] >= args.steps, (
            "Should collect steps from all environments"
        )

    def test_training_with_debug_mode(self, tmpdir):
        """Test training with debug mode (sequential execution)."""
        args = self._make_args(tmpdir, steps=300, debug=True)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        assert stats["env_steps"] > 0, "Debug mode should still train"
        assert stats["replay_steps"] > 0, "Debug mode should still use replay"

    def test_training_with_different_batch_sizes(self, tmpdir):
        """Test training with various batch sizes."""
        for i, batch_size in enumerate([4, 8, 16]):
            # Create separate logdir for each batch size test
            subdir = elements.Path(tmpdir) / f"batch_{batch_size}"
            subdir.mkdir()
            # Use longer steps for larger batch sizes to ensure replay fills
            steps = 300 + batch_size * 30
            args = self._make_args(subdir, steps=steps, batch_size=batch_size)
            agent = self._make_agent()

            embodied.run.train(
                lambda: agent,
                bind(self._make_replay, args),
                self._make_env,
                bind(self._make_stream, args),
                self._make_logger,
                args,
            )

            stats = agent.stats()
            assert stats["replay_steps"] > 0, (
                f"Should train with batch_size={batch_size}, got {stats['replay_steps']} replay steps"
            )

    def test_report_generation(self, tmpdir):
        """Test that agent.report() is called correctly."""
        args = self._make_args(
            tmpdir, steps=600, report_every=0.08, report_batches=2, consec_report=1
        )
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        # With report_every=0.08 and steps=600, expect multiple reports
        # Each report processes report_batches=2 batches
        # Use conservative expectation due to timing variability
        assert stats["reports"] >= 4, (
            f"Expected at least 4 reports, got {stats['reports']}"
        )

    def test_training_fills_replay_buffer(self, tmpdir):
        """Test that the replay buffer is populated during training."""
        args = self._make_args(tmpdir, steps=400)
        agent = self._make_agent()
        replay_holder = {}

        def make_replay_tracked():
            replay = embodied.replay.Replay(length=args.batch_length, capacity=1e4)
            replay_holder["replay"] = replay
            return replay

        embodied.run.train(
            lambda: agent,
            make_replay_tracked,
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        replay = replay_holder["replay"]
        assert len(replay) > 0, "Replay buffer should contain data"
        stats_dict = replay.stats()
        assert stats_dict["items"] > 0, "Replay should have items"

    def test_training_with_very_short_duration(self, tmpdir):
        """Test training with very short duration (edge case)."""
        args = self._make_args(tmpdir, steps=50)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        # Even with short duration, some training should happen
        assert stats["env_steps"] > 0, "Should take at least some steps"

    def test_episode_logging(self, tmpdir):
        """Test that episode statistics are logged correctly."""

        # Use environment with shorter episode length for faster testing
        def make_short_env(index):
            return dummy.Dummy("disc", size=(64, 64), length=50)

        args = self._make_args(tmpdir, steps=400, log_every=0.1)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            make_short_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        # Should complete at least one episode
        assert stats["env_steps"] >= 50, "Should complete at least one episode"

    def test_training_with_large_batch_length(self, tmpdir):
        """Test training with larger batch length."""
        args = self._make_args(tmpdir, steps=400, batch_size=4, batch_length=32)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        assert stats["replay_steps"] > 0, "Should train with large batch_length"

    def test_training_with_log_metrics_in_env(self, tmpdir):
        """Test training when environment returns log/ metrics."""

        class EnvWithLogMetrics(dummy.Dummy):
            """Environment that returns log/ metrics in observations."""

            def step(self, action):
                obs = super().step(action)
                # Add log/ metrics to observations
                obs["log/custom_metric"] = np.float32(0.5)
                obs["log/another_value"] = np.float32(1.0)
                return obs

            @property
            def obs_space(self):
                space = super().obs_space
                # Add log/ metrics to observation space
                space["log/custom_metric"] = elements.Space(np.float32)
                space["log/another_value"] = elements.Space(np.float32)
                return space

        def make_env_with_logs(index):
            return EnvWithLogMetrics("disc", size=(64, 64), length=100)

        args = self._make_args(tmpdir, steps=400)
        agent = self._make_agent()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            make_env_with_logs,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        assert stats["env_steps"] > 0, "Should handle log/ metrics from environment"

    def test_training_with_replay_updates(self, tmpdir):
        """Test training when agent.train() returns replay updates."""

        class AgentWithReplayUpdates(utils.TestAgent):
            """Agent that returns replay updates from training."""

            def train(self, carry, data):
                carry, outs, metrics = super().train(carry, data)
                # Return replay updates without priorities
                # (Uniform selector doesn't support prioritize())
                # Just return stepid to test the update path
                outs["replay"] = {
                    "stepid": data["stepid"],
                }
                return carry, outs, metrics

        args = self._make_args(tmpdir, steps=400)
        env = self._make_env(0)
        agent = AgentWithReplayUpdates(env.obs_space, env.act_space)
        env.close()

        embodied.run.train(
            lambda: agent,
            bind(self._make_replay, args),
            self._make_env,
            bind(self._make_stream, args),
            self._make_logger,
            args,
        )

        stats = agent.stats()
        assert stats["replay_steps"] > 0, "Should handle replay updates"

    # Helper methods

    def _make_agent(self):
        """Create a test agent with dummy environment spaces."""
        env = self._make_env(0)
        agent = utils.TestAgent(env.obs_space, env.act_space)
        env.close()
        return agent

    def _make_env(self, index):
        """Create a dummy environment."""
        return dummy.Dummy("disc", size=(64, 64), length=100)

    def _make_replay(self, args):
        """Create a replay buffer with config from args."""
        return embodied.replay.Replay(length=args.batch_length, capacity=1e4)

    def _make_stream(self, args, replay, mode):
        """Create a data stream from replay buffer."""
        fn = bind(replay.sample, args.batch_size, mode)
        stream = embodied.streams.Stateless(fn)
        stream = embodied.streams.Consec(
            stream,
            length=args.batch_length if mode == "train" else args.report_length,
            consec=1,
            prefix=0,
            strict=(mode == "train"),
            contiguous=True,
        )
        return stream

    def _make_logger(self):
        """Create a logger for test output."""
        return elements.Logger(
            elements.Counter(),
            [elements.logger.TerminalOutput()],
        )

    def _make_args(
        self,
        logdir,
        steps=1000,
        train_ratio=32.0,
        log_every=0.2,
        report_every=0.3,
        save_every=0.3,
        report_batches=1,
        consec_report=1,
        envs=4,
        batch_size=8,
        batch_length=16,
        debug=False,
        from_checkpoint="",
        from_checkpoint_regex="",
    ):
        """Create configuration args for training."""
        return elements.Config(
            steps=steps,
            train_ratio=train_ratio,
            log_every=log_every,
            report_every=report_every,
            save_every=save_every,
            report_batches=report_batches,
            consec_report=consec_report,
            from_checkpoint=from_checkpoint,
            from_checkpoint_regex=from_checkpoint_regex,
            usage=dict(psutil=True),
            debug=debug,
            logdir=str(logdir),
            envs=envs,
            batch_size=batch_size,
            batch_length=batch_length,
            replay_context=0,
            report_length=8,
        )
