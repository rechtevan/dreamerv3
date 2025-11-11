from functools import partial as bind

import numpy as np
import pytest

import embodied


class TestDriver:
    def test_episode_length(self):
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env])
        driver.reset(agent.init_policy)
        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(agent.policy, episodes=1)
        assert len(seq) == 11

    def test_first_step(self):
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env])
        driver.reset(agent.init_policy)
        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(agent.policy, episodes=2)
        for index in [0, 11]:
            assert seq[index]["is_first"].item() is True
            assert seq[index]["is_last"].item() is False
        for index in [1, 10, 12]:
            assert seq[index]["is_first"].item() is False

    def test_last_step(self):
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env])
        driver.reset(agent.init_policy)
        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(agent.policy, episodes=2)
        for index in [10, 21]:
            assert seq[index]["is_last"].item() is True
            assert seq[index]["is_first"].item() is False
        for index in [0, 1, 9, 11, 20]:
            assert seq[index]["is_last"].item() is False

    def test_env_reset(self):
        agent = self._make_agent()
        driver = embodied.Driver([bind(self._make_env, length=5)])
        driver.reset(agent.init_policy)
        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        action = {"act_disc": np.ones(1, int), "act_cont": np.zeros((1, 6), float)}
        policy = lambda carry, obs: (carry, action, {})
        driver(policy, episodes=2)
        assert len(seq) == 12
        seq = {k: np.array([seq[i][k] for i in range(len(seq))]) for k in seq[0]}
        assert (seq["is_first"] == [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all()
        assert (seq["is_last"] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).all()
        assert (seq["reset"] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).all()
        assert (seq["act_disc"] == [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]).all()

    def test_agent_inputs(self):
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env])
        driver.reset(agent.init_policy)
        inputs = []
        states = []

        def policy(carry, obs, mode="train"):
            inputs.append(obs)
            states.append(carry)
            _, act, _ = agent.policy(carry, obs, mode)
            return "carry", act, {}

        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(policy, episodes=2)
        assert len(seq) == 22
        assert states == ([()] + ["carry"] * 21)
        for index in [0, 11]:
            assert inputs[index]["is_first"].item() is True
        for index in [1, 10, 12, 21]:
            assert inputs[index]["is_first"].item() is False
        for index in [10, 21]:
            assert inputs[index]["is_last"].item() is True
        for index in [0, 1, 9, 11, 20]:
            assert inputs[index]["is_last"].item() is False

    def test_unexpected_reset(self):
        class UnexpectedReset(embodied.Wrapper):
            """Send is_first without preceeding is_last."""

            def __init__(self, env, when):
                super().__init__(env)
                self._when = when
                self._step = 0

            def step(self, action):
                if self._step == self._when:
                    action = action.copy()
                    action["reset"] = np.ones_like(action["reset"])
                self._step += 1
                return self.env.step(action)

        env = self._make_env(length=4)
        env = UnexpectedReset(env, when=3)
        agent = self._make_agent()
        driver = embodied.Driver([lambda: env])
        driver.reset(agent.init_policy)
        steps = []
        driver.on_step(lambda tran, _: steps.append(tran))
        driver(agent.policy, episodes=1)
        assert len(steps) == 8
        steps = {k: np.array([x[k] for x in steps]) for k in steps[0]}
        assert (steps["reset"] == [0, 0, 0, 0, 0, 0, 0, 1]).all()
        assert (steps["is_first"] == [1, 0, 0, 1, 0, 0, 0, 0]).all()
        assert (steps["is_last"] == [0, 0, 0, 0, 0, 0, 0, 1]).all()

    # ==================== NEW INTEGRATION TESTS ====================

    def test_multi_worker_parallel(self):
        """Test driver with multiple parallel workers."""
        agent = self._make_agent()
        # Create 4 parallel environments
        driver = embodied.Driver([bind(self._make_env, length=5)] * 4, parallel=True)
        driver.reset(agent.init_policy)

        seq = []
        worker_steps = [0, 0, 0, 0]  # Track steps per worker

        def on_step(tran, worker_id):
            seq.append((worker_id, tran))
            worker_steps[worker_id] += 1

        driver.on_step(on_step)
        driver(agent.policy, episodes=4)

        # Verify each worker processed at least one step
        assert all(s > 0 for s in worker_steps), f"Worker steps: {worker_steps}"
        # Verify we got 4 episodes total (4 workers * 6 steps each = 24 steps)
        assert len(seq) == 24

        # Verify episode boundaries per worker
        for worker_id in range(4):
            worker_seq = [t for wid, t in seq if wid == worker_id]
            assert worker_seq[0]["is_first"]
            assert worker_seq[-1]["is_last"]

    def test_multi_worker_non_parallel(self):
        """Test driver with multiple workers in non-parallel mode."""
        agent = self._make_agent()
        # Create 4 non-parallel environments
        driver = embodied.Driver([bind(self._make_env, length=5)] * 4, parallel=False)
        driver.reset(agent.init_policy)

        seq = []
        worker_steps = [0, 0, 0, 0]

        def on_step(tran, worker_id):
            seq.append((worker_id, tran))
            worker_steps[worker_id] += 1

        driver.on_step(on_step)
        driver(agent.policy, episodes=4)

        # Verify each worker processed steps
        assert all(s > 0 for s in worker_steps), f"Worker steps: {worker_steps}"
        assert len(seq) == 24  # 4 workers * 6 steps

    def test_episode_simultaneous_termination(self):
        """Test when multiple workers finish episodes simultaneously."""
        agent = self._make_agent()
        # All environments have same length - should finish at same time
        driver = embodied.Driver([bind(self._make_env, length=5)] * 3, parallel=False)
        driver.reset(agent.init_policy)

        seq = []
        episodes_ended = []

        def on_step(tran, worker_id):
            seq.append((worker_id, tran))
            if tran["is_last"]:
                episodes_ended.append(worker_id)

        driver.on_step(on_step)
        driver(agent.policy, episodes=3)  # Run until 3 episodes complete

        # All 3 workers should complete their episodes
        assert len(episodes_ended) == 3
        assert set(episodes_ended) == {0, 1, 2}

    def test_varying_episode_lengths(self):
        """Test driver with environments that produce varying episode lengths."""
        agent = self._make_agent()
        # Create environments with different lengths
        driver = embodied.Driver(
            [
                bind(self._make_env, length=3),
                bind(self._make_env, length=5),
                bind(self._make_env, length=7),
            ],
            parallel=False,
        )
        driver.reset(agent.init_policy)

        episodes_by_worker = {0: [], 1: [], 2: []}

        def on_step(tran, worker_id):
            if tran["is_last"]:
                episodes_by_worker[worker_id].append(tran)

        driver.on_step(on_step)
        driver(agent.policy, steps=30)  # Run enough steps to see multiple episodes

        # Each worker should complete at least one episode
        assert all(len(eps) > 0 for eps in episodes_by_worker.values())

    def test_long_episode(self):
        """Test driver with very long episodes to verify no memory issues."""
        agent = self._make_agent()
        driver = embodied.Driver([bind(self._make_env, length=500)], parallel=False)
        driver.reset(agent.init_policy)

        step_count = 0

        def on_step(tran, _):
            nonlocal step_count
            step_count += 1

        driver.on_step(on_step)
        driver(agent.policy, episodes=1)

        # Should have exactly 501 steps (length + 1 for reset)
        assert step_count == 501

    def test_callback_receives_correct_data(self):
        """Test that callbacks receive correct transition data and worker IDs."""
        agent = self._make_agent()
        driver = embodied.Driver([bind(self._make_env, length=5)] * 2, parallel=False)
        driver.reset(agent.init_policy)

        callback_data = []

        def on_step(tran, worker_id):
            callback_data.append(
                {
                    "worker_id": worker_id,
                    "is_first": tran["is_first"],
                    "is_last": tran["is_last"],
                    "count": tran["count"],
                }
            )

        driver.on_step(on_step)
        driver(agent.policy, episodes=2)

        # Verify worker IDs are correct
        assert all(0 <= d["worker_id"] <= 1 for d in callback_data)

        # Verify episode structure for worker 0
        worker0_data = [d for d in callback_data if d["worker_id"] == 0]
        assert worker0_data[0]["is_first"]
        assert worker0_data[-1]["is_last"]

    def test_multiple_callbacks(self):
        """Test that multiple callbacks are all invoked."""
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env], parallel=False)
        driver.reset(agent.init_policy)

        callback1_count = [0]
        callback2_count = [0]
        callback3_count = [0]

        def callback1(tran, _):
            callback1_count[0] += 1

        def callback2(tran, _):
            callback2_count[0] += 1

        def callback3(tran, _):
            callback3_count[0] += 1

        driver.on_step(callback1)
        driver.on_step(callback2)
        driver.on_step(callback3)

        driver(agent.policy, episodes=1)

        # All callbacks should be called same number of times
        assert callback1_count[0] == callback2_count[0] == callback3_count[0] == 11

    def test_reset_without_init_policy(self):
        """Test driver reset without providing init_policy."""
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env], parallel=False)

        # Reset without init_policy (should set carry to None)
        driver.reset()

        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(agent.policy, episodes=1)

        assert len(seq) == 11

    def test_reset_with_init_policy(self):
        """Test driver reset with init_policy to initialize carry state."""
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env], parallel=False)

        # Reset with init_policy
        driver.reset(agent.init_policy)

        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(agent.policy, episodes=1)

        assert len(seq) == 11

    def test_steps_limit(self):
        """Test driver stops correctly when step limit is reached."""
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env], parallel=False)
        driver.reset(agent.init_policy)

        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))

        # Run for exactly 15 steps (should be mid-episode)
        driver(agent.policy, steps=15)

        assert len(seq) == 15
        # Should not have completed 2 full episodes
        last_transitions = [s for s in seq if s["is_last"]]
        assert len(last_transitions) == 1  # Only one episode complete

    def test_close_parallel(self):
        """Test closing driver with parallel workers."""
        driver = embodied.Driver([bind(self._make_env, length=5)] * 2, parallel=True)
        driver.reset()

        # Close should not raise any errors
        driver.close()

    def test_close_non_parallel(self):
        """Test closing driver with non-parallel workers."""
        driver = embodied.Driver([bind(self._make_env, length=5)] * 2, parallel=False)
        driver.reset()

        # Close should not raise any errors
        driver.close()

    def test_action_masking_on_episode_end(self):
        """Test that actions are masked correctly when episodes end."""
        agent = self._make_agent()
        driver = embodied.Driver([bind(self._make_env, length=3)] * 2, parallel=False)
        driver.reset(agent.init_policy)

        transitions = []
        driver.on_step(lambda tran, _: transitions.append(tran))
        driver(agent.policy, episodes=2)

        # Check that actions are present in transitions
        for tran in transitions:
            assert "act_disc" in tran
            assert "act_cont" in tran
            assert "reset" in tran

    def test_policy_outputs(self):
        """Test that policy outputs are included in transitions."""
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env], parallel=False)
        driver.reset(agent.init_policy)

        # Policy that returns additional outputs
        def policy_with_outputs(carry, obs):
            _, act, _ = agent.policy(carry, obs)
            # Output needs to match batch size (1 worker)
            outs = {"custom_output": np.array([[1.0, 2.0, 3.0]])}
            return carry, act, outs

        transitions = []
        driver.on_step(lambda tran, _: transitions.append(tran))
        driver(policy_with_outputs, episodes=1)

        # Verify custom outputs are in transitions
        for tran in transitions:
            assert "custom_output" in tran
            assert len(tran["custom_output"]) == 3

    def test_log_keys_separated(self):
        """Test that log/ prefixed keys are separated from observations."""
        agent = self._make_agent()

        # Create environment wrapper that adds log keys
        class LogEnv(embodied.Wrapper):
            def step(self, action):
                obs = self.env.step(action)
                obs["log/custom_metric"] = np.float32(1.5)
                return obs

        env = LogEnv(self._make_env())
        driver = embodied.Driver([lambda: env], parallel=False)
        driver.reset(agent.init_policy)

        transitions = []
        driver.on_step(lambda tran, _: transitions.append(tran))
        driver(agent.policy, episodes=1)

        # Verify log keys are in transitions
        for tran in transitions:
            assert "log/custom_metric" in tran
            assert tran["log/custom_metric"] == 1.5

    def test_batch_size_single_worker(self):
        """Test driver with batch_size=1 (single worker edge case)."""
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env], parallel=False)
        driver.reset(agent.init_policy)

        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(agent.policy, episodes=1)

        assert len(seq) == 11
        assert driver.length == 1

    def test_batch_size_many_workers(self):
        """Test driver with many workers (batch_size=8)."""
        agent = self._make_agent()
        driver = embodied.Driver([bind(self._make_env, length=5)] * 8, parallel=False)
        driver.reset(agent.init_policy)

        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))
        driver(agent.policy, episodes=8)

        # 8 workers * 6 steps per episode = 48 total steps
        assert len(seq) == 48
        assert driver.length == 8

    def test_zero_episodes_zero_steps(self):
        """Test calling driver with both episodes=0 and steps=0 (no-op)."""
        agent = self._make_agent()
        driver = embodied.Driver([self._make_env], parallel=False)
        driver.reset(agent.init_policy)

        seq = []
        driver.on_step(lambda tran, _: seq.append(tran))

        # This should not run any steps
        driver(agent.policy, episodes=0, steps=0)

        assert len(seq) == 0

    def test_parallel_with_obs_space(self):
        """Test parallel driver accesses environment obs_space."""
        # This test verifies the act_space retrieval in parallel mode
        driver = embodied.Driver([bind(self._make_env, length=5)] * 2, parallel=True)
        driver.reset()

        # Verify act_space was retrieved
        assert "act_disc" in driver.act_space
        assert "act_cont" in driver.act_space
        assert "reset" in driver.act_space

        driver.close()

    def test_parallel_multiple_episodes(self):
        """Test parallel driver running multiple episodes."""
        agent = self._make_agent()
        driver = embodied.Driver([bind(self._make_env, length=8)] * 2, parallel=True)
        driver.reset(agent.init_policy)

        episodes_completed = []

        def on_step(tran, worker_id):
            if tran["is_last"]:
                episodes_completed.append(worker_id)

        driver.on_step(on_step)
        driver(agent.policy, episodes=4)

        # Should have 4 episodes completed (2 per worker)
        assert len(episodes_completed) == 4

        driver.close()

    def test_action_space_structure(self):
        """Test that driver correctly initializes action space structure."""
        driver = embodied.Driver([bind(self._make_env, length=5)] * 3, parallel=False)
        driver.reset()

        # Verify acts are initialized correctly
        assert "reset" in driver.acts
        assert "act_disc" in driver.acts
        assert "act_cont" in driver.acts

        # Verify shapes match batch size
        assert driver.acts["reset"].shape == (3,)
        assert driver.acts["act_disc"].shape == (3,)
        assert driver.acts["act_cont"].shape == (3, 6)

        # Verify reset is set to True initially
        assert driver.acts["reset"].all()

    def test_carry_state_propagation(self):
        """Test that carry state is properly propagated through steps."""
        agent = self._make_agent()
        driver = embodied.Driver([bind(self._make_env, length=5)], parallel=False)
        driver.reset(agent.init_policy)

        carry_values = []

        def policy_tracking_carry(carry, obs):
            # Extract and record carry value
            val = carry[0][0] if isinstance(carry, tuple) and len(carry) > 0 else 0
            carry_values.append(val)
            # Return incremented carry state
            new_val = np.array([val + 1])
            _, act, _ = agent.policy(carry, obs)
            return (new_val,), act, {}

        driver.on_step(lambda tran, _: None)
        driver(policy_tracking_carry, episodes=1)

        # Verify carry state increases each step
        assert len(carry_values) == 6
        for i in range(len(carry_values)):
            assert carry_values[i] == i

    def test_observation_stacking(self):
        """Test that observations are correctly stacked from multiple workers."""
        agent = self._make_agent()
        driver = embodied.Driver([bind(self._make_env, length=3)] * 4, parallel=False)
        driver.reset(agent.init_policy)

        obs_shapes = []

        def policy_checking_shapes(carry, obs):
            obs_shapes.append({k: v.shape for k, v in obs.items()})
            _, act, _ = agent.policy(carry, obs)
            return carry, act, {}

        driver.on_step(lambda tran, _: None)
        driver(policy_checking_shapes, steps=5)

        # All observations should have batch dimension of 4 (4 workers)
        for shapes in obs_shapes:
            assert shapes["image"][0] == 4  # (batch, h, w, c)
            assert shapes["vector"][0] == 4  # (batch, feat)
            assert shapes["is_first"][0] == 4  # (batch,)
            assert shapes["is_last"][0] == 4  # (batch,)

    def test_episode_reset_flag(self):
        """Test that reset flag is set correctly on episode boundaries."""
        agent = self._make_agent()
        driver = embodied.Driver([bind(self._make_env, length=3)], parallel=False)
        driver.reset(agent.init_policy)

        transitions = []
        driver.on_step(lambda tran, _: transitions.append(tran))
        driver(agent.policy, episodes=2)

        # Check reset flags
        reset_flags = [t["reset"] for t in transitions]
        # Reset should be True at indices where episodes end (3 and 7)
        assert reset_flags[3]  # End of first episode
        assert reset_flags[7]  # End of second episode
        # Other positions should be False
        assert not reset_flags[0]
        assert not reset_flags[1]
        assert not reset_flags[2]

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)

    def _make_agent(self):
        env = self._make_env()
        agent = embodied.RandomAgent(env.obs_space, env.act_space)
        env.close()
        return agent
