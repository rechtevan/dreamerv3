"""Random agent for baseline evaluation and testing.

This module provides a simple random agent that samples actions uniformly
from the action space without any learning. It's useful for:
- Establishing baseline performance in environments
- Testing environment integration and data pipelines
- Debugging training infrastructure without complex agent logic

The RandomAgent implements the standard agent interface but performs no
actual learning or policy improvement.
"""

import numpy as np


class RandomAgent:
    """Agent that selects actions uniformly at random from the action space.

    This baseline agent provides a minimal implementation of the agent interface,
    sampling actions from the action space distribution without any learning.
    It serves as a sanity check for environments and training infrastructure.

    The agent maintains no internal state and ignores all observations and
    training data. All actions are sampled independently from the action space.

    Attributes:
        obs_space: Dictionary of observation space specifications.
        act_space: Dictionary of action space specifications.

    Example:
        # Create a random agent
        agent = RandomAgent(env.obs_space, env.act_space)

        # Use in training loop
        carry = agent.init_policy(batch_size=8)
        carry, actions, outs = agent.policy(carry, obs)
    """

    def __init__(self, obs_space, act_space):
        """Initialize the random agent.

        Args:
            obs_space: Dictionary of observation space specifications.
            act_space: Dictionary of action space specifications.
        """
        self.obs_space = obs_space
        self.act_space = act_space

    def init_policy(self, batch_size):
        """Initialize policy state (no state needed for random agent).

        Args:
            batch_size: Number of parallel environments.

        Returns:
            Empty tuple as this agent maintains no state.
        """
        return ()

    def init_train(self, batch_size):
        """Initialize training state (no state needed for random agent).

        Args:
            batch_size: Number of training examples in batch.

        Returns:
            Empty tuple as this agent maintains no state.
        """
        return ()

    def init_report(self, batch_size):
        """Initialize reporting state (no state needed for random agent).

        Args:
            batch_size: Number of examples for generating reports.

        Returns:
            Empty tuple as this agent maintains no state.
        """
        return ()

    def policy(self, carry, obs, mode="train"):
        """Sample random actions from the action space.

        Ignores observations and samples actions uniformly from each action
        space distribution. The 'reset' action is excluded as it's handled
        by the environment wrapper.

        Args:
            carry: Policy state (ignored, empty tuple).
            obs: Dictionary of observations with 'is_first' key for batch size.
            mode: Execution mode, either 'train' or 'eval' (ignored).

        Returns:
            Tuple of (carry, actions, outputs):
                - carry: Unchanged empty tuple
                - actions: Dict of randomly sampled actions for each space
                - outputs: Empty dict (no policy outputs)
        """
        batch_size = len(obs["is_first"])
        act = {
            k: np.stack([v.sample() for _ in range(batch_size)])
            for k, v in self.act_space.items()
            if k != "reset"
        }
        return carry, act, {}

    def train(self, carry, data):
        """Perform training step (no-op for random agent).

        Args:
            carry: Training state (ignored, empty tuple).
            data: Training batch data (ignored).

        Returns:
            Tuple of (carry, outputs, metrics):
                - carry: Unchanged empty tuple
                - outputs: Empty dict (no training outputs)
                - metrics: Empty dict (no training metrics)
        """
        return carry, {}, {}

    def report(self, carry, data):
        """Generate reports (no-op for random agent).

        Args:
            carry: Report state (ignored, empty tuple).
            data: Data for generating reports (ignored).

        Returns:
            Tuple of (carry, metrics):
                - carry: Unchanged empty tuple
                - metrics: Empty dict (no metrics to report)
        """
        return carry, {}

    def stream(self, st):
        """Return the data stream unchanged.

        Args:
            st: Input data stream.

        Returns:
            The same stream unchanged (no preprocessing needed).
        """
        return st

    def save(self):
        """Save agent state (no state to save).

        Returns:
            None as the agent has no learnable parameters or state.
        """
        return None

    def load(self, data=None):
        """Load agent state (no-op as agent has no state).

        Args:
            data: Checkpoint data (ignored).
        """
        pass
