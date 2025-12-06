"""Evaluation-only mode for trained agents.

This module provides a script for evaluating pre-trained agents without any
training. It loads a checkpoint and runs the agent's policy in evaluation mode
across multiple environment workers, collecting episode statistics and metrics.

This is useful for:
    - Benchmarking trained agents on test environments
    - Generating rollouts for analysis or visualization
    - Measuring final performance after training completes
    - Validating checkpoint integrity and reproducibility
"""

from collections import defaultdict
from functools import partial as bind

import elements
import numpy as np

import embodied


def eval_only(make_agent, make_env, make_logger, args):
    """Run evaluation-only mode with a pre-trained agent checkpoint.

    Loads a trained agent from a checkpoint and evaluates it across multiple
    environment workers without any training. The agent's policy runs in evaluation
    mode (deterministic or low-exploration behavior) and collects episode statistics
    including scores, lengths, and custom metrics.

    The evaluation loop:
        1. Loads agent from checkpoint (requires args.from_checkpoint)
        2. Initializes parallel environment workers with the agent's init policy
        3. Runs the agent's evaluation policy for the specified number of steps
        4. Collects per-episode metrics (score, length, rewards, custom logs)
        5. Periodically logs aggregated statistics and system usage
        6. Continues until reaching args.steps total environment steps

    Episode metrics are aggregated across all workers and logged at intervals
    defined by args.log_every. Per-worker episode statistics are tracked
    independently and merged at logging time.

    Args:
        make_agent: Callable that returns an initialized agent instance with
            policy() and init_policy() methods. The agent must be compatible
            with checkpoint loading.
        make_env: Callable that takes a worker index and returns an environment
            instance implementing the embodied.Env interface (step, reset, etc.).
        make_logger: Callable that returns a logger instance with add(), write(),
            and close() methods for metric tracking.
        args: Configuration object containing:
            - from_checkpoint (str): Path to checkpoint file (required)
            - logdir (str): Directory for evaluation logs
            - steps (int): Total environment steps to evaluate
            - envs (int): Number of parallel environment workers
            - log_every (int): Logging interval in steps
            - debug (bool): If True, disables parallel execution
            - usage (dict): System resource monitoring configuration

    Note:
        This function requires args.from_checkpoint to be set, otherwise it will
        fail the assertion at startup. The checkpoint must contain an 'agent' key
        with the trained agent state.

    Note:
        Images (uint8 arrays with 3 dimensions) are only logged for worker 0 to
        reduce memory and disk usage. All other metrics are logged for all workers.
    """
    assert args.from_checkpoint

    agent = make_agent()
    logger = make_logger()

    logdir = elements.Path(args.logdir)
    logdir.mkdir()
    print("Logdir", logdir)
    step = logger.step
    usage = elements.Usage(**args.usage)
    agg = elements.Agg()
    epstats = elements.Agg()
    episodes: dict[int, elements.Agg] = defaultdict(elements.Agg)
    should_log = elements.when.Clock(args.log_every)
    policy_fps = elements.FPS()

    @elements.timer.section("logfn")
    def logfn(tran, worker):
        episode = episodes[worker]
        tran["is_first"] and episode.reset()
        episode.add("score", tran["reward"], agg="sum")
        episode.add("length", 1, agg="sum")
        episode.add("rewards", tran["reward"], agg="stack")
        for key, value in tran.items():
            isimage = (value.dtype == np.uint8) and (value.ndim == 3)
            if isimage and worker == 0:
                episode.add(f"policy_{key}", value, agg="stack")
            elif key.startswith("log/"):
                assert value.ndim == 0, (key, value.shape, value.dtype)
                episode.add(key + "/avg", value, agg="avg")
                episode.add(key + "/max", value, agg="max")
                episode.add(key + "/sum", value, agg="sum")
        if tran["is_last"]:
            result = episode.result()
            logger.add(
                {
                    "score": result.pop("score"),
                    "length": result.pop("length"),
                },
                prefix="episode",
            )
            rew = result.pop("rewards")
            if len(rew) > 1:
                result["reward_rate"] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    fns = [bind(make_env, i) for i in range(args.envs)]
    driver = embodied.Driver(fns, parallel=(not args.debug))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(logfn)

    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(args.from_checkpoint, keys=["agent"])

    print("Start evaluation")
    policy = lambda *args: agent.policy(*args, mode="eval")
    driver.reset(agent.init_policy)
    while step < args.steps:
        driver(policy, steps=10)
        if should_log(step):
            logger.add(agg.result())
            logger.add(epstats.result(), prefix="epstats")
            logger.add(usage.stats(), prefix="usage")
            logger.add({"fps/policy": policy_fps.result()})
            logger.add({"timer": elements.timer.stats()["summary"]})
            logger.write()

    logger.close()
