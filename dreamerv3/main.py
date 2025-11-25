import importlib
import os
import pathlib
import sys
from functools import partial as bind


folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import numpy as np
import portal
import ruamel.yaml as yaml

import embodied


"""DreamerV3 main entry point and factory functions.

This module provides the primary entry point for DreamerV3 training and evaluation,
along with factory functions for creating agents, environments, loggers, replay buffers,
and data streams. Handles configuration loading, environment setup, and orchestration
of different training scripts (train, train_eval, parallel, eval_only).

Key components:
    - main(): CLI entry point with config loading and script orchestration
    - make_agent(): Agent creation from configuration
    - make_env(): Environment creation with suite-specific constructors
    - wrap_env(): Standard environment wrapper application
    - make_logger(): Multi-output logger setup (terminal, jsonl, tensorboard, wandb)
    - make_replay(): Replay buffer with configurable selectors
    - make_stream(): Data streaming with consecutive sampling
"""


def main(argv=None):
    """DreamerV3 training and evaluation entry point.

    Loads configuration from configs.yaml, parses command-line arguments, sets up
    logging and distributed training infrastructure, and orchestrates the selected
    training/evaluation script (train, train_eval, eval_only, parallel variants).

    Configuration is loaded in layers:
        1. Base defaults from configs.yaml
        2. Named configs specified via --configs flag (e.g., atari, crafter)
        3. Command-line overrides (e.g., --batch_size 16)

    Supports distributed training via JOB_COMPLETION_INDEX environment variable
    for replica indexing. Creates logdir, saves final config, and delegates to
    embodied.run scripts based on config.script value.

    Args:
        argv: Command-line arguments list. If None, uses sys.argv.
            Common flags:
                --configs: Config names to merge (e.g., "crafter size50m")
                --logdir: Output directory for logs and checkpoints
                --task: Environment task (e.g., "atari_pong")
                --script: Training script to run (train, train_eval, etc.)

    Raises:
        NotImplementedError: If config.script specifies unknown training script.
    """
    from .agent import Agent

    [elements.print(line) for line in Agent.banner]

    configs = elements.Path(folder / "configs.yaml").read()
    configs = yaml.YAML(typ="safe").load(configs)
    parsed, other = elements.Flags(configs=["defaults"]).parse_known(argv)
    config = elements.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)
    config = config.update(
        logdir=(config.logdir.format(timestamp=elements.timestamp()))
    )

    if "JOB_COMPLETION_INDEX" in os.environ:
        config = config.update(replica=int(os.environ["JOB_COMPLETION_INDEX"]))
    print("Replica:", config.replica, "/", config.replicas)

    logdir = elements.Path(config.logdir)
    print("Logdir:", logdir)
    print("Run script:", config.script)
    if not config.script.endswith(("_env", "_replay")):
        logdir.mkdir()
        config.save(logdir / "config.yaml")

    def init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        errfile=config.errfile and logdir / "error",
        clientkw=dict(logging_color="cyan"),
        serverkw=dict(logging_color="cyan"),
        initfns=[init],
        ipv6=config.ipv6,
    )

    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    if config.script == "train":
        embodied.run.train(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args,
        )

    elif config.script == "train_eval":
        embodied.run.train_eval(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_replay, config, "eval_replay", "eval"),
            bind(make_env, config),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args,
        )

    elif config.script == "eval_only":
        embodied.run.eval_only(
            bind(make_agent, config),
            bind(make_env, config),
            bind(make_logger, config),
            args,
        )

    elif config.script == "parallel":
        embodied.run.parallel.combined(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_replay, config, "replay_eval", "eval"),
            bind(make_env, config),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args,
        )

    elif config.script == "parallel_env":
        is_eval = config.replica >= args.envs
        embodied.run.parallel.parallel_env(
            bind(make_env, config), config.replica, args, is_eval
        )

    elif config.script == "parallel_envs":
        is_eval = config.replica >= args.envs
        embodied.run.parallel.parallel_envs(
            bind(make_env, config), bind(make_env, config), args
        )

    elif config.script == "parallel_replay":
        embodied.run.parallel.parallel_replay(
            bind(make_replay, config, "replay"),
            bind(make_replay, config, "replay_eval", "eval"),
            bind(make_stream, config),
            args,
        )

    else:
        raise NotImplementedError(config.script)


def make_agent(config):
    """Create DreamerV3 agent from configuration.

    Factory function that instantiates an environment to extract observation and
    action spaces, then creates either a DreamerV3 Agent or RandomAgent based on
    configuration. Filters out log observations and reset actions before passing
    spaces to the agent.

    Args:
        config: Configuration object containing:
            - task: Environment task name (e.g., "atari_pong")
            - random_agent: If True, creates RandomAgent instead of DreamerV3
            - agent: Agent-specific hyperparameters (RSSM, heads, optimizers)
            - logdir: Directory for checkpoints and logs
            - seed: Random seed for reproducibility
            - jax: JAX configuration (platform, precision, etc.)
            - batch_size: Training batch size
            - batch_length: Sequence length for training
            - replay_context: Context length for recurrent state
            - report_length: Sequence length for reporting
            - replica: Current replica index (for distributed training)
            - replicas: Total number of replicas

    Returns:
        Agent: Configured DreamerV3 Agent or RandomAgent instance.
    """
    from .agent import Agent

    env = make_env(config, 0)
    notlog = lambda k: not k.startswith("log/")
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
    env.close()
    if config.random_agent:
        return embodied.RandomAgent(obs_space, act_space)
    cpdir = elements.Path(config.logdir)
    cpdir = cpdir.parent if config.replicas > 1 else cpdir
    return Agent(
        obs_space,
        act_space,
        elements.Config(
            **config.agent,
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )


def make_logger(config):
    """Create logger with multiple output backends.

    Configures logging infrastructure with terminal output plus optional backends
    (JSONL, TensorBoard, WandB, Scope, Expa). Automatically applies action repeat
    multiplier to logged steps for accurate environment step counts.

    Args:
        config: Configuration object containing:
            - logdir: Directory for log files and outputs
            - logger.filter: Regex filter for terminal output metrics
            - logger.outputs: List of output backends ("jsonl", "tensorboard",
              "wandb", "scope", "expa")
            - logger.fps: Logging frequency for TensorBoard (FPS)
            - logger.user: Username for Expa logging
            - env: Environment configurations (for action repeat multiplier)
            - task: Task name for extracting suite prefix
            - flat: Flattened config dict for Expa

    Returns:
        elements.Logger: Configured logger with specified outputs.

    Raises:
        NotImplementedError: If unknown output backend specified in config.logger.outputs.
    """
    step = elements.Counter()
    logdir = config.logdir
    multiplier = config.env.get(config.task.split("_")[0], {}).get("repeat", 1)
    outputs = []
    outputs.append(elements.logger.TerminalOutput(config.logger.filter, "Agent"))
    for output in config.logger.outputs:
        if output == "jsonl":
            outputs.append(elements.logger.JSONLOutput(logdir, "metrics.jsonl"))
            outputs.append(
                elements.logger.JSONLOutput(logdir, "scores.jsonl", "episode/score")
            )
        elif output == "tensorboard":
            outputs.append(elements.logger.TensorBoardOutput(logdir, config.logger.fps))
        elif output == "expa":
            exp = logdir.split("/")[-4]
            run = "/".join(logdir.split("/")[-3:])
            proj = "embodied" if logdir.startswith(("/cns/", "gs://")) else "debug"
            outputs.append(
                elements.logger.ExpaOutput(
                    exp, run, proj, config.logger.user, config.flat
                )
            )
        elif output == "wandb":
            name = "/".join(logdir.split("/")[-4:])
            outputs.append(elements.logger.WandBOutput(name))
        elif output == "scope":
            outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
        else:
            raise NotImplementedError(output)
    logger = elements.Logger(step, outputs, multiplier)
    return logger


def make_replay(config, folder, mode="train"):
    """Create replay buffer with configurable sampling selectors.

    Instantiates experience replay buffer with optional prioritized/recency-weighted
    sampling. Automatically adjusts capacity, sequence length, and selectors based on
    train vs eval mode. Supports uniform, prioritized (TD-error), and recency-based
    sampling strategies.

    Args:
        config: Configuration object containing:
            - logdir: Base directory for replay storage
            - batch_size: Training batch size
            - batch_length: Training sequence length
            - report_length: Reporting/eval sequence length
            - consec_train: Number of consecutive training sequences
            - consec_report: Number of consecutive reporting sequences
            - replay_context: Context length for recurrent state burn-in
            - replay.size: Maximum replay buffer capacity (number of steps)
            - replay.online: Whether to use online (recent-only) mode
            - replay.chunksize: Chunk size for storage optimization
            - replay.fracs: Mixture weights for selectors (uniform, priority, recency)
            - replay.recexp: Recency weighting exponent
            - replay.prio: Prioritized replay hyperparameters
            - jax.compute_dtype: Computation dtype (affects loss validity checks)
            - replica: Current replica index
            - replicas: Total number of replicas
        folder: Subdirectory name within logdir for this replay buffer
            (e.g., "replay", "eval_replay")
        mode: "train" or "eval" - controls capacity and sequence length

    Returns:
        embodied.replay.Replay: Configured replay buffer instance.

    Raises:
        AssertionError: If batch_size * sequence_length exceeds capacity, or if
            using prioritized replay with incompatible compute dtype (float16).
    """
    batlen = config.batch_length if mode == "train" else config.report_length
    consec = config.consec_train if mode == "train" else config.consec_report
    capacity = config.replay.size if mode == "train" else config.replay.size / 10
    length = consec * batlen + config.replay_context
    assert config.batch_size * length <= capacity

    directory = elements.Path(config.logdir) / folder
    if config.replicas > 1:
        directory /= f"{config.replica:05}"
    kwargs = dict(
        length=length,
        capacity=int(capacity),
        online=config.replay.online,
        chunksize=config.replay.chunksize,
        directory=directory,
    )

    if config.replay.fracs.uniform < 1 and mode == "train":
        assert config.jax.compute_dtype in ("bfloat16", "float32"), (
            "Gradient scaling for low-precision training can produce invalid loss "
            "outputs that are incompatible with prioritized replay."
        )
        recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
        selectors = embodied.replay.selectors
        kwargs["selector"] = selectors.Mixture(
            dict(
                uniform=selectors.Uniform(),
                priority=selectors.Prioritized(**config.replay.prio),
                recency=selectors.Recency(recency),
            ),
            config.replay.fracs,
        )

    return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
    """Create environment instance with suite-specific constructor.

    Factory function that creates environment based on suite name (extracted from
    task string). Supports multiple RL benchmarks (Atari, DMC, Crafter, Minecraft,
    etc.) with automatic constructor resolution via importlib. Applies standard
    wrappers after construction.

    Args:
        config: Configuration object containing:
            - task: Task string in format "suite_taskname" (e.g., "atari_pong")
            - env: Suite-specific environment configurations
            - seed: Base random seed (combined with index for determinism)
            - logdir: Base directory for environment-specific logs
        index: Environment index for seeding and logging (e.g., worker ID)
        **overrides: Additional kwargs to override suite-specific config

    Returns:
        embodied.Env: Wrapped environment instance ready for interaction.

    Raises:
        KeyError: If suite name not recognized in constructor mapping.
        ImportError: If suite-specific dependencies not installed.
    """
    suite, task = config.task.split("_", 1)
    if suite == "memmaze":
        import memory_maze

        from embodied.envs import from_gym
    ctor = {
        "dummy": "embodied.envs.dummy:Dummy",
        "gym": "embodied.envs.from_gym:FromGym",
        "dm": "embodied.envs.from_dmenv:FromDM",
        "crafter": "embodied.envs.crafter:Crafter",
        "dmc": "embodied.envs.dmc:DMC",
        "atari": "embodied.envs.atari:Atari",
        "atari100k": "embodied.envs.atari:Atari",
        "dmlab": "embodied.envs.dmlab:DMLab",
        "minecraft": "embodied.envs.minecraft:Minecraft",
        "loconav": "embodied.envs.loconav:LocoNav",
        "pinpad": "embodied.envs.pinpad:PinPad",
        "langroom": "embodied.envs.langroom:LangRoom",
        "procgen": "embodied.envs.procgen:ProcGen",
        "bsuite": "embodied.envs.bsuite:BSuite",
        "memmaze": lambda task, **kw: from_gym.FromGym(f"MemoryMaze-{task}-v0", **kw),
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(":")
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    if kwargs.pop("use_seed", False):
        kwargs["seed"] = hash((config.seed, index)) % (2**32 - 1)
    if kwargs.pop("use_logdir", False):
        kwargs["logdir"] = elements.Path(config.logdir) / f"env{index}"
    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def wrap_env(env, config):
    """Apply standard environment wrappers for DreamerV3 compatibility.

    Applies sequence of wrappers to ensure environment compatibility with DreamerV3:
        1. NormalizeAction: Scales continuous actions to [-1, 1] range
        2. UnifyDtypes: Ensures consistent dtypes across obs/actions
        3. CheckSpaces: Validates space definitions match actual outputs
        4. ClipAction: Clips continuous actions to valid range

    Args:
        env: Base environment instance (unwrapped)
        config: Configuration object (currently unused but kept for consistency)

    Returns:
        embodied.Env: Wrapped environment with normalization and validation.
    """
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_stream(config, replay, mode):
    """Create data stream with consecutive sequence sampling.

    Wraps replay buffer sampling with consecutive sequence extraction. Ensures
    sampled batches contain properly structured consecutive sequences for recurrent
    model training, with optional context prefix for state burn-in.

    Args:
        config: Configuration object containing:
            - batch_size: Number of sequences per batch
            - batch_length: Training sequence length (for mode="train")
            - report_length: Reporting sequence length (for mode="report")
            - consec_train: Number of consecutive training sequences
            - consec_report: Number of consecutive reporting sequences
            - replay_context: Context prefix length for recurrent state
        replay: Replay buffer instance to sample from
        mode: Sampling mode ("train" or "report") - controls sequence length and
            strictness of consecutive enforcement

    Returns:
        embodied.streams.Consec: Data stream yielding consecutive sequence batches
            with shape [B, T, ...] where B=batch_size, T=sequence_length.
    """
    fn = bind(replay.sample, config.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=config.batch_length if mode == "train" else config.report_length,
        consec=config.consec_train if mode == "train" else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == "train"),
        contiguous=True,
    )

    return stream


if __name__ == "__main__":
    main()
