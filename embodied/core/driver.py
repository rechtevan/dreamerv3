"""Environment driver for parallel and sequential episode collection.

This module provides the Driver class for collecting experience from multiple
environment instances. It supports both parallel (multiprocess) and sequential
execution modes, making it easy to scale data collection.

The Driver manages:
- Environment instantiation and lifecycle
- Batched action execution across environments
- Episode boundary detection and handling
- Callback execution for each transition

Example:
    >>> def make_env():
    ...     return MyEnvironment()
    >>> driver = Driver([make_env] * 4, parallel=True)
    >>> driver.on_step(lambda trn, i: replay.add(trn))
    >>> driver(policy, steps=10000)
    >>> driver.close()
"""

import time
import typing

import cloudpickle  # type: ignore
import elements
import numpy as np
import portal


class Driver:
    """Collects experience from multiple environments using a policy.

    The Driver executes a policy across multiple environment instances,
    collecting transitions and triggering callbacks. It supports both
    parallel (multiprocess) and sequential execution modes.

    In parallel mode, environments run in separate processes communicating
    via pipes, enabling true parallelism on multi-core systems. In sequential
    mode, environments run in the main process, useful for debugging.

    Attributes:
        parallel: Whether running in parallel mode.
        length: Number of environment instances.
        act_space: Action space (dict of Space objects).
        callbacks: List of callback functions called on each step.
        carry: Current policy carry state.
        acts: Current actions to execute.

    Example:
        >>> driver = Driver([make_env] * 4, parallel=True)
        >>> driver.on_step(lambda trn, i: print(f"Env {i}: {trn['reward']}"))
        >>> driver(my_policy, episodes=100)
        >>> driver.close()
    """

    def __init__(self, make_env_fns, parallel=True, **kwargs):
        """Initialize the Driver with environment factory functions.

        Args:
            make_env_fns: List of callables that create environment instances.
                Each callable should return an environment with obs_space,
                act_space attributes and step(action) method.
            parallel: If True, run environments in separate processes.
                If False, run sequentially in the main process.
            **kwargs: Additional keyword arguments passed to policy and callbacks.
        """
        assert len(make_env_fns) >= 1
        self.parallel = parallel
        self.kwargs = kwargs
        self.length = len(make_env_fns)
        if parallel:
            import multiprocessing as mp

            context = mp.get_context()
            self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
            self.stop = context.Event()
            fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
            self.procs = [
                portal.Process(self._env_server, self.stop, i, pipe, fn, start=True)
                for i, (fn, pipe) in enumerate(zip(fns, pipes))
            ]
            self.pipes[0].send(("act_space",))
            self.act_space = self._receive(self.pipes[0])
        else:
            self.envs = [fn() for fn in make_env_fns]
            self.act_space = self.envs[0].act_space
        self.callbacks = []
        self.acts: dict[str, np.ndarray] | None = None
        self.carry: typing.Any = None
        self.reset()

    def reset(self, init_policy=None):
        """Reset the driver state for a new collection run.

        Initializes actions to zeros (with reset=True) and optionally
        initializes the policy carry state.

        Args:
            init_policy: Optional callable that takes batch size and returns
                initial policy carry state. If None, carry is set to None.
        """
        self.acts = {
            k: np.zeros((self.length,) + v.shape, v.dtype)
            for k, v in self.act_space.items()
        }
        self.acts["reset"] = np.ones(self.length, bool)
        self.carry = init_policy and init_policy(self.length)

    def close(self):
        """Close all environments and terminate worker processes.

        In parallel mode, kills all worker processes. In sequential mode,
        calls close() on each environment instance.
        """
        if self.parallel:
            [proc.kill() for proc in self.procs]
        else:
            [env.close() for env in self.envs]

    def on_step(self, callback):
        """Register a callback to be called after each environment step.

        Args:
            callback: Function called as callback(transition, env_index, **kwargs)
                where transition is a dict containing obs, actions, and outputs.
        """
        self.callbacks.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        """Run the policy for a specified number of steps or episodes.

        Collects experience by repeatedly stepping all environments with
        the policy until either the step or episode target is reached.

        Args:
            policy: Callable that takes (carry, obs, **kwargs) and returns
                (new_carry, actions, outputs). Actions and outputs are dicts.
            steps: Minimum number of environment steps to collect.
            episodes: Minimum number of complete episodes to collect.
        """
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        acts = self.acts
        assert acts is not None  # Always set by reset()
        assert all(len(x) == self.length for x in acts.values())
        assert all(isinstance(v, np.ndarray) for v in acts.values())
        acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
        if self.parallel:
            [pipe.send(("step", act)) for pipe, act in zip(self.pipes, acts)]
            obs_list = [self._receive(pipe) for pipe in self.pipes]
        else:
            obs_list = [env.step(act) for env, act in zip(self.envs, acts)]
        obs = {k: np.stack([x[k] for x in obs_list]) for k in obs_list[0].keys()}
        logs = {k: v for k, v in obs.items() if k.startswith("log/")}
        obs = {k: v for k, v in obs.items() if not k.startswith("log/")}
        assert all(len(x) == self.length for x in obs.values()), obs
        self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
        assert all(k not in acts for k in outs), (list(outs.keys()), list(acts.keys()))
        if obs["is_last"].any():
            mask = ~obs["is_last"]
            acts = {k: self._mask(v, mask) for k, v in acts.items()}
        reset = obs["is_last"].copy()
        self.acts = {**acts, "reset": reset}
        trans = {**obs, **acts, **outs, **logs, "reset": reset}
        for i in range(self.length):
            trn = elements.tree.map(lambda x: x[i], trans)
            [fn(trn, i, **self.kwargs) for fn in self.callbacks]
        step += len(obs["is_first"])
        episode += obs["is_last"].sum()
        return step, episode

    def _mask(self, value, mask):
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        return value * mask.astype(value.dtype)

    def _receive(self, pipe):
        try:
            msg, arg = pipe.recv()
            if msg == "error":
                raise RuntimeError(arg)
            assert msg == "result"
            return arg
        except Exception:
            print("Terminating workers due to an exception.")
            [proc.kill() for proc in self.procs]
            raise

    @staticmethod
    def _env_server(stop, envid, pipe, ctor):  # pragma: no cover
        # Note: This method runs in subprocesses via portal.Process with cloudpickle.
        # Coverage.py cannot track cloudpickled functions even with subprocess coverage
        # enabled, as the deserialized code object cannot be mapped back to the source.
        # The code IS executed and tested, just not tracked by coverage tools.
        try:
            ctor = cloudpickle.loads(ctor)
            env = ctor()
            while not stop.is_set():
                if not pipe.poll(0.1):
                    time.sleep(0.1)
                    continue
                try:
                    msg, *args = pipe.recv()
                except EOFError:
                    return
                if msg == "step":
                    assert len(args) == 1
                    act = args[0]
                    obs = env.step(act)
                    pipe.send(("result", obs))
                elif msg == "obs_space":
                    assert len(args) == 0
                    pipe.send(("result", env.obs_space))
                elif msg == "act_space":
                    assert len(args) == 0
                    pipe.send(("result", env.act_space))
                else:
                    raise ValueError(f"Invalid message {msg}")
        except ConnectionResetError:
            print("Connection to driver lost")
        except Exception as e:
            pipe.send(("error", e))
            raise
        finally:
            try:
                env.close()
            except Exception:
                pass
            pipe.close()
