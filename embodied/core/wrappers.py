"""Environment wrappers for observation and action transformations.

This module provides a collection of wrapper classes that modify environment
behavior by transforming observations, actions, rewards, or episode boundaries.
All wrappers inherit from the base Wrapper class which provides transparent
attribute delegation to the wrapped environment.

Available wrappers:
    - Wrapper: Base class with attribute delegation
    - TimeLimit: Truncate episodes after a maximum duration
    - ActionRepeat: Repeat actions for multiple timesteps
    - ClipAction: Clip continuous actions to bounds
    - NormalizeAction: Normalize actions from [-1, 1] to env bounds
    - UnifyDtypes: Convert observation/action dtypes to standard types
    - CheckSpaces: Validate observations and actions against spaces
    - DiscretizeAction: Convert continuous actions to discrete bins
    - ResizeImage: Resize image observations to target size
    - BackwardReturn: Add backward discounted return to observations
    - AddObs: Add a constant observation key
    - RestartOnException: Restart environment on errors

Example:
    >>> env = SomeEnvironment()
    >>> env = TimeLimit(env, duration=1000)
    >>> env = ActionRepeat(env, repeat=4)
    >>> env = NormalizeAction(env)
    >>> obs = env.step({'action': np.zeros(4), 'reset': True})
"""

import functools
import time

import elements
import numpy as np


class Wrapper:
    """Base wrapper class providing transparent attribute delegation.

    All wrapper classes should inherit from this base class. It transparently
    delegates attribute access to the wrapped environment, allowing wrappers
    to be stacked without explicit forwarding of methods and properties.

    Attributes:
        env: The wrapped environment instance.
    """

    def __init__(self, env):
        """Initialize the wrapper with an environment.

        Args:
            env: Environment to wrap. Must implement step() method and have
                obs_space and act_space properties.
        """
        self.env = env

    def __len__(self):
        """Return the length of the wrapped environment."""
        return len(self.env)

    def __bool__(self):
        """Return the boolean value of the wrapped environment."""
        return bool(self.env)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment.

        Args:
            name: Attribute name to access.

        Returns:
            The attribute value from the wrapped environment.

        Raises:
            AttributeError: If name starts with '__' (dunder methods).
            ValueError: If the wrapped environment doesn't have the attribute.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self.env, name)
        except AttributeError:
            raise ValueError(name)


class TimeLimit(Wrapper):
    """Wrapper that truncates episodes after a maximum number of steps.

    Enforces a maximum episode length by setting is_last=True when the step
    count reaches the duration. Supports two modes: full reset (calls env
    reset) or soft reset (continues episode with is_first=True marker).

    Attributes:
        _duration: Maximum steps per episode (0 or None disables limit).
        _reset: If True, perform full env reset; if False, soft reset.
        _step: Current step count within episode.
        _done: Whether the current episode has ended.
    """

    def __init__(self, env, duration, reset=True):
        """Initialize the time limit wrapper.

        Args:
            env: Environment to wrap.
            duration: Maximum steps per episode. Set to 0 or None to disable.
            reset: If True, call env.step with reset=True on episode end.
                If False, continue without reset but mark is_first=True.
        """
        super().__init__(env)
        self._duration = duration
        self._reset = reset
        self._step = 0
        self._done = False

    def step(self, action):
        """Execute one step, enforcing the time limit.

        Args:
            action: Action dictionary with 'reset' key and action values.

        Returns:
            Observation dictionary with is_last=True if time limit reached.
        """
        if action["reset"] or self._done:
            self._step = 0
            self._done = False
            if self._reset:
                action.update(reset=True)
                return self.env.step(action)
            else:
                action.update(reset=False)
                obs = self.env.step(action)
                obs["is_first"] = True
                return obs
        self._step += 1
        obs = self.env.step(action)
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
        self._done = obs["is_last"]
        return obs


class ActionRepeat(Wrapper):
    """Wrapper that repeats each action for multiple environment steps.

    Applies the same action for a specified number of timesteps, accumulating
    rewards. Early termination occurs if the episode ends during repeats.
    Commonly used to reduce effective decision frequency while maintaining
    fine-grained environment simulation.

    Attributes:
        _repeat: Number of times to repeat each action.
    """

    def __init__(self, env, repeat):
        """Initialize the action repeat wrapper.

        Args:
            env: Environment to wrap.
            repeat: Number of times to repeat each action. Must be >= 1.
        """
        super().__init__(env)
        self._repeat = repeat

    def step(self, action):
        """Execute action repeatedly, accumulating rewards.

        Args:
            action: Action dictionary with 'reset' key and action values.

        Returns:
            Observation dictionary with accumulated reward from all repeats.
            Episode-ending observations (is_last, is_terminal) are preserved.
        """
        if action["reset"]:
            return self.env.step(action)
        reward = 0.0
        for _ in range(self._repeat):
            obs = self.env.step(action)
            reward += obs["reward"]
            if obs["is_last"] or obs["is_terminal"]:
                break
        obs["reward"] = np.float32(reward)
        return obs


class ClipAction(Wrapper):
    """Wrapper that clips continuous actions to specified bounds.

    Ensures all action values fall within [low, high] range. Useful for
    ensuring policy outputs stay within valid ranges for the environment.

    Attributes:
        _key: Action dictionary key to clip.
        _low: Lower bound for clipping.
        _high: Upper bound for clipping.
    """

    def __init__(self, env, key="action", low=-1, high=1):
        """Initialize the action clipping wrapper.

        Args:
            env: Environment to wrap.
            key: Action dictionary key to clip. Default 'action'.
            low: Lower bound for clipping. Default -1.
            high: Upper bound for clipping. Default 1.
        """
        super().__init__(env)
        self._key = key
        self._low = low
        self._high = high

    def step(self, action):
        """Execute step with clipped action values.

        Args:
            action: Action dictionary with values to clip.

        Returns:
            Observation from environment after applying clipped action.
        """
        clipped = np.clip(action[self._key], self._low, self._high)
        return self.env.step({**action, self._key: clipped})


class NormalizeAction(Wrapper):
    """Wrapper that normalizes actions from [-1, 1] to environment bounds.

    Allows policies to output actions in normalized [-1, 1] range while the
    wrapper converts them to the environment's native action bounds. Only
    normalizes dimensions with finite bounds; unbounded dimensions pass through.

    Attributes:
        _key: Action dictionary key to normalize.
        _space: Original action space from environment.
        _mask: Boolean mask for dimensions with finite bounds.
        _low: Lower bounds (original for finite, -1 for infinite).
        _high: Upper bounds (original for finite, 1 for infinite).
    """

    def __init__(self, env, key="action"):
        """Initialize the action normalization wrapper.

        Args:
            env: Environment to wrap.
            key: Action dictionary key to normalize. Default 'action'.
        """
        super().__init__(env)
        self._key = key
        self._space = env.act_space[key]
        self._mask = np.isfinite(self._space.low) & np.isfinite(self._space.high)
        self._low = np.where(self._mask, self._space.low, -1)
        self._high = np.where(self._mask, self._space.high, 1)

    @functools.cached_property
    def act_space(self):
        """Return modified action space with normalized bounds.

        Returns:
            Action space dictionary with normalized bounds [-1, 1] for
            dimensions that have finite original bounds.
        """
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = elements.Space(np.float32, self._space.shape, low, high)
        return {**self.env.act_space, self._key: space}

    def step(self, action):
        """Execute step with denormalized action values.

        Args:
            action: Action dictionary with normalized [-1, 1] values.

        Returns:
            Observation from environment after converting action to
            original bounds.
        """
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self.env.step({**action, self._key: orig})


# class ExpandScalars(Wrapper):
#
#   def __init__(self, env):
#     super().__init__(env)
#     self._obs_expanded = []
#     self._obs_space = {}
#     for key, space in self.env.obs_space.items():
#       if space.shape == () and key != 'reward' and not space.discrete:
#         space = elements.Space(space.dtype, (1,), space.low, space.high)
#         self._obs_expanded.append(key)
#       self._obs_space[key] = space
#     self._act_expanded = []
#     self._act_space = {}
#     for key, space in self.env.act_space.items():
#       if space.shape == () and not space.discrete:
#         space = elements.Space(space.dtype, (1,), space.low, space.high)
#         self._act_expanded.append(key)
#       self._act_space[key] = space
#
#   @functools.cached_property
#   def obs_space(self):
#     return self._obs_space
#
#   @functools.cached_property
#   def act_space(self):
#     return self._act_space
#
#   def step(self, action):
#     action = {
#         key: np.squeeze(value, 0) if key in self._act_expanded else value
#         for key, value in action.items()}
#     obs = self.env.step(action)
#     obs = {
#         key: np.expand_dims(value, 0) if key in self._obs_expanded else value
#         for key, value in obs.items()}
#     return obs
#
#
# class FlattenTwoDimObs(Wrapper):
#
#   def __init__(self, env):
#     super().__init__(env)
#     self._keys = []
#     self._obs_space = {}
#     for key, space in self.env.obs_space.items():
#       if len(space.shape) == 2:
#         space = elements.Space(
#             space.dtype,
#             (int(np.prod(space.shape)),),
#             space.low.flatten(),
#             space.high.flatten())
#         self._keys.append(key)
#       self._obs_space[key] = space
#
#   @functools.cached_property
#   def obs_space(self):
#     return self._obs_space
#
#   def step(self, action):
#     obs = self.env.step(action).copy()
#     for key in self._keys:
#       obs[key] = obs[key].flatten()
#     return obs
#
#
# class FlattenTwoDimActions(Wrapper):
#
#   def __init__(self, env):
#     super().__init__(env)
#     self._origs = {}
#     self._act_space = {}
#     for key, space in self.env.act_space.items():
#       if len(space.shape) == 2:
#         space = elements.Space(
#             space.dtype,
#             (int(np.prod(space.shape)),),
#             space.low.flatten(),
#             space.high.flatten())
#         self._origs[key] = space.shape
#       self._act_space[key] = space
#
#   @functools.cached_property
#   def act_space(self):
#     return self._act_space
#
#   def step(self, action):
#     action = action.copy()
#     for key, shape in self._origs.items():
#       action[key] = action[key].reshape(shape)
#     return self.env.step(action)


class UnifyDtypes(Wrapper):
    """Wrapper that standardizes observation and action dtypes.

    Converts all floating point types to float32, integer types to int32,
    and preserves uint8 (common for images). Ensures consistent dtypes
    across different environments for neural network compatibility.

    Attributes:
        _obs_space: Modified observation space with unified dtypes.
        _act_space: Modified action space with unified dtypes.
        _obs_outer: Dtype conversions for observations.
        _act_inner: Dtype conversions for actions.
    """

    def __init__(self, env):
        """Initialize the dtype unification wrapper.

        Args:
            env: Environment to wrap.
        """
        super().__init__(env)
        self._obs_space, _, self._obs_outer = self._convert(env.obs_space)
        self._act_space, self._act_inner, _ = self._convert(env.act_space)

    @property
    def obs_space(self):
        """Return observation space with unified dtypes."""
        return self._obs_space

    @property
    def act_space(self):
        """Return action space with unified dtypes."""
        return self._act_space

    def step(self, action):
        """Execute step with dtype conversions.

        Args:
            action: Action dictionary to convert and execute.

        Returns:
            Observation dictionary with unified dtypes.
        """
        action = action.copy()
        for key, dtype in self._act_inner.items():
            action[key] = np.asarray(action[key], dtype)
        obs = self.env.step(action)
        for key, dtype in self._obs_outer.items():
            obs[key] = np.asarray(obs[key], dtype)
        return obs

    def _convert(self, spaces):
        """Convert space dtypes to unified standard types.

        Args:
            spaces: Dictionary of Space objects.

        Returns:
            Tuple of (converted_spaces, before_dtypes, after_dtypes).
        """
        results, befores, afters = {}, {}, {}
        for key, space in spaces.items():
            before = after = space.dtype
            if np.issubdtype(before, np.floating):
                after = np.float32
            elif np.issubdtype(before, np.uint8):
                after = np.uint8
            elif np.issubdtype(before, np.integer):
                after = np.int32
            befores[key] = before
            afters[key] = after
            results[key] = elements.Space(after, space.shape, space.low, space.high)
        return results, befores, afters


class CheckSpaces(Wrapper):
    """Wrapper that validates observations and actions against their spaces.

    Performs runtime validation to catch bugs where observations or actions
    don't match their declared spaces. Useful during development and debugging.
    Raises detailed errors when validation fails.

    The wrapper also ensures observation and action keys don't overlap.
    """

    def __init__(self, env):
        """Initialize the space checking wrapper.

        Args:
            env: Environment to wrap.

        Raises:
            AssertionError: If observation and action space keys overlap.
        """
        assert not (env.obs_space.keys() & env.act_space.keys()), (
            env.obs_space.keys(),
            env.act_space.keys(),
        )
        super().__init__(env)

    def step(self, action):
        """Execute step with validation of actions and observations.

        Args:
            action: Action dictionary to validate and execute.

        Returns:
            Observation dictionary (validated).

        Raises:
            TypeError: If value has invalid type.
            ValueError: If value doesn't match its space constraints.
        """
        for key, value in action.items():
            self._check(value, self.env.act_space[key], key)
        obs = self.env.step(action)
        for key, value in obs.items():
            self._check(value, self.env.obs_space[key], key)
        return obs

    def _check(self, value, space, key):
        """Validate a value against its space.

        Args:
            value: Value to validate.
            space: Space object defining valid values.
            key: Name of the value for error messages.

        Raises:
            TypeError: If value has an invalid type.
            ValueError: If value doesn't fit in the space.
        """
        if not isinstance(
            value, (np.ndarray, np.generic, list, tuple, int, float, bool)
        ):
            raise TypeError(f"Invalid type {type(value)} for key {key}.")
        if value in space:
            return
        dtype = np.array(value).dtype
        shape = np.array(value).shape
        lowest, highest = np.min(value), np.max(value)
        raise ValueError(
            f"Value for '{key}' with dtype {dtype}, shape {shape}, "
            f"lowest {lowest}, highest {highest} is not in {space}."
        )


class DiscretizeAction(Wrapper):
    """Wrapper that converts continuous actions to discrete bins.

    Maps discrete action indices to evenly-spaced continuous values in [-1, 1].
    Useful for applying discrete action algorithms (DQN, etc.) to continuous
    control environments.

    Attributes:
        _dims: Number of action dimensions.
        _values: Array of discrete bin values in [-1, 1].
        _key: Action dictionary key to discretize.
    """

    def __init__(self, env, key="action", bins=5):
        """Initialize the action discretization wrapper.

        Args:
            env: Environment to wrap.
            key: Action dictionary key to discretize. Default 'action'.
            bins: Number of discrete bins per action dimension. Default 5.
        """
        super().__init__(env)
        self._dims = np.squeeze(env.act_space[key].shape, 0).item()
        self._values = np.linspace(-1, 1, bins)
        self._key = key

    @functools.cached_property
    def act_space(self):
        """Return modified action space with discrete actions.

        Returns:
            Action space dictionary with discrete integer action space.
        """
        space = elements.Space(np.int32, self._dims, 0, len(self._values))
        return {**self.env.act_space, self._key: space}

    def step(self, action):
        """Execute step by converting discrete indices to continuous values.

        Args:
            action: Action dictionary with discrete bin indices.

        Returns:
            Observation from environment after converting to continuous action.
        """
        continuous = np.take(self._values, action[self._key])
        return self.env.step({**action, self._key: continuous})


class ResizeImage(Wrapper):
    """Wrapper that resizes image observations to a target size.

    Automatically detects image observations (>1D with different size) and
    resizes them using nearest-neighbor interpolation. Requires PIL/Pillow.

    Attributes:
        _size: Target (height, width) for resizing.
        _keys: List of observation keys to resize.
        _Image: PIL Image module (imported lazily).
    """

    def __init__(self, env, size=(64, 64)):
        """Initialize the image resizing wrapper.

        Args:
            env: Environment to wrap.
            size: Target (height, width) tuple. Default (64, 64).
        """
        super().__init__(env)
        self._size = size
        self._keys = [
            k
            for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size
        ]
        print(f"Resizing keys {','.join(self._keys)} to {self._size}.")
        if self._keys:
            from PIL import Image

            self._Image = Image

    @functools.cached_property
    def obs_space(self):
        """Return observation space with resized image dimensions.

        Returns:
            Observation space dictionary with updated image shapes.
        """
        spaces = self.env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = elements.Space(np.uint8, shape)
        return spaces

    def step(self, action):
        """Execute step and resize image observations.

        Args:
            action: Action dictionary.

        Returns:
            Observation dictionary with resized images.
        """
        obs = self.env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        """Resize a single image using nearest-neighbor interpolation.

        Args:
            image: Input image as numpy array.

        Returns:
            Resized image as numpy array.
        """
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


# class RenderImage(Wrapper):
#
#   def __init__(self, env, key='image'):
#     super().__init__(env)
#     self._key = key
#     self._shape = self.env.render().shape
#
#   @functools.cached_property
#   def obs_space(self):
#     spaces = self.env.obs_space
#     spaces[self._key] = elements.Space(np.uint8, self._shape)
#     return spaces
#
#   def step(self, action):
#     obs = self.env.step(action)
#     obs[self._key] = self.env.render()
#     return obs


class BackwardReturn(Wrapper):
    """Wrapper that adds discounted backward return to observations.

    Computes a running discounted sum of rewards (backward return) and adds
    it to observations. Resets on episode boundaries. Useful for reward
    shaping or auxiliary objectives.

    Attributes:
        _discount: Discount factor computed as (1 - 1/horizon).
        _bwreturn: Current accumulated backward return value.
    """

    def __init__(self, env, horizon):
        """Initialize the backward return wrapper.

        Args:
            env: Environment to wrap.
            horizon: Effective horizon for discount calculation.
                Discount = 1 - 1/horizon (e.g., horizon=100 -> discount=0.99).
        """
        super().__init__(env)
        self._discount = 1 - 1 / horizon
        self._bwreturn = 0.0

    @functools.cached_property
    def obs_space(self):
        """Return observation space with backward return added.

        Returns:
            Observation space dictionary including 'bwreturn' scalar.
        """
        return {
            **self.env.obs_space,
            "bwreturn": elements.Space(np.float32),
        }

    def step(self, action):
        """Execute step and compute backward return.

        Args:
            action: Action dictionary.

        Returns:
            Observation dictionary with 'bwreturn' key added.
        """
        obs = self.env.step(action)
        self._bwreturn *= (1 - obs["is_first"]) * self._discount
        self._bwreturn += obs["reward"]
        obs["bwreturn"] = np.float32(self._bwreturn)
        return obs


class AddObs(Wrapper):
    """Wrapper that adds a constant observation key to the environment.

    Injects a fixed value into observations at every step. Useful for adding
    metadata, environment identifiers, or constant features.

    Attributes:
        _key: Name of the observation key to add.
        _value: Constant value to inject.
        _space: Space object describing the value.
    """

    def __init__(self, env, key, value, space):
        """Initialize the add observation wrapper.

        Args:
            env: Environment to wrap.
            key: Name for the new observation key.
            value: Constant value to add to observations.
            space: Space object describing the value's type and bounds.
        """
        super().__init__(env)
        self._key = key
        self._value = value
        self._space = space

    @functools.cached_property
    def obs_space(self):
        """Return observation space with the new key added.

        Returns:
            Observation space dictionary including the new constant key.
        """
        return {
            **self.env.obs_space,
            self._key: self._space,
        }

    def step(self, action):
        """Execute step and add constant observation.

        Args:
            action: Action dictionary.

        Returns:
            Observation dictionary with constant value added.
        """
        obs = self.env.step(action)
        obs[self._key] = self._value
        return obs


class RestartOnException(Wrapper):
    """Wrapper that restarts the environment on specified exceptions.

    Provides fault tolerance by catching specified exceptions and recreating
    the environment. Tracks failure frequency within a time window to prevent
    infinite restart loops. Useful for environments that may crash occasionally.

    Attributes:
        _ctor: Constructor callable to create new environment instances.
        _exceptions: Tuple of exception types to catch.
        _window: Time window (seconds) for counting failures.
        _maxfails: Maximum failures allowed within window before raising.
        _wait: Seconds to wait before restarting after a failure.
        _last: Timestamp of last failure.
        _fails: Count of failures in current window.
    """

    def __init__(self, ctor, exceptions=(Exception,), window=300, maxfails=2, wait=20):
        """Initialize the restart-on-exception wrapper.

        Args:
            ctor: Callable that creates and returns a new environment instance.
            exceptions: Exception type(s) to catch and restart on.
                Default catches all exceptions.
            window: Time window in seconds for counting failures. Failures
                outside this window reset the counter. Default 300 (5 min).
            maxfails: Maximum allowed failures within window. Default 2.
            wait: Seconds to wait before restarting. Default 20.

        Raises:
            RuntimeError: If failures exceed maxfails within window.
        """
        if not isinstance(exceptions, (tuple, list)):
            exceptions = [exceptions]
        self._ctor = ctor
        self._exceptions = tuple(exceptions)
        self._window = window
        self._maxfails = maxfails
        self._wait = wait
        self._last = time.time()
        self._fails = 0
        super().__init__(self._ctor())

    def step(self, action):
        """Execute step with automatic restart on failures.

        Args:
            action: Action dictionary.

        Returns:
            Observation from environment (possibly after restart).

        Raises:
            RuntimeError: If environment crashes too many times within window.
        """
        try:
            return self.env.step(action)
        except self._exceptions as e:
            if time.time() > self._last + self._window:
                self._last = time.time()
                self._fails = 1
            else:
                self._fails += 1
            if self._fails > self._maxfails:
                raise RuntimeError("The env crashed too many times.")
            message = f"Restarting env after crash with {type(e).__name__}: {e}"
            print(message, flush=True)
            time.sleep(self._wait)
            self.env = self._ctor()
            action["reset"] = np.ones_like(action["reset"])
            return self.env.step(action)
