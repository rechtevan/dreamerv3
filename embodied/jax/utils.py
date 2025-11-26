"""JAX utility classes for normalization, parameter tracking, and layer scanning.

This module provides utilities for training neural networks with JAX:
- Normalize: Online normalization with multiple strategies (meanstd, percentile)
- SlowModel: Exponential moving average parameter tracking (target networks)
- LayerScan: Efficient sequential layer processing using jax.lax.scan

These utilities are used throughout DreamerV3 for stable training and efficient
computation.
"""

import functools

import jax
import jax.numpy as jnp
import ninjax as nj

from . import internal


sg = jax.lax.stop_gradient
f32 = jnp.float32
i32 = jnp.int32

COMPUTE_DTYPE = jnp.bfloat16


class Normalize(nj.Module):
    """Online normalization with multiple strategies for stable training.

    Maintains running statistics to normalize inputs dynamically during training.
    Supports multiple normalization strategies:
    - 'none': No normalization (returns 0.0, 1.0)
    - 'meanstd': Standard normalization using running mean and standard deviation
    - 'perc': Percentile-based normalization using configurable quantiles

    The module tracks exponential moving averages of statistics and optionally
    applies bias correction during early training steps. All statistics are
    synchronized across devices using pmean for multi-device setups.

    Attributes:
        rate: Update rate for running statistics (default: 0.01). Higher values
            adapt faster but are less stable.
        limit: Minimum standard deviation / scale to prevent division by zero
            (default: 1e-8).
        perclo: Lower percentile for percentile normalization (default: 5.0).
        perchi: Upper percentile for percentile normalization (default: 95.0).
        debias: If True, applies bias correction for early training steps
            (default: True).
        impl: Normalization strategy - 'none', 'meanstd', or 'perc'.

    Example:
        >>> # Standard normalization
        >>> norm = Normalize('meanstd')
        >>> mean, std = norm(values, update=True)
        >>> normalized = (values - mean) / std
        >>>
        >>> # Percentile normalization (robust to outliers)
        >>> norm = Normalize('perc')
        >>> offset, scale = norm(values, update=True)
        >>> normalized = (values - offset) / scale
    """

    rate: float = 0.01
    limit: float = 1e-8
    perclo: float = 5.0
    perchi: float = 95.0
    debias: bool = True

    def __init__(self, impl):
        """Initialize the normalization module.

        Args:
            impl: Normalization strategy - 'none', 'meanstd', or 'perc'.

        Raises:
            NotImplementedError: If impl is not a supported strategy.
        """
        self.impl = impl
        if self.debias and self.impl != "none":
            self.corr = nj.Variable(jnp.zeros, (), f32, name="corr")
        if self.impl == "none":
            pass
        elif self.impl == "meanstd":
            self.mean = nj.Variable(jnp.zeros, (), f32, name="mean")
            self.sqrs = nj.Variable(jnp.zeros, (), f32, name="sqrs")
        elif self.impl == "perc":
            self.lo = nj.Variable(jnp.zeros, (), f32, name="lo")
            self.hi = nj.Variable(jnp.zeros, (), f32, name="hi")
        else:
            raise NotImplementedError(self.impl)

    def __call__(self, x, update):
        """Optionally update statistics and return current normalization parameters.

        Args:
            x: Input values to potentially update statistics from.
            update: If True, update running statistics with x before returning.

        Returns:
            Tuple of (offset, scale) for normalization:
                - meanstd: (mean, std)
                - perc: (lower_percentile, scale)
                - none: (0.0, 1.0)
        """
        if update:
            self.update(x)
        return self.stats()

    def update(self, x):
        """Update running statistics with new values.

        Computes statistics from x and updates the exponential moving averages.
        For multi-device setups, statistics are synchronized across devices.

        Args:
            x: New input values to incorporate into running statistics.

        Raises:
            NotImplementedError: If impl is not a supported strategy.
        """
        x = sg(f32(x))
        if self.impl == "none":
            pass
        elif self.impl == "meanstd":
            self._update(self.mean, self._mean(x))
            self._update(self.sqrs, self._mean(jnp.square(x)))
        elif self.impl == "perc":
            self._update(self.lo, self._perc(x, self.perclo))
            self._update(self.hi, self._perc(x, self.perchi))
        else:
            raise NotImplementedError(self.impl)
        if self.debias and self.impl != "none":
            self._update(self.corr, 1.0)

    def stats(self):
        """Compute current normalization statistics.

        Returns bias-corrected statistics if debias=True. The bias correction
        compensates for the initialization bias of exponential moving averages
        during early training.

        Returns:
            Tuple of (offset, scale):
                - meanstd: (mean, std) where std >= limit
                - perc: (lower_percentile, upper - lower) where scale >= limit
                - none: (0.0, 1.0)

        Raises:
            NotImplementedError: If impl is not a supported strategy.
        """
        corr = 1.0
        if self.debias and self.impl != "none":
            corr /= jnp.maximum(self.rate, self.corr.read())
        if self.impl == "none":
            return 0.0, 1.0
        elif self.impl == "meanstd":
            mean = self.mean.read() * corr
            std = jnp.sqrt(jax.nn.relu(self.sqrs.read() * corr - mean**2))
            std = jnp.maximum(self.limit, std)
            return mean, std
        elif self.impl == "perc":
            lo, hi = self.lo.read() * corr, self.hi.read() * corr
            return sg(lo), sg(jnp.maximum(self.limit, hi - lo))
        else:
            raise NotImplementedError(self.impl)

    def _mean(self, x):
        x = x.mean()
        axes = internal.get_data_axes()
        if axes:
            x = jax.lax.pmean(x, axes)
        return x

    def _perc(self, x, q):
        axes = internal.get_data_axes()
        if axes:
            x = jax.lax.all_gather(x, axes)
        x = jnp.percentile(x, q)
        return x

    def _update(self, var, x):
        var.write((1 - self.rate) * var.read() + self.rate * sg(x))


class SlowModel:
    """Exponential moving average tracker for model parameters (target network).

    Maintains a slowly-updating copy of another model's parameters using
    exponential moving average (EMA). Commonly used for target networks in
    reinforcement learning (e.g., DQN, DDPG) to stabilize training by providing
    slowly-changing target values.

    The slow model is updated via: model = mix * source + (1 - mix) * model
    where mix is determined by the rate and every parameters.

    Attributes:
        source: Source model to track parameters from.
        model: Target model whose parameters are slowly updated.
        rate: Mixing rate for parameter updates (0 to 1). If rate < 0.5, it's
            the EMA decay rate. If rate == 1.0, parameters are copied directly.
        every: Update frequency - only update when count % every == 0.
        count: Update counter tracking how many times update() has been called.

    Example:
        >>> # Create a slowly-updating target network
        >>> source_net = MyNetwork()
        >>> target_net = MyNetwork()  # Same architecture
        >>> slow_target = SlowModel(target_net, source=source_net, rate=0.01, every=1)
        >>>
        >>> # During training
        >>> slow_target.update()  # Slowly sync target_net <- source_net
        >>> target_values = slow_target(obs)  # Use like the model
    """

    def __init__(self, model, *, source, rate=1.0, every=1):
        """Initialize the slow model tracker.

        Args:
            model: Target model whose parameters will be slowly updated.
            source: Source model to track parameters from.
            rate: Mixing rate (0 to 1). Must be 1.0 or < 0.5 to ensure stability.
            every: Update interval - only update when count % every == 0.
                Default is 1 (update every call).

        Raises:
            AssertionError: If rate is not 1.0 and not < 0.5.
        """
        assert rate == 1 or rate < 0.5, rate
        self.source = source
        self.model = model
        self.rate = rate
        self.every = every
        name = self.model.path + "_count"
        self.count = nj.Variable(jnp.zeros, (), i32, name=name)

    def __getattr__(self, name):
        self._initonce()
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self._initonce()
        return self.model(*args, **kwargs)

    def update(self):
        """Update the slow model's parameters toward the source model.

        Performs an exponential moving average update of the model's parameters:
            model = mix * source + (1 - mix) * model

        The mixing rate is self.rate if count % every == 0, otherwise 0 (no update).
        This allows controlling both the smoothing rate and update frequency.

        The model must be initialized (have parameters) before calling update.
        Initialization happens automatically on first use via _initonce().
        """
        self._initonce()
        mix = jnp.where(self.count.read() % self.every == 0, self.rate, 0)
        fn = lambda src, dst: mix * src + (1 - mix) * dst
        values = jax.tree.map(fn, self.source.values, self.model.values)
        [self.model.write(k, v) for k, v in values.items()]
        self.count.write(self.count.read() + 1)

    def _initonce(self, *args, method=None, **kwargs):
        assert self.source.values, "no parameters to track"
        if not self.model.values:
            p = self.model.path + "/"
            nj.context().update({p + k: v for k, v in self.source.values.items()})
        assert self.model.values.keys() == self.source.values.keys(), (
            self.model.values.keys(),
            self.source.values.keys(),
        )


class LayerScan:
    """Wrapper to apply a module sequentially using jax.lax.scan for efficiency.

    Converts a module's forward pass into a scanned operation, applying it
    repeatedly over a sequence dimension. This is memory-efficient compared to
    unrolling loops, as JAX can optimize scan operations to use constant memory.

    Commonly used for stacking transformer layers or recurrent layers where the
    same module is applied multiple times with different parameters (one set per
    layer).

    Attributes:
        module: The module to scan over.
        count: Number of times to apply the module.
        names: Tuple of method names to wrap with scan (default: ("__call__",)).

    Example:
        >>> # Apply a transformer layer 12 times efficiently
        >>> layer = TransformerLayer()
        >>> stacked = LayerScan(layer, count=12)
        >>> output = stacked(input)  # Applies layer 12 times with scan
    """

    def __init__(self, module, count, names=("__call__",)):
        """Initialize the layer scan wrapper.

        Args:
            module: Module to apply sequentially.
            count: Number of sequential applications (layer depth).
            names: Method names to wrap with scan (default: ("__call__",)).
        """
        self.module = module
        self.count = count
        self.names = names

    def __call__(self, *args, **kwargs):
        # Magic methods need to be forwarded explicitly.
        return self.__getattr__("__call__")(*args, **kwargs)

    def __getattr__(self, name):
        value = getattr(self.module, name)
        if name in self.names:
            assert callable(value)
            value = nj.pure(value, nested=True)
            value = functools.partial(layer_scan, value, self.module.path, self.count)
        return value


def layer_scan(fn, scope, count, inp, *args, **kwargs):
    """Apply a Ninjax function sequentially using jax.lax.scan.

    This function handles the complex logic of scanning over a Ninjax module while
    properly managing state variables. It separates module state into:
    - Inner state: Variables scoped within the module (scanned per layer)
    - Outer state: Variables outside the module (shared across layers)
    - Unchanging variables: Read but not modified
    - Changing variables: Modified during execution

    The scan operation applies the function 'count' times, efficiently reusing
    memory by only keeping one layer's activations in memory at a time.

    Args:
        fn: Pure Ninjax function to scan (obtained via nj.pure).
        scope: Module scope path to identify inner vs outer variables.
        count: Number of scan iterations (number of layers).
        inp: Input to the first layer (output of layer i feeds into layer i+1).
        *args: Additional arguments passed to fn (must match scan structure).
        **kwargs: Keyword arguments passed to fn.

    Returns:
        Output from the final scan iteration. If fn returns a tuple, all elements
        are returned; otherwise just the single output.

    Note:
        This is an internal utility used by LayerScan. Direct usage requires
        understanding of Ninjax state management.
    """
    isinner = lambda k: k.startswith(scope + "/")

    args_ = jax.tree.map(lambda x: x[0], args)  # Copy structure
    kwargs_ = jax.tree.map(lambda x: x, kwargs)  # Copy structure
    state_ = {k: v[0] if isinner(k) else v for k, v in nj.context().items()}
    state, _, accessed, modified, created = fn(
        state_,
        inp,
        *args_,
        ignore=True,
        track=True,
        seed=nj.seed(None, True),
        **kwargs_,
    )

    # print('-' * 79)
    # print('accessed:', accessed)
    # print('modified:', modified)
    # print('created:', created)

    inner = lambda xs: {k: v for k, v in xs.items() if isinner(k)}
    outer = lambda xs: {k: v for k, v in xs.items() if not isinner(k)}

    unchanging = {
        k: v
        for k, v in nj.context().items()
        if k in accessed and k not in modified and k not in created
    }
    unchanging_inner = inner(unchanging)
    unchanging_outer = outer(unchanging)

    creations = {k: v for k, v in state.items() if k in created}
    creations_inner = inner(creations)
    creations_outer = outer(creations)
    nj.context().update(creations_outer)
    del creations_inner  # Will be created inside the scan.

    # Inner values do not exist yet, so we only keep them in the creations. This
    # is fine, because inner values cannot change across scan iterations anyways.
    # Outer values can change over iterations, so we need to thread them even
    # during creation.
    changing_inner = inner(
        {
            # k: v for k, v in state.items()
            k: v
            for k, v in nj.context().items()
            if k in modified and k not in created
        }
    )
    changing_outer = outer({k: v for k, v in state.items() if k in modified})

    # f = lambda x: {k: v.shape for k, v in x.items()}
    # print('-' * 79)
    # print('unchanging_inner', f(unchanging_inner))
    # print('unchanging_outer', f(unchanging_outer))
    # print('creations_inner', f(inner(creations)))
    # print('creations_outer', f(creations_outer))
    # print('changing_inner', f(changing_inner))
    # print('changing_outer', f(changing_outer))

    def body(carry, x):
        inp, changing_outer = carry
        arg, seed, unchanging_inner, changing_inner = x
        state = {
            **unchanging_inner,
            **unchanging_outer,
            **changing_inner,
            **changing_outer,
        }
        state, out = fn(state, inp, *arg, **kwargs, seed=seed)
        out, *other = out if isinstance(out, tuple) else (out,)
        changing = {k: v for k, v in state.items() if k in modified}
        changing_inner = inner(changing)
        changing_outer = outer(changing)
        creations = {k: v for k, v in state.items() if k in created}
        creations_inner = inner(creations)
        carry = (out, changing_outer)
        y = (other, creations_inner, changing_inner)
        return carry, y

    seeds = nj.seed(count, True)
    carry, ys = jax.lax.scan(
        f=body,
        init=(inp, changing_outer),
        xs=(args, seeds, unchanging_inner, changing_inner),
        length=count,
    )
    out, changing_outer = carry
    other, creations_inner, changing_inner = ys

    if nj.context().modify:
        nj.context().update(creations_inner)
        nj.context().update(changing_inner)
        nj.context().update(changing_outer)

    return (out, *other) if len(other) else out
