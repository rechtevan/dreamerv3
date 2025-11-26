"""Optimizer utilities with gradient scaling and custom transformations.

This module provides an Optimizer wrapper that handles:
- Automatic gradient scaling for float16/bfloat16 training
- Multi-device gradient synchronization via pmean
- Comprehensive metrics tracking (grad norm, update RMS, parameter counts)
- Parameter summary printing during initialization

It also includes custom Optax gradient transformations:
- clip_by_agc: Adaptive Gradient Clipping based on parameter norms
- scale_by_rms: RMSprop-style adaptive learning rates
- scale_by_momentum: Momentum with optional Nesterov acceleration
"""

import math

import jax
import jax.numpy as jnp
import ninjax as nj
import optax

from . import internal, nets


f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


class Optimizer(nj.Module):
    """JAX optimizer wrapper with gradient scaling and metrics tracking.

    Wraps an Optax optimizer to provide:
    - Automatic gradient scaling for mixed-precision training (fp16/bf16)
    - Multi-device gradient averaging via lax.pmean
    - Comprehensive training metrics (loss, grad norm, parameter stats)
    - Parameter count summaries for model inspection
    - Gradient overflow detection and automatic scale adjustment

    When using float16/bfloat16 compute dtype, the optimizer automatically:
    - Scales gradients by a dynamic scale factor before backprop
    - Unscales gradients after backprop
    - Adjusts scale factor based on gradient overflow/underflow
    - Skips updates when gradients overflow (via optax.apply_if_finite)

    Attributes:
        modules: List of Ninjax modules to optimize parameters for.
        opt: Underlying Optax optimizer instance.
        step: Training step counter.
        scaling: Whether gradient scaling is enabled (for fp16/bf16).
        grad_scale: Current gradient scale factor (if scaling enabled).
        good_steps: Consecutive steps without overflow (if scaling enabled).
        summary_depth: Depth for parameter summary tree printing.

    Example:
        # Create optimizer for a model
        optimizer = Optimizer(
            modules=[encoder, decoder],
            opt=optax.adam(learning_rate=3e-4)
        )

        # Training step
        def loss_fn(x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        metrics = optimizer(loss_fn, x, y)
        print(metrics['opt/loss'], metrics['opt/grad_norm'])
    """

    summary_depth: int = 2

    def __init__(self, modules, opt):
        """Initialize the optimizer wrapper.

        Args:
            modules: Ninjax module(s) whose parameters should be optimized.
                Can be a single module or list/tuple of modules.
            opt: Optax optimizer instance (e.g., optax.adam(1e-4)).
                Will be wrapped with apply_if_finite if gradient scaling is enabled.
        """
        modules = modules if isinstance(modules, (list, tuple)) else (modules,)
        self.modules = modules
        self.opt = opt
        self.step = nj.Variable(jnp.array, 0, i32, name="step")
        self.scaling = jnp.float16 == nets.COMPUTE_DTYPE
        if self.scaling:
            self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
            self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name="grad_scale")
            self.good_steps = nj.Variable(jnp.array, 0, i32, name="good_steps")

    def __call__(self, lossfn, *args, has_aux=False, **kwargs):
        """Compute gradients, apply optimizer update, and return metrics.

        Performs a complete optimization step:
        1. Compute loss and gradients via nj.grad
        2. Apply gradient scaling (if enabled for fp16/bf16)
        3. Average gradients across devices (if multi-device)
        4. Apply optimizer update to parameters
        5. Update gradient scale factor (if scaling enabled)
        6. Compute and return metrics

        Args:
            lossfn: Loss function to differentiate. Should accept *args, **kwargs
                and return either a scalar loss or (loss, aux) if has_aux=True.
            *args: Positional arguments to pass to lossfn.
            has_aux: If True, lossfn returns (loss, aux). If False, lossfn returns
                just the loss scalar. Default False.
            **kwargs: Keyword arguments to pass to lossfn.

        Returns:
            If has_aux=False: Dictionary of metrics with keys like:
                - {name}/loss: Mean loss value
                - {name}/grad_norm: Global gradient norm
                - {name}/grad_rms: RMS of gradient values
                - {name}/update_rms: RMS of parameter updates
                - {name}/param_rms: RMS of parameter values
                - {name}/param_count: Total parameter count
                - {name}/updates: Training step number
                - {name}/grad_scale: Gradient scale factor (if scaling enabled)
                - {name}/grad_overflow: 1 if overflow, 0 otherwise (if scaling)

            If has_aux=True: Tuple of (metrics dict, aux) where aux is the
                auxiliary output from lossfn.
        """
        metrics = {}

        def lossfn2(*args, **kwargs):
            outs = lossfn(*args, **kwargs)
            loss, aux = outs if has_aux else (outs, None)
            assert loss.dtype == f32, (self.name, loss.dtype)
            assert loss.shape == (), (self.name, loss.shape)
            if self.scaling:
                loss *= sg(self.grad_scale.read())
            return loss, aux

        loss, params, grads, aux = nj.grad(lossfn2, self.modules, has_aux=True)(
            *args, **kwargs
        )
        if self.scaling:
            loss *= 1 / self.grad_scale.read()

        counts = {k: math.prod(v.shape) for k, v in params.items()}
        if nj.creating():
            print(self._summarize_params(counts, self.summary_depth))

        axes = internal.get_data_axes()
        if axes:
            grads = jax.tree.map(lambda x: jax.lax.pmean(x, axes), grads)

        if self.scaling:
            invscale = 1 / self.grad_scale.read()
            grads = jax.tree.map(lambda x: x * invscale, grads)

        state = self.sub("state", nj.Tree, self.opt.init, params)
        updates, new_state = self.opt.update(grads, state.read(), params)
        nj.context().update(optax.apply_updates(params, updates))
        state.write(new_state)
        grad_norm = optax.global_norm(grads)
        if self.scaling:
            self._update_scale(grads, jnp.isfinite(grad_norm))
            grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
            self.step.write(self.step.read() + i32(jnp.isfinite(grad_norm)))
            metrics["grad_scale"] = self.grad_scale.read()
            metrics["grad_overflow"] = f32(~jnp.isfinite(grad_norm))
        else:
            self.step.write(self.step.read() + 1)
        metrics["loss"] = loss.mean()
        metrics["updates"] = self.step.read()
        metrics["grad_norm"] = grad_norm
        metrics["grad_rms"] = nets.rms(grads)
        metrics["update_rms"] = nets.rms(updates)
        metrics["param_rms"] = nets.rms([x.values for x in self.modules])
        metrics["param_count"] = jnp.array(list(counts.values()), f32).sum()
        metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return (metrics, aux) if has_aux else metrics

    def _update_scale(self, grads, finite):
        """Update gradient scale factor based on overflow/underflow detection.

        Dynamically adjusts the gradient scale factor for fp16/bf16 training:
        - If gradients are finite and good_steps < 1000: increment counter
        - If gradients are finite and good_steps >= 1000: double scale factor
        - If gradients overflow (not finite): halve scale factor

        Args:
            grads: Gradient tree (not used, only finite flag matters).
            finite: Boolean indicating whether gradients are finite.

        Returns:
            The finite flag (unchanged).
        """
        keep = finite & (self.good_steps.read() < 1000)
        incr = finite & (self.good_steps.read() >= 1000)
        decr = ~finite
        self.good_steps.write(i32(keep) * (self.good_steps.read() + 1))
        self.grad_scale.write(
            jnp.clip(
                f32(keep) * self.grad_scale.read()
                + f32(incr) * self.grad_scale.read() * 2
                + f32(decr) * self.grad_scale.read() / 2,
                1e-4,
                1e5,
            )
        )
        return finite

    def _summarize_params(self, counts, depth):
        """Generate a hierarchical parameter count summary for printing.

        Creates a formatted summary showing total parameter counts grouped by
        module hierarchy up to a specified depth. Used during initialization
        to display model architecture information.

        Args:
            counts: Dictionary mapping parameter paths to their element counts.
            depth: Maximum hierarchy depth to display (e.g., 2 shows "module/submodule").

        Returns:
            Multi-line string with formatted parameter count summary.
        """
        lines = []
        pfxs = []
        for key in counts:
            parts = key.split("/")
            pfxs += ["/".join(parts[: i + 1]) for i in range(min(len(parts), depth))]
        subcounts = {
            prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
            for prefix in set(pfxs)
        }
        lines = [f"Optimizer {self.name} has {sum(counts.values()):,} params:"]
        for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
            lines.append(f"{count:>14,} {prefix}")
        return "\n".join(lines)


def clip_by_agc(clip=0.3, pmin=1e-3):
    """Adaptive Gradient Clipping based on parameter norms.

    Clips gradients relative to the norm of their corresponding parameters,
    preventing large updates that could destabilize training. Unlike global
    gradient clipping, AGC clips each parameter tensor independently based
    on its own norm.

    The update norm is clipped to: clip * max(pmin, ||param||)

    Args:
        clip: Maximum ratio of update norm to parameter norm (default: 0.3).
        pmin: Minimum parameter norm threshold to prevent division by zero
            (default: 1e-3). Prevents aggressive clipping for small parameters.

    Returns:
        Optax GradientTransformation that can be chained with other transforms.

    Example:
        >>> optimizer = optax.chain(
        ...     clip_by_agc(clip=0.3),
        ...     optax.adam(learning_rate=1e-4)
        ... )
    """

    def init_fn(params):
        return ()

    def update_fn(updates, state, params=None):
        def fn(param, update):
            unorm = jnp.linalg.norm(update.flatten(), 2)
            pnorm = jnp.linalg.norm(param.flatten(), 2)
            upper = clip * jnp.maximum(pmin, pnorm)
            return update * (1 / jnp.maximum(1.0, unorm / upper))

        updates = jax.tree.map(fn, params, updates) if clip else updates
        return updates, ()

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):
    """RMSprop-style adaptive learning rate scaling.

    Adapts the learning rate for each parameter based on the running average
    of squared gradients (RMS). This provides per-parameter adaptive learning
    rates that automatically adjust based on gradient magnitude history.

    Maintains exponential moving average of squared gradients:
        nu_t = beta * nu_{t-1} + (1 - beta) * grad^2
        update = grad / (sqrt(nu_t) + eps)

    Args:
        beta: Decay rate for the moving average of squared gradients
            (default: 0.999). Higher values = longer memory.
        eps: Small constant for numerical stability to prevent division
            by zero (default: 1e-8).

    Returns:
        Optax GradientTransformation that scales updates by inverse RMS.

    Example:
        >>> optimizer = optax.chain(
        ...     scale_by_rms(beta=0.999),
        ...     optax.scale(-1e-4)  # Learning rate
        ... )
    """

    def init_fn(params):
        nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
        step = jnp.zeros((), i32)
        return (step, nu)

    def update_fn(updates, state, params=None):
        step, nu = state
        step = optax.safe_int32_increment(step)
        nu = jax.tree.map(lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
        nu_hat = optax.bias_correction(nu, beta, step)
        updates = jax.tree.map(lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
        return updates, (step, nu)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):
    """Momentum-based gradient accumulation with optional Nesterov acceleration.

    Accumulates exponentially weighted moving average of gradients to smooth
    updates and accelerate learning. Nesterov momentum provides a "look-ahead"
    by applying momentum to the already-updated momentum, often improving
    convergence.

    Standard momentum:
        mu_t = beta * mu_{t-1} + (1 - beta) * grad
        update = mu_t

    Nesterov momentum (lookahead):
        mu_t = beta * mu_{t-1} + (1 - beta) * grad
        update = beta * mu_t + (1 - beta) * grad

    Args:
        beta: Decay rate for momentum accumulation (default: 0.9).
            Higher values = more smoothing, slower response to changes.
        nesterov: If True, use Nesterov accelerated momentum (default: False).
            Nesterov momentum often converges faster in practice.

    Returns:
        Optax GradientTransformation that returns momentum-smoothed updates.

    Example:
        >>> # Standard momentum
        >>> optimizer = optax.chain(
        ...     scale_by_momentum(beta=0.9),
        ...     optax.scale(-1e-3)
        ... )
        >>> # Nesterov momentum
        >>> optimizer = optax.chain(
        ...     scale_by_momentum(beta=0.9, nesterov=True),
        ...     optax.scale(-1e-3)
        ... )
    """

    def init_fn(params):
        mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
        step = jnp.zeros((), i32)
        return (step, mu)

    def update_fn(updates, state, params=None):
        step, mu = state
        step = optax.safe_int32_increment(step)
        mu = optax.update_moment(updates, mu, beta, 1)
        if nesterov:
            mu_nesterov = optax.update_moment(updates, mu, beta, 1)
            mu_hat = optax.bias_correction(mu_nesterov, beta, step)
        else:
            mu_hat = optax.bias_correction(mu, beta, step)
        return mu_hat, (step, mu)

    return optax.GradientTransformation(init_fn, update_fn)
