"""Distribution output classes for neural network heads.

This module provides output distribution classes used by neural network heads
in DreamerV3. Each class wraps network outputs (logits, means, etc.) and provides:
- pred(): Get the predicted value (mode/mean/sample)
- loss(): Compute negative log-likelihood loss
- sample(): Sample from the distribution
- logp(): Log-probability of an event
- entropy(): Distribution entropy
- kl(): KL divergence from another distribution

Distribution Types:
- MSE, Huber: Regression outputs (mean squared error, Huber loss)
- Normal: Gaussian distribution for continuous values
- Binary: Bernoulli distribution for binary classification
- Categorical: Categorical distribution for multi-class discrete outputs
- OneHot, TwoHot: Discretized continuous distributions

Utility Classes:
- Output: Base class defining the distribution interface
- Agg: Aggregates distribution outputs over multiple dimensions
- Frozen: Wraps a distribution with stop_gradient for frozen predictions
- Concat: Concatenates multiple distributions along an axis

All distributions support batched operations and integrate seamlessly with
JAX's automatic differentiation.
"""

import functools

import jax
import jax.numpy as jnp


i32 = jnp.int32
f32 = jnp.float32
sg = jax.lax.stop_gradient


class Output:
    """Base class for distribution outputs from neural network heads.

    Defines the standard interface that all distribution output classes must implement.
    Provides methods for prediction, loss computation, sampling, and distribution
    statistics (entropy, KL divergence).

    The loss() method implements maximum likelihood training by computing the
    negative log-probability of the target under the predicted distribution.

    Subclasses must implement:
    - pred(): Return the predicted value (mode, mean, or sample)
    - sample(seed, shape): Sample from the distribution
    - logp(event): Compute log-probability of an event
    - entropy(): Compute distribution entropy
    - kl(other): Compute KL divergence from another distribution

    Example:
        >>> # Typical usage in a neural network head
        >>> logits = network(x)  # [B, num_classes]
        >>> output = Categorical(logits)
        >>> prediction = output.pred()  # Most likely class
        >>> loss = output.loss(target)  # Negative log-likelihood
        >>> sample = output.sample(seed)  # Random sample
    """

    def __repr__(self):
        name = type(self).__name__
        pred = self.pred()
        return f"{name}({pred.dtype}, shape={pred.shape})"

    def pred(self):
        """Return the predicted value (mode, mean, or sample).

        Returns:
            Array with the predicted value(s).
        """
        raise NotImplementedError

    def loss(self, target):
        """Compute negative log-likelihood loss for the target.

        Args:
            target: Target values (automatically stopped from gradient).

        Returns:
            Negative log-probability loss for each element.
        """
        return -self.logp(sg(target))

    def sample(self, seed, shape=()):
        """Sample from the distribution.

        Args:
            seed: JAX random key for sampling.
            shape: Additional shape dimensions for the sample.

        Returns:
            Sample from the distribution.
        """
        raise NotImplementedError

    def logp(self, event):
        """Compute log-probability of an event.

        Args:
            event: Event to compute log-probability for.

        Returns:
            Log-probability of the event.
        """
        raise NotImplementedError

    def prob(self, event):
        """Compute probability of an event.

        Args:
            event: Event to compute probability for.

        Returns:
            Probability of the event (exp of log-probability).
        """
        return jnp.exp(self.logp(event))

    def entropy(self):
        """Compute entropy of the distribution.

        Returns:
            Entropy value(s).
        """
        raise NotImplementedError

    def kl(self, other):
        """Compute KL divergence from another distribution.

        Args:
            other: Another distribution of the same type.

        Returns:
            KL divergence KL(self || other).
        """
        raise NotImplementedError


class Agg(Output):
    """Aggregates distribution outputs over multiple dimensions.

    Wraps another distribution and aggregates loss, entropy, and KL divergence
    over specified trailing dimensions. Commonly used when the output has
    spatial or temporal structure (e.g., images, sequences) and you want to
    compute total loss/entropy across those dimensions.

    The aggregation function (default: sum) is applied to the last `dims`
    dimensions. Log-probabilities are always summed (as they represent
    independent events), while loss/entropy use the configurable aggregation.

    Attributes:
        output: Wrapped distribution to aggregate.
        axes: List of axes to aggregate over (negative indices from the end).
        agg: Aggregation function (default: jnp.sum). Can use jnp.mean, etc.

    Example:
        >>> # Image reconstruction with per-pixel MSE
        >>> mean = decoder(latent)  # [B, H, W, C]
        >>> pixel_dist = MSE(mean)  # Per-pixel distribution
        >>> output = Agg(pixel_dist, dims=3)  # Sum over H, W, C
        >>> loss = output.loss(target_image)  # Scalar loss per batch
        >>>
        >>> # Sequence modeling with aggregated loss
        >>> logits = network(x)  # [B, T, vocab_size]
        >>> token_dist = Categorical(logits)
        >>> output = Agg(token_dist, dims=1)  # Sum over time
        >>> loss = output.loss(target_seq)  # Loss per sequence
    """

    def __init__(self, output, dims, agg=jnp.sum):
        """Initialize aggregation wrapper.

        Args:
            output: Distribution to wrap and aggregate.
            dims: Number of trailing dimensions to aggregate over.
            agg: Aggregation function (default: jnp.sum). Applied to loss/entropy.
        """
        self.output = output
        self.axes = [-i for i in range(1, dims + 1)]
        self.agg = agg

    def __repr__(self):
        name = type(self.output).__name__
        pred = self.pred()
        dims = len(self.axes)
        return f"{name}({pred.dtype}, shape={pred.shape}, agg={dims})"

    def pred(self):
        """Return prediction from wrapped distribution.

        Returns:
            Prediction (delegates to wrapped distribution).
        """
        return self.output.pred()

    def loss(self, target):
        """Compute aggregated loss.

        Args:
            target: Target values.

        Returns:
            Loss aggregated over specified dimensions.
        """
        loss = self.output.loss(target)
        return self.agg(loss, self.axes)

    def sample(self, seed, shape=()):
        """Sample from wrapped distribution.

        Args:
            seed: JAX random key.
            shape: Additional batch shape dimensions.

        Returns:
            Sample (delegates to wrapped distribution).
        """
        return self.output.sample(seed, shape)

    def logp(self, event):
        """Compute aggregated log-probability.

        Args:
            event: Event to evaluate.

        Returns:
            Log-probability summed over specified dimensions.
        """
        return self.output.logp(event).sum(self.axes)

    def prob(self, event):
        """Compute aggregated probability.

        Args:
            event: Event to evaluate.

        Returns:
            Probability summed over specified dimensions.
        """
        return self.output.prob(event).sum(self.axes)

    def entropy(self):
        """Compute aggregated entropy.

        Returns:
            Entropy aggregated over specified dimensions.
        """
        entropy = self.output.entropy()
        return self.agg(entropy, self.axes)

    def kl(self, other):
        """Compute aggregated KL divergence.

        Args:
            other: Another Agg distribution with the same structure.

        Returns:
            KL divergence aggregated over specified dimensions.
        """
        assert isinstance(other, Agg), other
        kl = self.output.kl(other.output)
        return self.agg(kl, self.axes)


class Frozen:
    """Wraps a distribution with stop_gradient for frozen predictions.

    Creates a version of a distribution where all method outputs have
    stop_gradient applied, preventing gradients from flowing back through
    the distribution. Commonly used for target networks, baseline values,
    or when you want to use predictions without training them.

    All method calls are delegated to the wrapped distribution, but results
    are automatically wrapped with jax.lax.stop_gradient.

    Attributes:
        output: The wrapped distribution to freeze.

    Example:
        >>> # Frozen target network for TD learning
        >>> target_logits = target_network(next_state)
        >>> target_dist = Categorical(target_logits)
        >>> frozen_target = Frozen(target_dist)
        >>> target_value = frozen_target.pred()  # No gradients flow
        >>>
        >>> # Frozen baseline for policy gradient
        >>> value_dist = Normal(value_network(state))
        >>> frozen_baseline = Frozen(value_dist)
        >>> baseline = frozen_baseline.pred()  # Use but don't train
    """

    def __init__(self, output):
        """Initialize frozen distribution wrapper.

        Args:
            output: Distribution to wrap with stop_gradient.
        """
        self.output = output

    def __getattr__(self, name):
        """Delegate attribute access to wrapped distribution.

        Args:
            name: Attribute/method name.

        Returns:
            Wrapped method that applies stop_gradient to results.

        Raises:
            AttributeError: If name starts with '__'.
            ValueError: If attribute doesn't exist on wrapped distribution.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            fn = getattr(self.output, name)
        except AttributeError:
            raise ValueError(name)
        return functools.partial(self._wrapper, fn)

    def _wrapper(self, fn, *args, **kwargs):
        """Wrap method call with stop_gradient.

        Args:
            fn: Method to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result with stop_gradient applied.
        """
        result = fn(*args, **kwargs)
        result = sg(result)
        return result


class Concat:
    """Concatenates multiple distributions along a specified axis.

    Splits inputs along an axis, applies different distributions to each segment,
    and concatenates the results. Useful for hybrid architectures where different
    parts of the output use different distribution types (e.g., mixed continuous
    and discrete action spaces).

    The axis is split at the specified midpoints, with each segment processed
    by its corresponding distribution. Method calls are distributed to the
    appropriate distribution based on the segment.

    Attributes:
        outputs: List of distributions to apply to each segment.
        midpoints: Tuple of indices where to split the axis (len = len(outputs) - 1).
        axis: Axis along which to split and concatenate.

    Example:
        >>> # Mixed action space: 3 continuous + 2 discrete actions
        >>> logits = network(state)  # [B, 5]
        >>> continuous_dist = Normal(logits[..., :3])
        >>> discrete_dist = Categorical(logits[..., 3:])
        >>> output = Concat(
        ...     outputs=[continuous_dist, discrete_dist],
        ...     midpoints=[3],
        ...     axis=-1
        ... )
        >>> actions = output.sample(key)  # Mixed action sample
        >>> loss = output.loss(expert_actions)  # Combined loss
    """

    def __init__(self, outputs, midpoints, axis):
        """Initialize concatenated distribution.

        Args:
            outputs: List of distributions, one per segment.
            midpoints: Split points along axis (must have len(outputs) - 1 elements).
            axis: Axis along which to split inputs and concatenate outputs.
        """
        assert len(midpoints) == len(outputs) - 1
        self.outputs = outputs
        self.midpoints = tuple(midpoints)
        self.axis = axis

    def __getattr__(self, name):
        """Delegate method calls to all distributions.

        Args:
            name: Method name to call on each distribution.

        Returns:
            Wrapped method that processes segments and concatenates results.

        Raises:
            AttributeError: If name starts with '__'.
            ValueError: If any distribution doesn't have the method.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            fns = [getattr(x, name) for x in self.outputs]
        except AttributeError:
            raise ValueError(name)
        return functools.partial(self._wrapper, fns)

    def _wrapper(self, fns, *args, **kwargs):
        """Apply methods to segments and concatenate results.

        Args:
            fns: List of methods to apply to each segment.
            *args: Positional arguments (will be sliced per segment).
            **kwargs: Keyword arguments (will be sliced per segment).

        Returns:
            Concatenated results from all segments.
        """
        los = (None,) + self.midpoints
        his = self.midpoints + (None,)
        results = []
        for fn, lo, hi in zip(fns, los, his):
            segment = [slice(None, None, None)] * (self.axis + 1)
            segment[self.axis] = slice(lo, hi, None)
            segment = tuple(segment)
            a, kw = jax.tree.map(lambda x: x[segment], (args, kwargs))
            results.append(fn(*a, **kw))
        return jax.tree.map(lambda *xs: jnp.concatenate(xs, self.axis), *results)


class MSE(Output):
    """Mean Squared Error output for regression tasks.

    Predicts a continuous value using mean squared error loss. Optionally
    applies a squashing function to the target before computing loss (e.g.,
    symlog for value prediction in RL).

    This is a deterministic output - pred() returns the mean directly.

    Attributes:
        mean: Predicted mean value.
        squash: Optional function to apply to targets before loss computation.

    Example:
        >>> # Standard regression
        >>> mean = network(x)  # [B, D]
        >>> output = MSE(mean)
        >>> loss = output.loss(targets)  # MSE loss
        >>>
        >>> # With symlog squashing (common in RL value prediction)
        >>> from embodied.jax import nets
        >>> output = MSE(mean, squash=nets.symlog)
        >>> loss = output.loss(rewards)
    """

    def __init__(self, mean, squash=None):
        """Initialize MSE output.

        Args:
            mean: Predicted mean values.
            squash: Optional function to apply to targets before loss (e.g., symlog).
        """
        self.mean = f32(mean)
        self.squash = squash or (lambda x: x)

    def pred(self):
        """Return the predicted mean.

        Returns:
            Mean values.
        """
        return self.mean

    def loss(self, target):
        """Compute mean squared error loss.

        Args:
            target: Target values (floating point).

        Returns:
            Squared error for each element.
        """
        assert jnp.issubdtype(target.dtype, jnp.floating), target.dtype
        assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
        return jnp.square(self.mean - sg(self.squash(f32(target))))


class Huber(Output):
    """Huber loss (Charbonnier loss) output for robust regression.

    A smooth approximation of L1 loss that is less sensitive to outliers than
    MSE while still being differentiable everywhere. Uses the soft Huber
    formulation: sqrt(distance² + eps²) - eps.

    This loss transitions from quadratic (like MSE) for small errors to linear
    (like L1) for large errors, providing robustness to outliers.

    Attributes:
        mean: Predicted mean value.
        eps: Smoothing parameter controlling the transition point (default: 1.0).
            Larger values make the loss smoother but less robust to outliers.

    Example:
        >>> # Robust value prediction
        >>> mean = network(x)  # [B, D]
        >>> output = Huber(mean, eps=1.0)
        >>> loss = output.loss(targets)  # Robust to outliers
    """

    def __init__(self, mean, eps=1.0):
        """Initialize Huber loss output.

        Args:
            mean: Predicted mean values.
            eps: Smoothing parameter for Huber loss (default: 1.0).
        """
        self.mean = f32(mean)
        self.eps = eps

    def pred(self):
        """Return the predicted mean.

        Returns:
            Mean values.
        """
        return self.mean

    def loss(self, target):
        """Compute Huber loss.

        Args:
            target: Target values (floating point).

        Returns:
            Huber loss for each element: sqrt(dist² + eps²) - eps.
        """
        assert jnp.issubdtype(target.dtype, jnp.floating), target.dtype
        assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
        dist = self.mean - sg(f32(target))
        return jnp.sqrt(jnp.square(dist) + jnp.square(self.eps)) - self.eps


class Normal(Output):
    """Gaussian (Normal) distribution for continuous outputs.

    Models continuous values using a normal distribution with learned mean
    and standard deviation. Commonly used for continuous action spaces,
    stochastic value predictions, and latent variables.

    Predictions return the mean (mode of the distribution). Sampling adds
    Gaussian noise scaled by stddev. Supports full probabilistic operations
    including entropy and KL divergence.

    Attributes:
        mean: Mean of the Gaussian distribution.
        stddev: Standard deviation (broadcastable to mean shape).

    Example:
        >>> # Continuous action distribution
        >>> mean = network(state)  # [B, action_dim]
        >>> output = Normal(mean, stddev=0.5)
        >>> action = output.sample(key)  # Sample action
        >>> logp = output.logp(action)  # Log-probability
        >>>
        >>> # For policy training
        >>> loss = output.loss(expert_action)  # Behavior cloning
    """

    def __init__(self, mean, stddev=1.0):
        """Initialize Normal distribution.

        Args:
            mean: Mean of the distribution.
            stddev: Standard deviation (scalar or array, default: 1.0).
        """
        self.mean = f32(mean)
        self.stddev = jnp.broadcast_to(f32(stddev), self.mean.shape)

    def pred(self):
        """Return the mean (mode) of the distribution.

        Returns:
            Mean values.
        """
        return self.mean

    def sample(self, seed, shape=()):
        """Sample from the Gaussian distribution.

        Args:
            seed: JAX random key.
            shape: Additional batch shape dimensions.

        Returns:
            Sample from N(mean, stddev).
        """
        sample = jax.random.normal(seed, shape + self.mean.shape, f32)
        return sample * self.stddev + self.mean

    def logp(self, event):
        """Compute log-probability under the Gaussian.

        Args:
            event: Values to evaluate (floating point).

        Returns:
            Log-probability for each element.
        """
        assert jnp.issubdtype(event.dtype, jnp.floating), event.dtype
        return jax.scipy.stats.norm.logpdf(f32(event), self.mean, self.stddev)

    def entropy(self):
        """Compute differential entropy.

        Returns:
            Entropy: 0.5 * log(2 * pi * stddev^2) + 0.5
        """
        return 0.5 * jnp.log(2 * jnp.pi * jnp.square(self.stddev)) + 0.5

    def kl(self, other):
        """Compute KL divergence from another Normal distribution.

        Args:
            other: Another Normal distribution.

        Returns:
            KL(self || other) for each element.
        """
        assert isinstance(other, type(self)), (self, other)
        return 0.5 * (
            jnp.square(self.stddev / other.stddev)
            + jnp.square(other.mean - self.mean) / jnp.square(other.stddev)
            + 2 * jnp.log(other.stddev)
            - 2 * jnp.log(self.stddev)
            - 1
        )


class Binary(Output):
    """Bernoulli (Binary) distribution for binary classification.

    Models binary outcomes (0 or 1) using a Bernoulli distribution with a
    learned logit. Commonly used for binary classification, binary masking,
    and episode termination prediction.

    Predictions return True/False based on logit sign. Sampling draws from
    the Bernoulli distribution with probability sigmoid(logit).

    Attributes:
        logit: Unnormalized log-odds (logit) for the positive class.

    Example:
        >>> # Episode termination prediction
        >>> logit = network(state)  # [B]
        >>> output = Binary(logit)
        >>> continues = output.pred()  # Boolean predictions
        >>> loss = output.loss(true_continues)  # Binary cross-entropy
        >>>
        >>> # Binary classification
        >>> logit = classifier(x)
        >>> output = Binary(logit)
        >>> prediction = output.sample(key)  # Stochastic 0/1 sample
    """

    def __init__(self, logit):
        """Initialize Binary distribution.

        Args:
            logit: Unnormalized log-odds for the positive class (p=1).
        """
        self.logit = f32(logit)

    def pred(self):
        """Return binary prediction based on logit sign.

        Returns:
            Boolean array: True if logit > 0, False otherwise.
        """
        return self.logit > 0

    def logp(self, event):
        """Compute log-probability of a binary event.

        Args:
            event: Binary events (0 or 1).

        Returns:
            Log-probability for each event.
        """
        event = f32(event)
        logp = jax.nn.log_sigmoid(self.logit)
        lognotp = jax.nn.log_sigmoid(-self.logit)
        return event * logp + (1 - event) * lognotp

    def sample(self, seed, shape=()):
        """Sample from the Bernoulli distribution.

        Args:
            seed: JAX random key.
            shape: Additional batch shape dimensions.

        Returns:
            Binary samples (0 or 1).
        """
        prob = jax.nn.sigmoid(self.logit)
        return jax.random.bernoulli(seed, prob, shape=shape + self.logit.shape)


class Categorical(Output):
    """Categorical distribution for discrete multi-class outputs.

    Models discrete choices from a finite set of classes using a categorical
    distribution. Commonly used for discrete action spaces, classification,
    and discrete latent variables.

    Predictions return the most likely class (argmax of logits). Sampling
    draws from the categorical distribution. Optionally mixes in a uniform
    distribution for exploration (unimix parameter).

    Attributes:
        logits: Unnormalized log-probabilities for each class.
        unimix: Mixture weight for uniform distribution (0 = no mixing).

    Example:
        >>> # Discrete action selection
        >>> logits = network(state)  # [B, num_actions]
        >>> output = Categorical(logits)
        >>> action = output.pred()  # Most likely action (argmax)
        >>> action_sample = output.sample(key)  # Stochastic action
        >>>
        >>> # With uniform mixing for exploration
        >>> output = Categorical(logits, unimix=0.01)  # 1% uniform
        >>> loss = output.loss(expert_action)  # Cross-entropy loss
    """

    def __init__(self, logits, unimix=0.0):
        """Initialize Categorical distribution.

        Args:
            logits: Unnormalized log-probabilities, shape [..., num_classes].
            unimix: Uniform mixing coefficient (0 to 1). If > 0, mixes in
                uniform distribution: p = (1-unimix)*softmax(logits) + unimix*uniform.
        """
        logits = f32(logits)
        if unimix:
            probs = jax.nn.softmax(logits, -1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - unimix) * probs + unimix * uniform
            logits = jnp.log(probs)
        self.logits = logits

    def pred(self):
        """Return the most likely class (argmax).

        Returns:
            Class indices (argmax of logits).
        """
        return jnp.argmax(self.logits, -1)

    def sample(self, seed, shape=()):
        """Sample from the categorical distribution.

        Args:
            seed: JAX random key.
            shape: Additional batch shape dimensions.

        Returns:
            Sampled class indices.
        """
        return jax.random.categorical(
            seed, self.logits, -1, shape + self.logits.shape[:-1]
        )

    def logp(self, event):
        """Compute log-probability of a class.

        Args:
            event: Class indices to evaluate.

        Returns:
            Log-probability for each event.
        """
        onehot = jax.nn.one_hot(event, self.logits.shape[-1])
        return (jax.nn.log_softmax(self.logits, -1) * onehot).sum(-1)

    def entropy(self):
        """Compute categorical entropy.

        Returns:
            Entropy: -sum(p * log(p))
        """
        logprob = jax.nn.log_softmax(self.logits, -1)
        prob = jax.nn.softmax(self.logits, -1)
        entropy = -(prob * logprob).sum(-1)
        return entropy

    def kl(self, other):
        logprob = jax.nn.log_softmax(self.logits, -1)
        logother = jax.nn.log_softmax(other.logits, -1)
        prob = jax.nn.softmax(self.logits, -1)
        return (prob * (logprob - logother)).sum(-1)


class OneHot(Output):
    """One-hot encoded categorical distribution with straight-through gradients.

    Wraps a Categorical distribution but returns one-hot vectors instead of
    class indices. Uses the straight-through estimator to allow gradients to
    flow through the discrete selection: gradients use the softmax probabilities
    while the forward pass uses hard one-hot vectors.

    This is commonly used for discrete latent variables in VAEs (e.g., VQ-VAE)
    and for categorical outputs that need to be fed into downstream networks
    as continuous vectors.

    Attributes:
        dist: Underlying Categorical distribution.

    Example:
        >>> # Discrete latent variable with straight-through gradients
        >>> logits = encoder(x)  # [B, num_classes]
        >>> output = OneHot(logits)
        >>> latent = output.pred()  # One-hot vector [B, num_classes]
        >>> # Forward: hard one-hot, Backward: soft probabilities
        >>>
        >>> # Categorical output for downstream network
        >>> logits = network(state)
        >>> output = OneHot(logits, unimix=0.01)
        >>> action_vec = output.sample(key)  # One-hot action vector
    """

    def __init__(self, logits, unimix=0.0):
        """Initialize one-hot categorical distribution.

        Args:
            logits: Unnormalized log-probabilities, shape [..., num_classes].
            unimix: Uniform mixing coefficient (0 to 1). See Categorical docs.
        """
        self.dist = Categorical(logits, unimix)

    def pred(self):
        """Return one-hot vector for the most likely class.

        Uses straight-through estimator: forward pass returns hard one-hot,
        backward pass uses softmax probabilities.

        Returns:
            One-hot vector with shape [..., num_classes].
        """
        index = self.dist.pred()
        return self._onehot_with_grad(index)

    def sample(self, seed, shape=()):
        """Sample a one-hot vector from the categorical distribution.

        Uses straight-through estimator for gradients.

        Args:
            seed: JAX random key.
            shape: Additional batch shape dimensions.

        Returns:
            One-hot sampled vector.
        """
        index = self.dist.sample(seed, shape)
        return self._onehot_with_grad(index)

    def logp(self, event):
        """Compute log-probability of a one-hot event.

        Args:
            event: One-hot encoded event vector.

        Returns:
            Log-probability of the event.
        """
        return (jax.nn.log_softmax(self.dist.logits, -1) * event).sum(-1)

    def entropy(self):
        """Compute categorical entropy.

        Returns:
            Entropy from the underlying categorical distribution.
        """
        return self.dist.entropy()

    def kl(self, other):
        """Compute KL divergence from another OneHot distribution.

        Args:
            other: Another OneHot distribution.

        Returns:
            KL divergence KL(self || other).
        """
        return self.dist.kl(other.dist)

    def _onehot_with_grad(self, index):
        """Convert index to one-hot with straight-through gradients.

        Forward pass: hard one-hot vector
        Backward pass: softmax probabilities

        Args:
            index: Class indices.

        Returns:
            One-hot vector with straight-through gradients.
        """
        value = jax.nn.one_hot(index, self.dist.logits.shape[-1], dtype=f32)
        probs = jax.nn.softmax(self.dist.logits, -1)
        value = sg(value) + (probs - sg(probs))
        return value


class TwoHot(Output):
    """Discretized continuous distribution using two-hot encoding.

    Represents continuous values by discretizing them into bins and using a
    categorical distribution over those bins. Uses "two-hot" encoding where
    continuous values between bins are represented as a weighted combination
    of the two nearest bins (linear interpolation).

    This approach allows modeling continuous values with discrete distributions,
    enabling better gradient flow and more stable training compared to direct
    regression. Commonly used for value prediction in DreamerV3 (rewards, values).

    The prediction is computed as a probability-weighted average of bin centers,
    using a numerically stable symmetric sum to avoid bias at initialization.
    Loss uses linear interpolation to assign soft targets to the two nearest bins.

    Attributes:
        logits: Unnormalized log-probabilities over bins.
        probs: Softmax probabilities over bins.
        bins: Bin centers (must match logits.shape[-1]).
        squash: Optional function to transform targets before discretization.
        unsquash: Optional inverse function to transform predictions back.

    Example:
        >>> # Value prediction with symlog squashing
        >>> from embodied.jax import nets
        >>> logits = network(state)  # [B, 255]
        >>> bins = jnp.linspace(-20, 20, 255)
        >>> output = TwoHot(logits, bins, squash=nets.symlog, unsquash=nets.symexp)
        >>> value = output.pred()  # Continuous value prediction
        >>> loss = output.loss(target_value)  # Two-hot classification loss
        >>>
        >>> # Reward prediction
        >>> logits = reward_head(features)  # [B, T, 255]
        >>> output = TwoHot(logits, bins, squash=nets.symlog, unsquash=nets.symexp)
        >>> rewards = output.pred()  # Predicted rewards
    """

    def __init__(self, logits, bins, squash=None, unsquash=None):
        """Initialize two-hot discretized distribution.

        Args:
            logits: Unnormalized log-probabilities, shape [..., num_bins].
            bins: Bin centers (must have length num_bins, dtype float32).
            squash: Optional transformation applied to targets before binning
                (e.g., symlog for better value scaling).
            unsquash: Optional inverse transformation applied to predictions
                (e.g., symexp to undo symlog).
        """
        logits = f32(logits)
        assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
        assert bins.dtype == f32, bins.dtype
        self.logits = logits
        self.probs = jax.nn.softmax(logits)
        self.bins = jnp.array(bins)
        self.squash = squash or (lambda x: x)
        self.unsquash = unsquash or (lambda x: x)

    def pred(self):
        """Return predicted continuous value.

        Computes probability-weighted average of bin centers using a numerically
        stable symmetric summation to avoid initialization bias. For symmetric
        bins with uniform probabilities, this ensures zero prediction.

        Returns:
            Predicted continuous values (unsquashed).
        """
        # The naive implementation results in a non-zero result even if the bins
        # are symmetric and the probabilities uniform, because the sum operation
        # goes left to right, accumulating numerical errors. Instead, we use a
        # symmetric sum to ensure that the predicted rewards and values are
        # actually zero at initialization.
        # return self.unsquash((self.probs * self.bins).sum(-1))
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
            return self.unsquash(wavg)
        else:
            p1 = self.probs[..., : n // 2]
            p2 = self.probs[..., n // 2 :]
            b1 = self.bins[..., : n // 2]
            b2 = self.bins[..., n // 2 :]
            wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)
            return self.unsquash(wavg)

    def loss(self, target):
        """Compute two-hot classification loss.

        Discretizes continuous targets using linear interpolation between the
        two nearest bins. Creates a soft target that assigns weights to the
        two nearest bins inversely proportional to their distances from the
        target value. Computes cross-entropy loss with these soft targets.

        Args:
            target: Continuous target values (float32).

        Returns:
            Negative log-likelihood loss (cross-entropy with soft targets).
        """
        assert target.dtype == f32, target.dtype
        target = sg(self.squash(target))
        below = (self.bins <= target[..., None]).astype(i32).sum(-1) - 1
        above = len(self.bins) - (self.bins > target[..., None]).astype(i32).sum(-1)
        below = jnp.clip(below, 0, len(self.bins) - 1)
        above = jnp.clip(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - target))
        dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - target))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None]
            + jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None]
        )
        log_pred = self.logits - jax.scipy.special.logsumexp(
            self.logits, -1, keepdims=True
        )
        return -(target * log_pred).sum(-1)
