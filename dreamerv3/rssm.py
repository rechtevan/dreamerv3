"""Recurrent State-Space Model (RSSM) world model implementation.

This module implements the core components of the DreamerV3 world model:
- RSSM: Recurrent latent dynamics model with deterministic and stochastic states
- Encoder: Converts observations (images and vectors) to latent token representations
- Decoder: Reconstructs observations from latent features

The RSSM learns a compact representation of the environment by predicting future
latent states and reconstructing observations. It combines:
- Deterministic state (deter): GRU-like recurrent state capturing temporal dependencies
- Stochastic state (stoch): Categorical distributions capturing uncertainty
- Encoder/Decoder: Observation compression and reconstruction

Key Features:
- Block-structured GRU for efficient parallel computation
- Categorical latent distributions with unimix regularization
- Multi-layer encoders/decoders with configurable architectures
- Support for both image and vector observations
- KL divergence regularization (dynamics and representation losses)

Typical Usage:
    # Initialize world model components
    encoder = Encoder(obs_space)
    rssm = RSSM(act_space)
    decoder = Decoder(obs_space)

    # Encode observations to tokens
    _, _, tokens = encoder(carry, obs, reset, training)

    # Observe and update latent states
    carry, entries, feat = rssm.observe(carry, tokens, actions, reset, training)

    # Decode latent features to reconstructions
    _, _, recons = decoder(carry, feat, reset, training)
"""

import math

import einops
import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax
import embodied.jax.nets as nn


f32 = jnp.float32
sg = jax.lax.stop_gradient


class RSSM(nj.Module):
    """Recurrent State-Space Model for learning latent world dynamics.

    The RSSM learns a compact latent representation of the environment by combining:
    - Deterministic state (deter): Recurrent state updated via block-structured GRU
    - Stochastic state (stoch): Categorical distribution capturing uncertainty
    - Prior network: Predicts next stochastic state from deterministic state
    - Posterior network: Infers stochastic state from observations and deterministic state

    The model is trained with two KL divergence losses:
    - Dynamics loss (dyn): Regularizes prior predictions toward posterior
    - Representation loss (rep): Regularizes posterior toward prior

    Attributes:
        deter: Deterministic state dimension (must be divisible by blocks).
        hidden: Hidden layer size for MLPs.
        stoch: Number of stochastic categorical variables.
        classes: Number of classes per stochastic variable.
        norm: Normalization type ("rms", "layer", etc.).
        act: Activation function ("gelu", "relu", etc.).
        unroll: Whether to unroll scans for faster compilation.
        unimix: Uniform mixing probability for categorical distributions.
        outscale: Output layer initialization scale.
        imglayers: Number of layers in imagination (prior) network.
        obslayers: Number of layers in observation (posterior) network.
        dynlayers: Number of layers in dynamics (core) network.
        absolute: If True, only use tokens (not deter) for posterior inference.
        blocks: Number of blocks for block-structured GRU.
        free_nats: Minimum KL divergence (prevents over-regularization).
    """

    deter: int = 4096
    hidden: int = 2048
    stoch: int = 32
    classes: int = 32
    norm: str = "rms"
    act: str = "gelu"
    unroll: bool = False
    unimix: float = 0.01
    outscale: float = 1.0
    imglayers: int = 2
    obslayers: int = 1
    dynlayers: int = 1
    absolute: bool = False
    blocks: int = 8
    free_nats: float = 1.0

    def __init__(self, act_space, **kw):
        """Initialize RSSM with action space and network configuration.

        Args:
            act_space: Action space specification (dict of element.Space objects).
            **kw: Additional keyword arguments for network layers (e.g., winit, outscale).
        """
        assert self.deter % self.blocks == 0
        self.act_space = act_space
        self.kw = kw

    @property
    def entry_space(self):
        """Define the space of replay buffer entries.

        Returns:
            dict: Replay entry space with 'deter' and 'stoch' fields.
        """
        return dict(
            deter=elements.Space(np.float32, self.deter),
            stoch=elements.Space(np.float32, (self.stoch, self.classes)),
        )

    def initial(self, bsize):
        """Initialize RSSM state with zeros.

        Args:
            bsize: Batch size.

        Returns:
            dict: Initial carry state with zero-initialized 'deter' [B, deter] and
                'stoch' [B, stoch, classes] tensors.
        """
        carry = nn.cast(
            dict(
                deter=jnp.zeros([bsize, self.deter], f32),
                stoch=jnp.zeros([bsize, self.stoch, self.classes], f32),
            )
        )
        return carry

    def truncate(self, entries, carry=None):
        """Extract final state from sequence of entries.

        Used for truncated backpropagation through time, extracting the last
        timestep's state to use as initial state for the next training segment.

        Args:
            entries: Dict of RSSM entries with shape [B, T, ...].
            carry: Unused (for interface compatibility).

        Returns:
            dict: Carry state from last timestep [B, ...].
        """
        assert entries["deter"].ndim == 3, entries["deter"].shape
        carry = jax.tree.map(lambda x: x[:, -1], entries)
        return carry

    def starts(self, entries, carry, nlast):
        """Extract and flatten last N states from entries for imagination.

        Reshapes the last nlast states from [B, T, ...] to [B*nlast, ...] to use
        as starting points for parallel imagination rollouts.

        Args:
            entries: Dict of RSSM entries with shape [B, T, ...].
            carry: Current carry state (used to infer batch size).
            nlast: Number of recent states to extract.

        Returns:
            dict: Flattened entries from last nlast timesteps [B*nlast, ...].
        """
        B = len(jax.tree.leaves(carry)[0])
        return jax.tree.map(
            lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries
        )

    def observe(self, carry, tokens, action, reset, training, single=False):
        """Update RSSM states by observing encoded observations.

        Processes observations through the posterior network to infer latent states.
        Combines deterministic GRU dynamics with stochastic posterior inference.

        Args:
            carry: Current RSSM state dict with 'deter' [B, deter] and 'stoch' [B, stoch, classes].
            tokens: Encoded observation tokens [B, T, D] or [B, D] if single=True.
            action: Actions taken [B, T, ...] or [B, ...] if single=True.
            reset: Episode reset flags [B, T] or [B] if single=True.
            training: Whether in training mode.
            single: If True, process single timestep; if False, scan over time dimension.

        Returns:
            tuple: (carry, entries, feat) where:
                - carry: Updated RSSM state.
                - entries: Replay buffer entries (deter, stoch) [B, T, ...] or [B, ...].
                - feat: Features dict (deter, stoch, logit) [B, T, ...] or [B, ...].
        """
        carry, tokens, action = nn.cast((carry, tokens, action))
        if single:
            carry, (entry, feat) = self._observe(carry, tokens, action, reset, training)
            return carry, entry, feat
        else:
            unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
            carry, (entries, feat) = nj.scan(
                lambda carry, inputs: self._observe(carry, *inputs, training),
                carry,
                (tokens, action, reset),
                unroll=unroll,
                axis=1,
            )
            return carry, entries, feat

    def _observe(self, carry, tokens, action, reset, training):
        deter, stoch, action = nn.mask((carry["deter"], carry["stoch"], action), ~reset)
        action = nn.DictConcat(self.act_space, 1)(action)
        action = nn.mask(action, ~reset)
        deter = self._core(deter, stoch, action)
        tokens = tokens.reshape((*deter.shape[:-1], -1))
        x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
        for i in range(self.obslayers):
            x = self.sub(f"obs{i}", nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"obs{i}norm", nn.Norm, self.norm)(x))
        logit = self._logit("obslogit", x)
        stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
        carry = dict(deter=deter, stoch=stoch)
        feat = dict(deter=deter, stoch=stoch, logit=logit)
        entry = dict(deter=deter, stoch=stoch)
        assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
        return carry, (entry, feat)

    def imagine(self, carry, policy, length, training, single=False):
        """Imagine future latent trajectories using the prior network.

        Rolls out latent dynamics by predicting future states from actions,
        without using observations. Uses the prior network to predict stochastic
        states from deterministic states.

        Args:
            carry: Initial RSSM state dict with 'deter' [B, deter] and 'stoch' [B, stoch, classes].
            policy: Either a callable policy(state) -> action or precomputed actions [B, T, ...].
            length: Number of imagination steps.
            training: Whether in training mode.
            single: If True, imagine single step; if False, scan over time dimension.

        Returns:
            tuple: (carry, feat, action) where:
                - carry: Final RSSM state after imagination.
                - feat: Features dict (deter, stoch, logit) [B, T, ...] or [B, ...].
                - action: Actions taken during imagination [B, T, ...] or [B, ...].
        """
        if single:
            action = policy(sg(carry)) if callable(policy) else policy
            actemb = nn.DictConcat(self.act_space, 1)(action)
            deter = self._core(carry["deter"], carry["stoch"], actemb)
            logit = self._prior(deter)
            stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
            carry = nn.cast(dict(deter=deter, stoch=stoch))
            feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
            assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
            return carry, (feat, action)
        else:
            unroll = length if self.unroll else 1
            if callable(policy):
                carry, (feat, action) = nj.scan(
                    lambda c, _: self.imagine(c, policy, 1, training, single=True),
                    nn.cast(carry),
                    (),
                    length,
                    unroll=unroll,
                    axis=1,
                )
            else:
                carry, (feat, action) = nj.scan(
                    lambda c, a: self.imagine(c, a, 1, training, single=True),
                    nn.cast(carry),
                    nn.cast(policy),
                    length,
                    unroll=unroll,
                    axis=1,
                )
            # We can also return all carry entries but it might be expensive.
            # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
            # return carry, entries, feat, action
            return carry, feat, action

    def loss(self, carry, tokens, acts, reset, training):
        """Compute RSSM training losses.

        Computes KL divergence losses between prior and posterior distributions:
        - Dynamics loss (dyn): Trains prior to match posterior (forward prediction).
        - Representation loss (rep): Trains posterior to match prior (regularization).

        Args:
            carry: Initial RSSM state dict.
            tokens: Encoded observation tokens [B, T, D].
            acts: Actions taken [B, T, ...].
            reset: Episode reset flags [B, T].
            training: Whether in training mode.

        Returns:
            tuple: (carry, entries, losses, feat, metrics) where:
                - carry: Updated RSSM state.
                - entries: Replay buffer entries.
                - losses: Dict with 'dyn' and 'rep' losses [B, T].
                - feat: Features dict (deter, stoch, logit).
                - metrics: Dict with entropy metrics for logging.
        """
        metrics = {}
        carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
        prior = self._prior(feat["deter"])
        post = feat["logit"]
        dyn = self._dist(sg(post)).kl(self._dist(prior))
        rep = self._dist(post).kl(self._dist(sg(prior)))
        if self.free_nats:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)
        losses = {"dyn": dyn, "rep": rep}
        metrics["dyn_ent"] = self._dist(prior).entropy().mean()
        metrics["rep_ent"] = self._dist(post).entropy().mean()
        return carry, entries, losses, feat, metrics

    def _core(self, deter, stoch, action):
        stoch = stoch.reshape((stoch.shape[0], -1))
        action /= sg(jnp.maximum(1, jnp.abs(action)))
        g = self.blocks
        flat2group = lambda x: einops.rearrange(x, "... (g h) -> ... g h", g=g)
        group2flat = lambda x: einops.rearrange(x, "... g h -> ... (g h)", g=g)
        x0 = self.sub("dynin0", nn.Linear, self.hidden, **self.kw)(deter)
        x0 = nn.act(self.act)(self.sub("dynin0norm", nn.Norm, self.norm)(x0))
        x1 = self.sub("dynin1", nn.Linear, self.hidden, **self.kw)(stoch)
        x1 = nn.act(self.act)(self.sub("dynin1norm", nn.Norm, self.norm)(x1))
        x2 = self.sub("dynin2", nn.Linear, self.hidden, **self.kw)(action)
        x2 = nn.act(self.act)(self.sub("dynin2norm", nn.Norm, self.norm)(x2))
        x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
        x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
        for i in range(self.dynlayers):
            x = self.sub(f"dynhid{i}", nn.BlockLinear, self.deter, g, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"dynhid{i}norm", nn.Norm, self.norm)(x))
        x = self.sub("dyngru", nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
        gates = jnp.split(flat2group(x), 3, -1)
        reset, cand, update = [group2flat(x) for x in gates]
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter

    def _prior(self, feat):
        x = feat
        for i in range(self.imglayers):
            x = self.sub(f"prior{i}", nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"prior{i}norm", nn.Norm, self.norm)(x))
        return self._logit("priorlogit", x)

    def _logit(self, name, x):
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
        return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

    def _dist(self, logits):
        out = embodied.jax.outs.OneHot(logits, self.unimix)
        out = embodied.jax.outs.Agg(out, 1, jnp.sum)
        return out


class Encoder(nj.Module):
    """Encoder that converts observations to latent token representations.

    The encoder processes both vector and image observations:
    - Vector observations: Passed through MLP layers
    - Image observations: Processed through convolutional layers with spatial downsampling

    Outputs are concatenated into a unified token representation for the RSSM.

    Attributes:
        units: Hidden layer size for vector processing MLPs.
        norm: Normalization type ("rms", "layer", etc.).
        act: Activation function ("gelu", "relu", etc.).
        depth: Base channel depth for convolutional layers.
        mults: Tuple of depth multipliers for each conv layer.
        layers: Number of MLP layers for vector observations.
        kernel: Convolutional kernel size.
        symlog: If True, apply symlog squashing to vector observations.
        outer: If True, use different conv structure for first layer.
        strided: If True, use strided convolutions instead of max pooling.
    """

    units: int = 1024
    norm: str = "rms"
    act: str = "gelu"
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, **kw):
        """Initialize encoder with observation space.

        Args:
            obs_space: Dict of observation spaces (elements.Space objects).
            **kw: Additional keyword arguments for network layers.
        """
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.depths = tuple(self.depth * mult for mult in self.mults)
        self.kw = kw

    @property
    def entry_space(self):
        """Define replay buffer entry space (empty for encoder)."""
        return {}

    def initial(self, batch_size):
        """Return initial state (empty for encoder)."""
        return {}

    def truncate(self, entries, carry=None):
        """Return truncated state (empty for encoder)."""
        return {}

    def __call__(self, carry, obs, reset, training, single=False):
        """Encode observations into token representations.

        Processes vector and image observations through separate pathways, then
        concatenates into unified token representation.

        Args:
            carry: State (empty dict for encoder).
            obs: Dict of observations with keys matching obs_space.
            reset: Episode reset flags [B] or [B, T] if single=False.
            training: Whether in training mode.
            single: If True, process single timestep [B, ...]; if False, scan over time [B, T, ...].

        Returns:
            tuple: (carry, entries, tokens) where:
                - carry: Empty dict (encoder is stateless).
                - entries: Empty dict (no replay buffer entries).
                - tokens: Encoded observation tokens [B, D] or [B, T, D].
        """
        bdims = 1 if single else 2
        outs = []
        bshape = reset.shape

        if self.veckeys:
            vspace = {k: self.obs_space[k] for k in self.veckeys}
            vecs = {k: obs[k] for k in self.veckeys}
            squish = nn.symlog if self.symlog else lambda x: x
            x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
            x = x.reshape((-1, *x.shape[bdims:]))
            for i in range(self.layers):
                x = self.sub(f"mlp{i}", nn.Linear, self.units, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f"mlp{i}norm", nn.Norm, self.norm)(x))
            outs.append(x)

        if self.imgkeys:
            K = self.kernel
            imgs = [obs[k] for k in sorted(self.imgkeys)]
            assert all(x.dtype == jnp.uint8 for x in imgs)
            x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
            x = x.reshape((-1, *x.shape[bdims:]))
            for i, depth in enumerate(self.depths):
                if self.outer and i == 0:
                    x = self.sub(f"cnn{i}", nn.Conv2D, depth, K, **self.kw)(x)
                elif self.strided:
                    x = self.sub(f"cnn{i}", nn.Conv2D, depth, K, 2, **self.kw)(x)
                else:
                    x = self.sub(f"cnn{i}", nn.Conv2D, depth, K, **self.kw)(x)
                    B, H, W, C = x.shape
                    x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
                x = nn.act(self.act)(self.sub(f"cnn{i}norm", nn.Norm, self.norm)(x))
            assert 3 <= x.shape[-3] <= 16, x.shape
            assert 3 <= x.shape[-2] <= 16, x.shape
            x = x.reshape((x.shape[0], -1))
            outs.append(x)

        x = jnp.concatenate(outs, -1)
        tokens = x.reshape((*bshape, *x.shape[1:]))
        entries = {}
        return carry, entries, tokens


class Decoder(nj.Module):
    """Decoder that reconstructs observations from latent features.

    The decoder processes latent features (deter and stoch) to reconstruct observations:
    - Vector observations: Reconstructed through MLP layers and distribution heads
    - Image observations: Reconstructed through transposed convolutions with upsampling

    Supports both strided convolutions and explicit upsampling strategies.

    Attributes:
        units: Hidden layer size for MLP processing.
        norm: Normalization type ("rms", "layer", etc.).
        act: Activation function ("gelu", "relu", etc.).
        outscale: Output layer initialization scale.
        depth: Base channel depth for convolutional layers.
        mults: Tuple of depth multipliers for each conv layer.
        layers: Number of MLP layers for vector reconstruction.
        kernel: Convolutional kernel size.
        symlog: If True, use symlog MSE loss for vector observations.
        bspace: Block space dimension for efficient spatial reconstruction.
        outer: If True, use different conv structure for last layer.
        strided: If True, use strided transposed convolutions instead of upsampling.
    """

    units: int = 1024
    norm: str = "rms"
    act: str = "gelu"
    outscale: float = 1.0
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    bspace: int = 8
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, **kw):
        """Initialize decoder with observation space.

        Args:
            obs_space: Dict of observation spaces (elements.Space objects).
            **kw: Additional keyword arguments for network layers.
        """
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.depths = tuple(self.depth * mult for mult in self.mults)
        self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
        self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
        self.kw = kw

    @property
    def entry_space(self):
        """Define replay buffer entry space (empty for decoder)."""
        return {}

    def initial(self, batch_size):
        """Return initial state (empty for decoder)."""
        return {}

    def truncate(self, entries, carry=None):
        """Return truncated state (empty for decoder)."""
        return {}

    def __call__(self, carry, feat, reset, training, single=False):
        """Decode latent features into observation reconstructions.

        Processes RSSM latent features (deter and stoch) to reconstruct observations
        through separate pathways for vectors and images.

        Args:
            carry: State (empty dict for decoder).
            feat: Dict with 'deter' [B, deter] or [B, T, deter] and 'stoch' [B, stoch, classes]
                or [B, T, stoch, classes] latent features.
            reset: Episode reset flags [B] or [B, T] if single=False.
            training: Whether in training mode.
            single: If True, process single timestep [B, ...]; if False, scan over time [B, T, ...].

        Returns:
            tuple: (carry, entries, recons) where:
                - carry: Empty dict (decoder is stateless).
                - entries: Empty dict (no replay buffer entries).
                - recons: Dict of reconstructed observations with distribution outputs.
        """
        assert feat["deter"].shape[-1] % self.bspace == 0
        K = self.kernel
        recons = {}
        bshape = reset.shape
        inp = [nn.cast(feat[k]) for k in ("stoch", "deter")]
        inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
        inp = jnp.concatenate(inp, -1)

        if self.veckeys:
            spaces = {k: self.obs_space[k] for k in self.veckeys}
            o1, o2 = "categorical", ("symlog_mse" if self.symlog else "mse")
            outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
            kw = dict(**self.kw, act=self.act, norm=self.norm)
            x = self.sub("mlp", nn.MLP, self.layers, self.units, **kw)(inp)
            x = x.reshape((*bshape, *x.shape[1:]))
            kw = dict(**self.kw, outscale=self.outscale)
            outs = self.sub("vec", embodied.jax.DictHead, spaces, outputs, **kw)(x)
            recons.update(outs)

        if self.imgkeys:
            factor = 2 ** (len(self.depths) - int(bool(self.outer)))
            minres = [int(x // factor) for x in self.imgres]
            assert 3 <= minres[0] <= 16, minres
            assert 3 <= minres[1] <= 16, minres
            shape = (*minres, self.depths[-1])
            if self.bspace:
                u, g = math.prod(shape), self.bspace
                x0, x1 = nn.cast((feat["deter"], feat["stoch"]))
                x1 = x1.reshape((*x1.shape[:-2], -1))
                x0 = x0.reshape((-1, x0.shape[-1]))
                x1 = x1.reshape((-1, x1.shape[-1]))
                x0 = self.sub("sp0", nn.BlockLinear, u, g, **self.kw)(x0)
                x0 = einops.rearrange(
                    x0, "... (g h w c) -> ... h w (g c)", h=minres[0], w=minres[1], g=g
                )
                x1 = self.sub("sp1", nn.Linear, 2 * self.units, **self.kw)(x1)
                x1 = nn.act(self.act)(self.sub("sp1norm", nn.Norm, self.norm)(x1))
                x1 = self.sub("sp2", nn.Linear, shape, **self.kw)(x1)
                x = nn.act(self.act)(self.sub("spnorm", nn.Norm, self.norm)(x0 + x1))
            else:
                x = self.sub("space", nn.Linear, shape, **kw)(inp)
                x = nn.act(self.act)(self.sub("spacenorm", nn.Norm, self.norm)(x))
            for i, depth in reversed(list(enumerate(self.depths[:-1]))):
                if self.strided:
                    kw = dict(**self.kw, transp=True)
                    x = self.sub(f"conv{i}", nn.Conv2D, depth, K, 2, **kw)(x)
                else:
                    x = x.repeat(2, -2).repeat(2, -3)
                    x = self.sub(f"conv{i}", nn.Conv2D, depth, K, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f"conv{i}norm", nn.Norm, self.norm)(x))
            if self.outer:
                kw = dict(**self.kw, outscale=self.outscale)
                x = self.sub("imgout", nn.Conv2D, self.imgdep, K, **kw)(x)
            elif self.strided:
                kw = dict(**self.kw, outscale=self.outscale, transp=True)
                x = self.sub("imgout", nn.Conv2D, self.imgdep, K, 2, **kw)(x)
            else:
                x = x.repeat(2, -2).repeat(2, -3)
                kw = dict(**self.kw, outscale=self.outscale)
                x = self.sub("imgout", nn.Conv2D, self.imgdep, K, **kw)(x)
            x = jax.nn.sigmoid(x)
            x = x.reshape((*bshape, *x.shape[1:]))
            split = np.cumsum([self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
            for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
                out = embodied.jax.outs.MSE(out)
                out = embodied.jax.outs.Agg(out, 3, jnp.sum)
                recons[k] = out

        entries = {}
        return carry, entries, recons
