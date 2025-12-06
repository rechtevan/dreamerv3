"""JAX transformation utilities for distributed model initialization and execution.

This module provides utilities for transforming Ninjax models to work with JAX's
distributed primitives (pmap, shard_map, jit). It handles:

- Model initialization with parameter sharding across devices
- Model application with activation sharding and partition rules
- Layer-level sharding constraints via callback system
- Multi-device and multi-host coordination

The two main entry points are:
- init(): Initialize model parameters with specified sharding
- apply(): Execute model forward pass with input/output sharding

These utilities enable efficient model parallelism and data parallelism for
large-scale DreamerV3 training.
"""

import re
import threading
import typing
from collections import Counter

import jax
import ninjax as nj
from jax.sharding import PartitionSpec as P

from . import nets as nn


LOCK = threading.Lock()


# Add tracer_sharding attribute to abstract values. This allows us to use
# shard_map based on layer callback shardings, even though JAX does not
# currently expose the shardings of tracer objects.
TRACER_SHARDINGS: dict[typing.Any, typing.Any] = {}


def init(
    fn,
    mesh,
    arg_shardings,
    param_partition_rules=(),
    act_partition_rules=(),
    static_argnums=(),
    dummy_inputs=(),
    print_partition=False,
):
    """Initialize a Ninjax model with parameter sharding across devices.

    Transforms a Ninjax function into an initialization function that creates
    model parameters distributed across devices according to partition rules.
    The initialization process:
    1. Converts fn to a pure Ninjax function if needed
    2. Evaluates parameter shapes using dummy inputs
    3. Resolves partition rules to determine parameter sharding
    4. JIT-compiles initialization with proper shardings
    5. Executes to create sharded parameters on devices

    Args:
        fn: Ninjax module or function to initialize. If not already pure
            (nj.pure), will be converted automatically.
        mesh: JAX mesh defining device topology for sharding.
        arg_shardings: Sharding specs for function arguments (params, seed, *args).
        param_partition_rules: List of (regex_pattern, PartitionSpec) tuples
            for parameter sharding. Patterns are matched against parameter paths
            (e.g., ".*encoder.*" matches all encoder parameters).
        act_partition_rules: List of (regex_pattern, PartitionSpec) tuples
            for activation sharding during initialization (rarely needed).
        static_argnums: Indices of arguments to treat as static (not traced).
        dummy_inputs: Example inputs matching (params, seed, *args) for shape
            inference. Used to determine parameter shapes before allocation.
        print_partition: If True, print which parameters match which rules.

    Returns:
        Tuple of (params, params_sharding):
            - params: Initialized parameter dict distributed across devices
            - params_sharding: Dict mapping parameter names to their NamedSharding

    Raises:
        Exception: If any parameter doesn't match any partition rule.

    Example:
        >>> from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        >>> mesh = Mesh(jax.devices(), ('d',))
        >>> param_rules = [
        ...     ('.*encoder.*', P('d', None)),  # Shard encoder on data axis
        ...     ('.*', P()),  # Replicate everything else
        ... ]
        >>> dummy = ({'seed': 0}, jax.random.PRNGKey(0), inputs)
        >>> params, sharding = init(
        ...     model_fn, mesh, arg_shardings, param_rules, dummy_inputs=dummy
        ... )
    """

    def init(fun, **jit_kwargs):
        if not getattr(fun, "_is_pure", False):
            fun = nj.pure(fun)

        def wrapper(*args, **kwargs):
            state, out = fun(*args, create=True, modify=True, ignore=True, **kwargs)
            del out
            return state, ()

        return wrapper

    fn = init(fn)

    def fn(*args, inner=fn):  # type: ignore[no-redef]
        params, seed, *args = args  # type: ignore[assignment]
        old = nn.LAYER_CALLBACK
        nn.LAYER_CALLBACK = create_layer_callback(mesh, act_partition_rules)
        params, _ = inner(params, *args, seed=seed)
        nn.LAYER_CALLBACK = old
        return params

    fn = jax.jit(fn, static_argnums=static_argnums)

    params_shapes = fn.eval_shape(*dummy_inputs)
    params_sharding, grouping = resolve_rules(
        params_shapes, param_partition_rules, mesh
    )
    if print_partition:
        print_grouping(grouping)

    fn = jax.jit(fn, arg_shardings, params_sharding, static_argnums, None)
    params = fn(*dummy_inputs)

    return params, params_sharding


def apply(
    fn,
    mesh,
    in_shardings,
    out_shardings,
    partition_rules=(),
    static_argnums=(),
    single_output=False,
    return_params=False,
    donate_params=False,
    # shard_map specific
    split_rng=True,
    use_shardmap=False,
    first_outnums=(),
):
    """Transform a Ninjax function for distributed execution with sharding.

    Wraps a Ninjax function to execute across devices with specified input/output
    sharding and activation partition rules. This is the main entry point for
    running models in production after initialization.

    The transformation:
    1. Wraps fn to handle params/seed/args unpacking
    2. Optionally uses shard_map for SPMD execution (advanced)
    3. Applies layer callbacks for activation sharding constraints
    4. JIT-compiles with input/output sharding specifications
    5. Returns a compiled function ready for execution

    Args:
        fn: Ninjax function to transform. Should be a pure function or module.
        mesh: JAX mesh defining device topology.
        in_shardings: Sharding specs for inputs. If donate_params=True, expects
            (donated_params, allocated_params, seed, *args), otherwise
            (params, seed, *args).
        out_shardings: Sharding specs for outputs. Can be single sharding or
            list of shardings for multiple outputs.
        partition_rules: List of (regex_pattern, PartitionSpec) tuples for
            activation sharding. Patterns match layer names (e.g., ".*attn.*").
        static_argnums: Indices of arguments to treat as static.
        single_output: If True, fn returns single value instead of tuple.
            The wrapper will unpack the single-element tuple.
        return_params: If True, return (params, *outputs) instead of just outputs.
            Useful for updating params during training.
        donate_params: If True, first input is donated params that can be
            mutated in-place. Enables memory-efficient parameter updates.
        split_rng: If True and use_shardmap=True, fold process index into RNG
            to ensure different randomness across devices.
        use_shardmap: If True, use jax.experimental.shard_map for SPMD execution.
            Enables more aggressive compiler optimizations but more complex.
        first_outnums: Indices of outputs to add leading singleton dimension
            when using shard_map. Used for reduction outputs.

    Returns:
        JIT-compiled function with signature matching fn but operating on
        sharded inputs/outputs.

    Example:
        >>> # Basic usage
        >>> apply_fn = apply(
        ...     model.forward,
        ...     mesh,
        ...     in_shardings=[params_sharding, seed_sharding, data_sharding],
        ...     out_shardings=output_sharding,
        ...     partition_rules=[('.*', P())],
        ... )
        >>> outputs = apply_fn(params, seed, inputs)
        >>>
        >>> # With parameter donation for memory efficiency
        >>> train_fn = apply(
        ...     agent.train,
        ...     mesh,
        ...     in_shardings=[donated_sharding, allocated_sharding, ...],
        ...     out_shardings=[params_sharding, loss_sharding],
        ...     donate_params=True,
        ...     return_params=True,
        ... )
        >>> new_params, loss = train_fn(old_params, static_params, seed, batch)
    """
    if single_output:
        assert len(out_shardings) == 1

    def fn(*args, inner=fn):  # type: ignore[no-redef]
        if donate_params:
            donated, allocated, seed, *args = args  # type: ignore[assignment]
            params = {**donated, **allocated}
        else:
            params, seed, *args = args  # type: ignore[assignment]
        if use_shardmap and len(mesh.devices) > 1 and split_rng:
            seed = jax.random.fold_in(seed, jax.lax.axis_index("d"))
        params, outs = inner(params, *args, seed=seed)
        outs = (outs,) if single_output else outs
        assert isinstance(outs, tuple)
        return (params, *outs) if return_params else outs

    if use_shardmap and len(mesh.devices) > 1:

        def fn(*args, inner=fn):
            outs = list(inner(*args))
            for i in first_outnums:
                outs[i] = jax.tree.map(lambda x: x[None], outs[i])
            return tuple(outs)

        from jax.experimental.shard_map import shard_map

        ispecs = list(jax.tree.map(lambda s: s.spec, in_shardings))
        for i in sorted(static_argnums):
            ispecs.insert(i, None)
        ispecs = tuple(ispecs)  # type: ignore[assignment]
        ospecs = jax.tree.map(lambda s: s.spec, out_shardings)
        fn = shard_map(fn, mesh, ispecs, ospecs, check_rep=False)

        def fn(*args, inner=fn):
            outs = list(inner(*args))
            for i in first_outnums:
                outs[i] = jax.tree.map(lambda x: x[0], outs[i])
            return tuple(outs)

    if single_output:

        def fn(*args, inner=fn):
            outs = inner(*args)
            assert len(outs) == 1
            return outs[0]

    if single_output:
        out_shardings = out_shardings[0]
    donate = [0] if donate_params else []

    if not use_shardmap:

        def fn(*args, inner=fn):
            with LOCK:
                old = nn.LAYER_CALLBACK
                nn.LAYER_CALLBACK = create_layer_callback(mesh, partition_rules)
                outs = inner(*args)
                nn.LAYER_CALLBACK = old
            return outs

    fn = jax.jit(fn, in_shardings, out_shardings, static_argnums, None, donate)

    return fn


def create_layer_callback(mesh, partition_rules):
    """Create a layer callback function for activation sharding.

    The callback is invoked by network layers (via nets.LAYER_CALLBACK) to
    apply sharding constraints to intermediate activations. It matches layer
    names against partition rules and applies the corresponding sharding.

    The callback also stores sharding information in TRACER_SHARDINGS to enable
    shard_map to access sharding of traced values (workaround for JAX limitation).

    Args:
        mesh: JAX mesh for creating NamedSharding objects.
        partition_rules: List of (regex_pattern, PartitionSpec) tuples.
            Patterns are matched against full layer paths.

    Returns:
        Callback function with signature (y, name) -> y_sharded where:
            - y: Activation tensor or tree to apply sharding to
            - name: Layer name (combined with current Ninjax scope)
            - y_sharded: Same structure as y but with sharding constraints

    Raises:
        Exception: If layer name doesn't match any partition rule.

    Example:
        >>> rules = [
        ...     ('.*encoder.*', P('d', None)),  # Shard encoder activations
        ...     ('.*', P()),  # Replicate all other activations
        ... ]
        >>> callback = create_layer_callback(mesh, rules)
        >>> # Callback is set globally and invoked by layer implementations
        >>> nets.LAYER_CALLBACK = callback
    """

    def layer_callback(y, name):
        name = f"{nj.ninjax.SCOPE}/{name}"
        for rule, spec in partition_rules:
            if re.search(rule, name):
                sharding = jax.sharding.NamedSharding(mesh, spec)

                def apply(y):
                    y = jax.lax.with_sharding_constraint(y, sharding)
                    if not hasattr(type(y), "tracer_shardings"):
                        type(y).tracer_sharding = property(
                            lambda self: TRACER_SHARDINGS[id(self)]
                        )
                    TRACER_SHARDINGS[id(y)] = sharding
                    return y

                return jax.tree.map(apply, y)
        else:
            raise Exception(f"No matching rule found for activation key: {name}")

    return layer_callback


def resolve_rules(params, partition_rules, mesh):
    """Resolve partition rules to concrete sharding specifications.

    Matches each parameter name against partition rules (regex patterns) and
    assigns corresponding PartitionSpecs. Groups parameters by which rule
    they matched for reporting/debugging.

    Args:
        params: Dict of parameter names to shape/dtype structs or arrays.
        partition_rules: List of (regex_pattern, PartitionSpec) tuples.
            If empty, defaults to [(".*", P())] (replicate all).
        mesh: JAX mesh for creating NamedSharding objects.

    Returns:
        Tuple of (sharding, grouping):
            - sharding: Dict mapping param names to NamedSharding objects
            - grouping: Dict mapping rule patterns to list of matched param names

    Raises:
        Exception: If any parameter doesn't match any rule.
        AssertionError: If not all parameters were assigned a sharding.

    Example:
        >>> params = {'encoder/w': ..., 'decoder/w': ..., 'bias': ...}
        >>> rules = [
        ...     ('.*encoder.*', P('d', None)),
        ...     ('.*decoder.*', P(None, 'd')),
        ...     ('.*', P()),
        ... ]
        >>> sharding, grouping = resolve_rules(params, rules, mesh)
        >>> # sharding['encoder/w'] = NamedSharding(mesh, P('d', None))
        >>> # grouping['.*encoder.*'] = ['encoder/w']
    """
    if len(partition_rules) == 0:
        partition_rules = [(".*", P())]
    params_spec, grouping = dict(), dict()  # type: ignore[var-annotated]
    for k in params.keys():
        for rule, spec in partition_rules:
            if re.search(rule, k):
                params_spec[k] = spec
                if rule not in grouping:
                    grouping[rule] = []
                grouping[rule].append(k)
                break
        else:
            raise Exception(f"No matching rule found for param key: {k}")
    assert set(params.keys()) == set(params_spec.keys())
    sharding = jax.tree.map(
        lambda spec: jax.sharding.NamedSharding(mesh, spec), params_spec
    )
    return sharding, grouping


def print_grouping(grouping):
    """Print a summary of how partition rules matched parameters.

    Displays which parameters matched each rule and how many times each
    parameter name pattern appeared. This is useful for debugging partition
    rules and understanding how parameters are distributed.

    Args:
        grouping: Dict mapping rule patterns to lists of matched parameter names.
            Typically obtained from resolve_rules().

    Example:
        >>> print_grouping(grouping)
        Partition rule ".*encoder.*" matches 42 param tensors
        - .../linear/kernel: 20
        - .../linear/bias: 20
        - .../norm/scale: 2
        Partition rule ".*" matches 8 param tensors
        - .../output/kernel: 4
        - .../output/bias: 4
    """
    for rule, ps in grouping.items():
        if len(ps) == 0:
            continue
        print(f'Partition rule "{rule}" matches {len(ps)} param tensors')
        ks_list = ["/".join(p.split("/")[-2:]) for p in ps]
        ks_counter = Counter(ks_list)
        ks_counts = ks_counter.most_common(len(ks_counter))
        ks_formatted = [f"- .../{k}: {v}" for k, v in ks_counts]
        print("\n".join(ks_formatted))
