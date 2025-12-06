"""JAX internal utilities for setup, distributed training, and checkpointing.

This module provides low-level utilities for configuring JAX runtime behavior,
managing multi-host distributed training, and handling model checkpoint operations
with sharding. These utilities are primarily used internally by DreamerV3's JAX
infrastructure.

Key functionality:
- setup(): Configure JAX platform, compilation, and optimization flags
- Multi-host coordination: device_put, to_local, to_global, move
- Mesh creation: mesh() for device topology specification
- Checkpoint I/O: grouped_ckpt_fns, ckpt_fn for efficient sharded checkpointing
- Named axes: get_named_axes, get_data_axes for inspecting parallel axes
"""

import concurrent.futures
import math
import os
import string

import elements
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from . import nets


def setup(
    platform=None,
    compute_dtype=jnp.bfloat16,
    debug=False,
    jit=True,
    prealloc=False,
    mock_devices=0,
    transfer_guard=True,
    deterministic=True,
    autotune=1,
    gpuflags=True,
    tpuflags=False,
    xladump=None,
    debug_nans=False,
    process_id=-1,
    num_processes=1,
    coordinator_address=None,
    compilation_cache=True,
):
    """Configure JAX runtime environment and optimization settings.

    Sets up JAX with platform-specific optimizations, debugging options, and
    distributed training configuration. This function should be called once at
    the start of training to establish the JAX environment.

    Platform-specific optimizations:
    - GPU: Enables pipelining, async collectives, disables rematerialization
    - TPU: Enables async collectives, data parallel optimizations
    - CPU: Mock device support for testing

    Args:
        platform: Target platform - 'gpu', 'tpu', 'cpu', or None for auto-detect.
        compute_dtype: Default dtype for computation (jnp.bfloat16, jnp.float32,
            or jnp.float16). Also accepts dtype name as string.
        debug: If True, disables most XLA optimizations for easier debugging.
        jit: If True, enables JIT compilation. Set False for debugging.
        prealloc: If True, preallocate GPU memory. False allows dynamic allocation.
        mock_devices: Number of mock CPU devices for testing (0 disables).
        transfer_guard: If True, disallows implicit host-device transfers to catch
            performance bugs. Disabled when debug_nans=True.
        deterministic: If True, forces deterministic GPU ops (may reduce performance).
        autotune: XLA GPU autotune level (0-4). Higher values increase compile time
            but may improve runtime performance.
        gpuflags: If True and platform='gpu', applies GPU-specific XLA flags for
            optimization (pipelining, async collectives, etc.).
        tpuflags: If True and platform='tpu', applies TPU-specific XLA flags.
        xladump: If set, dumps XLA HLO graphs to this directory for inspection.
        debug_nans: If True, instruments code to detect NaN/Inf at every operation.
            Useful for debugging but very slow.
        process_id: Process rank for multi-host training (0 to num_processes-1).
            Only used when num_processes > 1.
        num_processes: Total number of processes for multi-host training.
        coordinator_address: Address of coordinator process for multi-host setup
            (e.g., 'localhost:1234'). Required when num_processes > 1.
        compilation_cache: If True, enables persistent compilation cache to speed
            up repeated runs.

    Raises:
        AssertionError: If multi-host parameters are invalid (process_id < 0,
            missing coordinator_address, etc.).
    """
    platform and jax.config.update("jax_platforms", platform)
    jax.config.update("jax_disable_most_optimizations", debug)
    jax.config.update("jax_disable_jit", not jit)
    if transfer_guard and jit and not debug_nans:
        jax.config.update("jax_transfer_guard", "disallow")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(bool(prealloc)).lower()
    jax.config.update("jax_debug_nans", debug_nans)
    jax.config.update("jax_enable_compilation_cache", compilation_cache)

    xlaflags = []
    xlaflags.append(f"--xla_gpu_autotune_level={autotune}")
    if deterministic:
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        xlaflags.append("--xla_gpu_deterministic_ops=true")
    if mock_devices:
        xlaflags.append(f"--xla_force_host_platform_device_count={mock_devices}")
    if xladump:
        elements.Path(xladump).mkdir()
        xlaflags.append(f"--xla_dump_to={xladump}")
        xlaflags.append("--xla_dump_hlo_as_long_text")
    if gpuflags and platform == "gpu":
        # xla_flags.append('--xla_gpu_enable_latency_hiding_scheduler=true')
        # xla_flags.append('--xla_gpu_enable_async_all_gather=true')
        # xla_flags.append('--xla_gpu_enable_async_reduce_scatter=true')
        # xla_flags.append('--xla_gpu_enable_triton_gemm=false')
        # os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        # os.environ['NCCL_IB_SL'] = '1'
        # os.environ['NCCL_NVLS_ENABLE'] = '0'
        # os.environ['CUDA_MODULE_LOADING'] = 'EAGER'
        xlaflags += [
            "--xla_disable_hlo_passes=rematerialization",
            "--xla_gpu_all_gather_combine_threshold_bytes=134217728",
            "--xla_gpu_all_reduce_combine_threshold_bytes=134217728",
            "--xla_gpu_enable_all_gather_combine_by_dim=false",
            "--xla_gpu_enable_highest_priority_async_stream=true",
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            "--xla_gpu_enable_pipelined_all_gather=true",
            "--xla_gpu_enable_pipelined_all_reduce=true",
            "--xla_gpu_enable_pipelined_reduce_scatter=true",
            "--xla_gpu_enable_reduce_scatter_combine_by_dim=false",
            "--xla_gpu_enable_triton_gemm=false",
            "--xla_gpu_enable_triton_softmax_fusion=false",
            "--xla_gpu_enable_while_loop_double_buffering=true",
            "--xla_gpu_graph_level=0",
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=67108864",
        ]
    if tpuflags and platform == "tpu":
        xlaflags += [
            "--xla_disable_hlo_passes=rematerialization",
            "--xla_tpu_megacore_fusion_allow_ags=false",
            "--xla_enable_async_collective_permute=true",
            "--xla_tpu_enable_ag_backward_pipelining=true",
            "--xla_tpu_enable_data_parallel_all_reduce_opt=true",
            "--xla_tpu_data_parallel_opt_different_sized_ops=true",
            "--xla_tpu_enable_async_collective_fusion=true",
            "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_enable_async_all_gather=true",
        ]
    if xlaflags:
        os.environ["XLA_FLAGS"] = " ".join(xlaflags)

    if num_processes > 1 and platform != "tpu":
        # Note that the process_id is unrelated to the jax.process_index() that JAX
        # will assign later. It is only used to establish initial communication and
        # for error handling, whereas jax.process_index() depends on the underlying
        # hardware mesh.
        assert process_id >= 0
        assert coordinator_address
        jax.distributed.initialize(coordinator_address, num_processes, process_id)
        index, count = jax.process_index(), jax.process_count()
        print(f"JAX multi-host initialized: ({process_id}) {index} / {count}")

    if isinstance(compute_dtype, str):
        compute_dtype = getattr(jnp, compute_dtype)
    nets.COMPUTE_DTYPE = compute_dtype


def get_named_axes():
    """Get all currently active named parallel axes.

    Iterates through lowercase letters to find which axes are currently defined
    in the JAX parallel context (e.g., from pmap, shard_map, or xmap). This is
    useful for introspection and conditional parallel operations.

    Returns:
        List of active axis names (strings) in alphabetical order.

    Example:
        >>> # Inside a pmap with axis_name='d'
        >>> get_named_axes()
        ['d']
        >>> # Outside any parallel context
        >>> get_named_axes()
        []
    """
    axes = []
    for x in string.ascii_lowercase:
        try:
            jax.lax.axis_index(x)
        except NameError:
            continue
        axes.append(x)
    return axes


def get_data_axes():
    """Get data parallelism axes if available.

    Checks whether the standard data parallelism axes ('d' for data and 'f' for
    FSDP/model parallel) are currently active. Returns both axes if available,
    or an empty tuple if either is missing.

    This is used throughout DreamerV3 to conditionally apply pmean/all_gather
    operations when running in data-parallel mode.

    Returns:
        Tuple of ('d', 'f') if both axes are active, otherwise empty tuple ().

    Example:
        >>> # Inside data-parallel context
        >>> get_data_axes()
        ('d', 'f')
        >>> # Outside parallel context
        >>> get_data_axes()
        ()
    """
    axes = ("d", "f")
    for x in axes:
        try:
            jax.lax.axis_index(x)
        except NameError:
            return ()
    return axes


def fetch_async(value):
    """Asynchronously fetch arrays from device to host memory.

    Initiates non-blocking transfers of all arrays in the value tree from device
    to host memory. In multi-host setups, first converts global arrays to local
    shards. This is useful for overlapping data transfers with computation.

    Args:
        value: PyTree of JAX arrays to fetch asynchronously.

    Returns:
        The same value (arrays will be transferred asynchronously in background).
        Call .block_until_ready() on arrays to wait for completion.

    Example:
        >>> params = fetch_async(model_params)
        >>> # Do other work while transfer happens
        >>> # Wait for completion before accessing
        >>> jax.tree.map(lambda x: x.block_until_ready(), params)
    """
    if is_multihost():
        value = to_local(value)
    with jax._src.config.explicit_device_get_scope():
        [x.copy_to_host_async() for x in jax.tree.leaves(value)]
    return value


def is_multihost():
    """Check if running in multi-host distributed mode.

    Returns:
        True if jax.process_count() > 1, indicating multi-host setup.
        False for single-host training.
    """
    return jax.process_count() > 1


def device_put(value, sharding):
    """Place value on devices according to sharding specification.

    In multi-host mode, constructs global arrays from process-local data.
    In single-host mode, uses standard jax.device_put.

    Args:
        value: PyTree of arrays to place on devices.
        sharding: Target sharding specification (NamedSharding or PyTree of shardings).

    Returns:
        PyTree of sharded JAX arrays distributed across devices.

    Example:
        >>> from jax.sharding import NamedSharding, PartitionSpec as P
        >>> mesh = jax.sharding.Mesh(devices, ('d',))
        >>> sharding = NamedSharding(mesh, P('d'))
        >>> local_data = np.random.randn(8, 64)  # Local shard
        >>> global_array = device_put(local_data, sharding)
    """
    if is_multihost():
        with jax._src.config.explicit_device_put_scope():
            value = jax.tree.map(
                lambda x: jax.make_array_from_process_local_data(sharding, x), value
            )
    else:
        value = jax.device_put(value, sharding)
    return value


def local_sharding(sharding):
    """Convert global sharding to local mesh sharding.

    Creates a new sharding using only the local subset of devices from the
    global mesh. This is useful for operations that should only access local
    shards in multi-host setups.

    Args:
        sharding: Global NamedSharding or PyTree of NamedShardings.

    Returns:
        NamedSharding(s) using local_mesh instead of global mesh, with same spec.

    Example:
        >>> global_sharding = NamedSharding(global_mesh, P('d'))
        >>> local_sharding = local_sharding(global_sharding)
        >>> # local_sharding uses only this host's devices
    """
    return jax.tree.map(
        lambda s: jax.sharding.NamedSharding(s.mesh.local_mesh, s.spec), sharding
    )


def to_local(x):
    """Convert global arrays to local process shards.

    Transforms global arrays (spanning all hosts) to arrays containing only the
    shards local to this process. This is necessary before transferring data
    to host memory in multi-host setups.

    Args:
        x: PyTree of global JAX arrays.

    Returns:
        PyTree of local arrays (same structure, only local shards).

    Example:
        >>> # global_array spans all hosts
        >>> local_array = to_local(global_array)
        >>> # local_array contains only this host's shard
        >>> np.array(local_array)  # Can now transfer to host
    """
    return jax.tree.map(_to_local, x)


def _to_local(x):
    shape, sharding = x.shape, x.sharding
    spec, mesh = sharding.spec, sharding.mesh
    fullspec = [*spec, *([None] * (len(shape) - len(spec)))]
    assert len(shape) == len(fullspec)
    shard_shape = []
    for d, s in zip(shape, fullspec):
        if s is None:
            ms, lms = 1, 1
        else:
            if not isinstance(s, tuple):
                s = (s,)
            ms = math.prod(mesh.shape[si] for si in s)
            lms = math.prod(mesh.local_mesh.shape[si] for si in s)
        shard_shape.append(d // ms * lms)
    shard_shape = tuple(shard_shape)  # type: ignore[assignment]
    arrs = [arr.data for arr in x.addressable_shards]
    sharding_local = jax.sharding.NamedSharding(mesh.local_mesh, spec)
    x = jax.make_array_from_single_device_arrays(shard_shape, sharding_local, arrs)  # type: ignore[arg-type]
    return x


def to_global(x, global_sharding):
    """Convert local arrays to global arrays spanning all hosts.

    Transforms local process arrays to global arrays using the specified global
    sharding. This is the inverse of to_local() and is used to construct global
    views from local shards.

    Args:
        x: PyTree of local JAX arrays.
        global_sharding: Target global sharding (NamedSharding or PyTree of them).

    Returns:
        PyTree of global arrays spanning all hosts according to global_sharding.

    Example:
        >>> # local_array is this host's shard
        >>> global_array = to_global(local_array, global_sharding)
        >>> # global_array now spans all hosts
    """
    if isinstance(global_sharding, jax.sharding.NamedSharding):
        return jax.tree.map(lambda xi: _to_global(xi, global_sharding), x)
    else:
        return jax.tree.map(lambda xi, gs: _to_global(xi, gs), x, global_sharding)


def _to_global(x, global_sharding):
    shape, sharding = x.shape, x.sharding
    spec = sharding.spec
    fullspec = [*spec, *([None] * (len(shape) - len(spec)))]
    assert len(shape) == len(fullspec)
    shard_shape = []
    for d, s in zip(shape, fullspec):
        if s is None:
            ms, lms = 1, 1
        else:
            if not isinstance(s, tuple):
                s = (s,)
            ms = math.prod(global_sharding.mesh.shape[si] for si in s)
            lms = math.prod(sharding.mesh.shape[si] for si in s)
        shard_shape.append(d // lms * ms)
    shard_shape = tuple(shard_shape)  # type: ignore[assignment]
    arrs = [arr.data for arr in x.addressable_shards]
    x = jax.make_array_from_single_device_arrays(shard_shape, global_sharding, arrs)  # type: ignore[arg-type]
    return x


def move(xs, dst_sharding):
    """Move arrays to new sharding (potentially resharding across devices).

    Transfers arrays from their current sharding to a new target sharding,
    handling multi-host communication if necessary. In multi-host setups,
    uses a local-global-local round-trip for correct resharding.

    Args:
        xs: PyTree of JAX arrays to reshard.
        dst_sharding: Target sharding specification.

    Returns:
        PyTree of arrays with dst_sharding.

    Example:
        >>> # Reshard from data-parallel to model-parallel
        >>> from_sharding = NamedSharding(mesh, P('d', None))
        >>> to_sharding = NamedSharding(mesh, P(None, 'm'))
        >>> params = move(params, to_sharding)
    """
    if is_multihost():
        xs = to_local(xs)
        xs = jax.device_put(xs, local_sharding(dst_sharding))
        xs = to_global(xs, dst_sharding)
    else:
        xs = jax.device_put(xs, dst_sharding)
    return xs


def mesh(devices, shape, names):
    """Create a JAX device mesh for model/data parallelism.

    Arranges devices into a multi-dimensional mesh topology for partitioning
    computations and data across devices. The mesh shape determines how devices
    are organized, and names assign semantic meaning to each axis (e.g., 'data',
    'model', 'fsdp').

    Args:
        devices: Flat list of JAX devices to arrange into mesh.
        shape: Comma-separated string of mesh dimensions (e.g., "2,4" for 2x4 mesh).
            Use -1 for one dimension to infer from total device count.
        names: Tuple of axis names corresponding to shape dimensions
            (e.g., ('data', 'model')).

    Returns:
        jax.sharding.Mesh object with specified topology and axis names.

    Raises:
        AssertionError: If shape is invalid (more than one -1, or doesn't divide
            evenly into device count).

    Example:
        >>> devices = jax.devices()  # 8 GPUs
        >>> mesh = mesh(devices, "2,4", ('data', 'model'))
        >>> # Creates 2x4 mesh with 'data' axis (size 2) and 'model' axis (size 4)
        >>>
        >>> # Auto-infer one dimension
        >>> mesh = mesh(devices, "-1,2", ('data', 'model'))
        >>> # Infers first dimension as 8/2=4, creating 4x2 mesh
    """
    shape = list(map(int, shape.split(",")))
    # At most a single -1 is allowed
    assert sum(i == -1 for i in shape) <= 1
    n = len(devices)
    prod = math.prod(i for i in shape if i != -1)
    assert n % prod == 0
    shape = [i if i != -1 else n // prod for i in shape]
    assert math.prod(shape) == n
    devices = np.array(devices).reshape(shape)
    return jax.sharding.Mesh(devices, names)


def grouped_ckpt_fns(params, chunksize):
    """Create grouped checkpoint gather/shard functions for efficient I/O.

    Splits parameters into groups based on memory size and creates optimized
    JIT-compiled functions for gathering (collecting shards) and sharding
    (distributing) each group. This enables efficient checkpoint saving and
    loading without running out of memory or blocking on compilation.

    Grouping strategy:
    - If chunksize <= 0: Single group with all parameters
    - Otherwise: Greedy grouping to stay under chunksize bytes per group

    Args:
        params: Dict of parameter name -> JAX array (must all have same mesh).
        chunksize: Maximum bytes per checkpoint group (0 or negative disables
            grouping). Typical value: 1-10 GB to balance parallelism and overhead.

    Returns:
        List of (keys, gather_fn, shard_fn) tuples, one per group:
            - keys: List of parameter names in this group
            - gather_fn: Compiled function to gather shards to replicated
            - shard_fn: Compiled function to shard replicated to original sharding

    Example:
        >>> groups = grouped_ckpt_fns(params, chunksize=5 * 1024**3)  # 5 GB
        >>> # Save checkpoint
        >>> for keys, gather_fn, _ in groups:
        ...     group_params = {k: params[k] for k in keys}
        ...     gathered = gather_fn(group_params)
        ...     save_to_disk(keys, gathered)
        >>>
        >>> # Load checkpoint
        >>> for keys, _, shard_fn in groups:
        ...     loaded = load_from_disk(keys)
        ...     params.update(shard_fn(loaded))
    """
    if chunksize <= 0:
        groups = [list(params.keys())]
    else:
        groups = []
        keys, size = [], 0
        for k, v in params.items():
            if size + v.nbytes <= chunksize:
                keys.append(k)
                size += v.nbytes
            else:
                groups.append(keys)
                keys, size = [k], v.nbytes
        keys and groups.append(keys)  # type: ignore[func-returns-value]
    assert sum(len(keys) for keys in groups) == len(params)
    assert all(len(keys) for keys in groups)
    msg = f"Compiling {len(groups)} checkpoint groups..."
    elements.print(msg, color="yellow")
    maxsize = max(sum(params[k].nbytes for k in g) for g in groups)
    print(f"Largest checkpoint group: {maxsize / (1024**3):.0f} GB")

    gather_fns, shard_fns = [], []
    with concurrent.futures.ThreadPoolExecutor(64) as pool:
        for keys in groups:
            gather_fn, shard_fn = ckpt_fn({k: params[k] for k in keys}, compile=False)
            gather_fns.append(pool.submit(gather_fn.compile))
            shard_fns.append(pool.submit(shard_fn.compile))
    gather_fns = [future.result() for future in gather_fns]
    shard_fns = [future.result() for future in shard_fns]

    return list(zip(groups, gather_fns, shard_fns))


def ckpt_fn(params, compile=True):
    """Create checkpoint gather/shard functions for a parameter group.

    Creates two JIT-compiled functions:
    1. gather_fn: Collects distributed shards to fully replicated arrays (for saving)
    2. shard_fn: Distributes replicated arrays back to original sharding (for loading)

    These functions perform collective communication to gather/scatter parameters
    across devices while preserving the original sharding specification.

    Args:
        params: Dict of parameter name -> JAX array (must all share same mesh).
        compile: If True, immediately compile the functions. If False, return
            lowered IR that can be compiled later (useful for parallel compilation).

    Returns:
        Tuple of (gather_fn, shard_fn):
            - gather_fn: Function or lowered IR that gathers sharded -> replicated
            - shard_fn: Function or lowered IR that shards replicated -> original

    Example:
        >>> gather_fn, shard_fn = ckpt_fn(params)
        >>> # Save checkpoint
        >>> replicated = gather_fn(params)  # Gather shards
        >>> np.save('checkpoint.npy', replicated)
        >>> # Load checkpoint
        >>> replicated = np.load('checkpoint.npy')
        >>> params = shard_fn(replicated)  # Distribute to shards
    """
    mesh = params[list(params.keys())[0]].sharding.mesh
    mirrored = jax.sharding.NamedSharding(mesh, P())
    struct = lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s)
    keys = params.keys()
    original = {k: params[k].sharding for k in keys}
    inspec = {k: struct(params[k], original[k]) for k in keys}
    gather_fn = jax.jit(lambda x: x, (original,), mirrored).lower(inspec)
    inspec = {k: struct(params[k], mirrored) for k in keys}
    shard_fn = jax.jit(lambda x: x, (mirrored,), original).lower(inspec)
    if compile:
        gather_fn = gather_fn.compile()  # type: ignore[assignment]
        shard_fn = shard_fn.compile()  # type: ignore[assignment]
    return gather_fn, shard_fn


# def node_mesh(mesh, mp_dims=('t',)):
#   n_mp = math.prod(mesh.shape[d] for d in mp_dims)
#   n_local = mesh.local_mesh.size
#   n_mp_nodes = max(1, n_mp // n_local)
#   total_nodes = mesh.size // n_local
#   n_data_nodes = total_nodes // n_mp_nodes
#   assert n_data_nodes * n_mp_nodes == total_nodes
#   data_node_rank, model_node_rank = divmod(jax.process_index(), n_mp_nodes)
#   data_node_size, model_node_size = n_data_nodes, n_mp_nodes
#   return {
#       'data_node_rank': data_node_rank,
#       'data_node_size': data_node_size,
#       'model_node_rank': model_node_rank,
#       'model_node_size': model_node_size,
#   }
