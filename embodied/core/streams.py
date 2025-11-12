"""Stream abstractions for composable data pipelines.

This module provides composable stream components for building data pipelines
in reinforcement learning workflows. Streams implement an iterator interface
with save/load support for checkpointing, enabling resumable data processing.

Key components:
- Stateless: Wraps callables or iterators without checkpoint state
- Prefetch: Asynchronous data prefetching with background worker thread
- Consec: Creates consecutive overlapping chunks from sequence data
- Zip: Combines multiple streams by concatenating outputs
- Map: Applies transformation functions to stream elements
- Mixer: Weighted probabilistic mixing of multiple data sources

All streams inherit from base.Stream and support:
- Iterator protocol (__iter__, __next__)
- Checkpointing (save/load methods for training resumption)
- Composition (streams can wrap other streams)

Typical usage:
    # Create a prefetched, transformed stream
    stream = Prefetch(
        source=Map(replay_buffer, transform_fn),
        amount=2
    )

    # Use in training loop
    for batch in stream:
        train_step(batch)

    # Save/restore state for checkpointing
    state = stream.save()
    stream.load(state)
"""

import functools
import queue
import threading

import elements
import numpy as np
import portal

from . import base


class Stateless(base.Stream):
    """Wrapper for stateless callables or iterators without checkpoint state.

    Converts a callable or iterator into a Stream that doesn't maintain any
    internal state for checkpointing. Useful for wrapping data sources that
    are inherently stateless or where state management is handled externally.

    The save/load methods intentionally do nothing, making this suitable for:
    - Pure functions that compute outputs deterministically
    - External data sources that handle their own state
    - Iterators where resumption isn't needed or possible

    Args:
        nextfn: Callable that returns the next element, or an iterator object.
            If an iterator is provided, its __next__ method will be used.
        *args: Positional arguments to pass to nextfn on each call.
        **kwargs: Keyword arguments to pass to nextfn on each call.

    Example:
        # Wrap a random data generator
        def random_batch():
            return np.random.randn(32, 64)

        stream = Stateless(random_batch)
        for batch in stream:
            process(batch)

        # Wrap an existing iterator
        stream = Stateless(iter(range(100)))
    """

    def __init__(self, nextfn, *args, **kwargs):
        if not callable(nextfn) and hasattr(nextfn, "__next__"):
            nextfn = nextfn.__next__
        self.nextfn = functools.partial(nextfn, *args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        """Generate the next element by calling the wrapped function.

        Returns:
            Output from the wrapped nextfn callable.

        Raises:
            Any exception raised by the wrapped nextfn.
        """
        return self.nextfn()

    def save(self):
        """Return checkpoint state (always None for stateless streams).

        Returns:
            None, indicating no state to save.
        """
        return None

    def load(self, data):
        """Load checkpoint state (no-op for stateless streams).

        Args:
            data: Ignored checkpoint data.
        """
        pass


class Prefetch(base.Stream):
    """Asynchronous data prefetching with a background worker thread.

    Prefetches data elements from a source stream in a separate thread, buffering
    them in a queue for immediate consumption. This overlaps data loading/processing
    with training computation, reducing idle time and improving throughput.

    The worker thread continuously fetches from the source, applies an optional
    transformation, and queues results. The main thread consumes from the queue
    without blocking on I/O or computation. A semaphore limits the prefetch buffer
    size to prevent excessive memory usage.

    Checkpointing is carefully handled to ensure consistency:
    - State is captured AFTER each successful fetch
    - On load, prefetched items are discarded and buffer is refilled
    - This ensures no data is lost or duplicated across checkpoint boundaries

    Args:
        source: Source stream or iterable to prefetch from. If the source has
            __iter__, it will be called to create an iterator.
        transform: Optional callable to apply to each fetched element before
            queueing. Defaults to identity function. Useful for CPU preprocessing
            that can run in parallel with GPU training.
        amount: Maximum number of elements to prefetch ahead. Default 1.
            Higher values increase throughput but use more memory. Typical: 1-4.

    Example:
        # Prefetch with CPU preprocessing
        def preprocess(batch):
            return {k: augment(v) for k, v in batch.items()}

        stream = Prefetch(replay_buffer, transform=preprocess, amount=2)

        # Worker thread fetches and preprocesses while training runs
        for batch in stream:
            train_step(batch)  # Batch is already preprocessed and ready

    Attributes:
        source: The wrapped source iterator.
        transform: Transformation function applied to each element.
        amount: Prefetch buffer size.
        state: Most recent checkpoint state from source.
        started: Whether the worker thread has been started.
    """

    def __init__(self, source, transform=None, amount=1):
        self.source = iter(source) if hasattr(source, "__iter__") else source()
        self.transform = transform or (lambda x: x)
        self.state = self._getstate()
        self.requests = threading.Semaphore(amount)
        self.amount = amount
        self.queue = queue.Queue()
        self.worker = portal.Thread(self._worker)
        self.started = False

    def __iter__(self):
        """Start the worker thread and return iterator.

        Returns:
            Self as an iterator.

        Raises:
            AssertionError: If called more than once (worker already started).
        """
        assert not self.started
        self.worker.start()
        self.started = True
        return self

    def __next__(self):
        """Retrieve the next prefetched element from the queue.

        Blocks until an element is available in the prefetch queue. Releases
        a semaphore permit to allow the worker to fetch another element.

        Returns:
            The next data element from the source stream (after transformation).

        Raises:
            RuntimeError: If the worker thread encountered an exception.
            AssertionError: If called before __iter__.
        """
        assert self.started
        result = self.queue.get()
        self.requests.release()
        if isinstance(result, str):
            raise RuntimeError(result)
        data, self.state = result
        return data

    def save(self):
        """Return checkpoint state of the source stream.

        Returns:
            Checkpoint state from the last successfully fetched element.
            This ensures consistency across checkpoint boundaries.
        """
        return self.state

    def load(self, state):
        """Restore checkpoint state and reset prefetch queue.

        Discards any prefetched elements in the queue and restores the source
        stream to the given state. If the worker is running, refills the
        prefetch buffer after loading.

        Args:
            state: Checkpoint state to restore to the source stream.
        """
        if self.started:
            for _ in range(self.amount):
                self.queue.get()
        self.source.load(state)
        if self.started:
            self.requests.release(self.amount)

    def _worker(self):
        """Worker thread loop that fetches and transforms data.

        Continuously acquires permits, fetches from source, transforms data,
        captures state, and enqueues results. Runs until an exception occurs
        or the thread is stopped.

        Exceptions are converted to strings and queued to be re-raised in the
        main thread, ensuring failures are visible to the caller.
        """
        try:
            while True:
                self.requests.acquire()
                data = next(self.source)
                data = self.transform(data)
                state = self._getstate()
                self.queue.put((data, state))
        except Exception as e:
            self.queue.put(str(e))
            raise

    def _getstate(self):
        """Retrieve checkpoint state from source if available.

        Returns:
            Source checkpoint state if source has save() method, else None.
        """
        if hasattr(self.source, "save"):
            return self.source.save()
        else:
            return None


class Consec(base.Stream):
    """Creates consecutive overlapping chunks from sequence data.

    Splits sequence batches into multiple consecutive chunks with optional
    prefix context. This is essential for training recurrent models where each
    chunk needs historical context (prefix) to establish the hidden state before
    the main training sequence begins.

    The stream reads a full sequence from the source, then yields `consec`
    overlapping chunks from it. Each chunk contains `length` main timesteps plus
    `prefix` context timesteps. This allows efficient use of long sequences
    collected from environments while providing necessary context for recurrence.

    Chunk generation pattern (length=3, consec=3, prefix=2):
        Source sequence: 0 1 2 3 4 5 6 7 8 9 10
        Chunk 1: [0 1 | 2 3 4]     (prefix: 0-1, main: 2-4)
        Chunk 2:     [2 3 | 4 5 6] (prefix: 2-3, main: 4-6)
        Chunk 3:         [4 5 | 6 7 8] (prefix: 4-5, main: 6-8)

    The prefix timesteps (marked with 'p' or '|') provide burn-in for recurrent
    state, while the main timesteps (marked with '#') are used for training.

    Args:
        source: Source stream yielding sequence batches as dicts with arrays of
            shape [batch, time, ...]. Must include 'is_first' key for validation.
        length: Number of main timesteps in each chunk (excluding prefix).
        consec: Number of consecutive chunks to extract from each source sequence.
        prefix: Number of prefix (context) timesteps prepended to each chunk.
            Default 0. These provide recurrent state burn-in.
        strict: If True, requires that source sequences have exactly
            consec * length + prefix timesteps. If False, allows longer sequences.
            Default True.
        contiguous: If True, converts output arrays to C-contiguous layout via
            np.ascontiguousarray. This can speed up downstream operations like
            network transfers but adds copy overhead. Default False.

    Example:
        # Extract 3 overlapping chunks of length 50 with 20-step prefix
        # from sequences of length 170 (3*50 + 20)
        stream = Consec(replay_buffer, length=50, consec=3, prefix=20)

        for chunk in stream:
            # chunk['consec'] indicates which chunk (0, 1, 2)
            # chunk['is_first'] shape: [batch, 70] (50 + 20)
            train_step(chunk)

    Attributes:
        source: The wrapped source stream.
        length: Main chunk length.
        consec: Number of chunks per sequence.
        prefix: Prefix length for context.
        index: Current chunk index within current sequence (0 to consec-1).
        current: Currently buffered source sequence being chunked.
    """

    def __init__(self, source, length, consec, prefix=0, strict=True, contiguous=False):
        self.source = source
        self.length = length
        self.consec = consec
        self.prefix = prefix
        self.strict = strict
        self.contiguous = contiguous
        self.index = 0
        self.current = None
        self.it = None

    def __iter__(self):
        """Initialize iterator from source.

        Returns:
            Self as an iterator.
        """
        self.it = iter(self.source)
        return self

    def __next__(self):
        """Generate the next consecutive chunk from current or new sequence.

        Fetches a new sequence from source when index reaches consec, then
        extracts overlapping chunks with stride=length. Each chunk includes
        prefix context timesteps followed by main training timesteps.

        Returns:
            Dict of arrays with shape [batch, length+prefix, ...]. Includes
            a 'consec' key indicating chunk index (0 to consec-1) for tracking.

        Raises:
            AssertionError: If source sequence is too short for the requested
                chunk configuration (requires >= consec*length + prefix timesteps).
            AssertionError: If strict=True and sequence length doesn't exactly
                match consec*length + prefix.
        """
        if self.index >= self.consec:
            self.index = 0
        if self.index == 0:
            self.current = next(self.it)
            available = self.current["is_first"].shape[-1]
            assert self.length * self.consec + self.prefix <= available, (
                self.length,
                self.consec,
                self.prefix,
                available,
            )
            if self.strict:
                assert self.consec * self.length + self.prefix == available, (
                    self.consec,
                    self.length,
                    self.prefix,
                    available,
                )
        start = self.index * self.length
        stop = start + (self.length + self.prefix)
        chunk = {k: v[:, start:stop] for k, v in self.current.items()}
        chunk["consec"] = np.full(chunk["is_first"].shape, self.index, np.int32)
        if self.contiguous:
            # This is expensive but can speed up following operations, such as
            # sending arrays via networking.
            chunk = {k: np.ascontiguousarray(v) for k, v in chunk.items()}
        self.index += 1
        return chunk

    def save(self):
        """Return checkpoint state including source state and chunk index.

        Returns:
            Dict with 'source' (source checkpoint) and 'index' (current chunk
            position within sequence). This ensures chunking resumes correctly.
        """
        return {
            "source": self.source.save(),
            "index": self.index,
        }

    def load(self, data):
        """Restore checkpoint state and chunk index.

        Args:
            data: Dict containing 'source' state and 'index' position.
        """
        self.source.load(data["source"])
        self.index = data["index"]


class Zip(base.Stream):
    """Combines multiple source streams by concatenating their outputs.

    Synchronously pulls from all source streams and concatenates corresponding
    arrays along the batch dimension. This is useful for combining data from
    multiple replay buffers, environments, or data sources to create larger
    training batches or mix different types of data.

    All sources must yield compatible nested structures (matching keys and shapes
    except for the concatenation dimension). The streams are stepped in lockstep,
    so they should produce elements at similar rates.

    Args:
        sources: List or tuple of source streams to zip together. Must have at
            least 2 sources. Each source should yield dicts of arrays.

    Example:
        # Combine batches from two replay buffers
        buffer1 = ReplayBuffer(capacity=100000)
        buffer2 = ReplayBuffer(capacity=100000)
        stream = Zip([buffer1, buffer2])

        for batch in stream:
            # batch contains concatenated data from both buffers
            # If buffer1 yields batch_size=16 and buffer2 yields batch_size=16,
            # result has batch_size=32
            train_step(batch)

    Attributes:
        sources: List of source streams to combine.
        iterators: List of active iterators from sources (after __iter__ called).
        started: Whether iteration has begun.
    """

    def __init__(self, sources):
        assert len(sources) > 1, len(sources)
        self.sources = sources
        self.iterators = None
        self.started = False

    def __iter__(self):
        """Initialize iterators for all sources.

        Returns:
            Self as an iterator.

        Raises:
            AssertionError: If called more than once (already started).
        """
        assert not self.started
        self.started = True
        self.iterators = [iter(x) for x in self.sources]
        return self

    def __next__(self):
        """Fetch from all sources and concatenate results.

        Returns:
            Nested dict/tuple structure with arrays concatenated along axis 0.
            Structure matches the sources' output structure.

        Raises:
            StopIteration: If any source is exhausted.
        """
        parts = [next(x) for x in self.iterators]
        result = elements.tree.map(lambda *el: np.concatenate(el), *parts)
        return result

    def save(self):
        """Return checkpoint states from all sources.

        Returns:
            List of checkpoint states, one per source iterator.
        """
        return [x.save() for x in self.iterators]

    def load(self, data):
        """Restore checkpoint states to all sources.

        Args:
            data: List of checkpoint states matching the sources.

        Raises:
            AssertionError: If data length doesn't match number of sources.
        """
        assert len(data) == len(self.iterators)
        [it.load(d) for it, d in zip(self.iterators, data)]


class Map(base.Stream):
    """Applies a transformation function to each element from a source stream.

    Wraps a source stream and applies a transformation function to every element
    yielded by the source. This is the primary mechanism for data preprocessing,
    augmentation, and transformation in stream pipelines.

    The transformation function can modify data shape, type, or structure. Common
    uses include data augmentation, normalization, filtering, type conversion, or
    adding derived fields.

    Args:
        source: Source stream to read from.
        fn: Transformation function taking the source element as first argument
            and returning the transformed element. Signature: fn(element, *args, **kwargs).
        *args: Additional positional arguments to pass to fn.
        **kwargs: Additional keyword arguments to pass to fn.

    Example:
        # Normalize and augment images
        def preprocess(batch, mean, std):
            batch['image'] = (batch['image'] - mean) / std
            batch['image'] = random_crop(batch['image'], size=(64, 64))
            return batch

        stream = Map(replay_buffer, preprocess, mean=0.5, std=0.25)

        for batch in stream:
            # batch contains normalized and augmented images
            train_step(batch)

        # Chain multiple transformations
        stream = Map(Map(source, normalize), augment)

    Attributes:
        source: The wrapped source stream.
        fn: Partial function combining transformation and arguments.
        iterator: Active iterator from source (after __iter__ called).
        started: Whether iteration has begun.
    """

    def __init__(self, source, fn, *args, **kwargs):
        self.source = source
        self.fn = lambda x: fn(x, *args, **kwargs)
        self.iterator = None
        self.started = False

    def __iter__(self):
        """Initialize iterator from source.

        Returns:
            Self as an iterator.

        Raises:
            AssertionError: If called more than once (already started).
        """
        assert not self.started
        self.started = True
        self.iterator = iter(self.source)
        return self

    def __next__(self):
        """Fetch next element from source and apply transformation.

        Returns:
            Transformed element after applying fn to source output.

        Raises:
            AssertionError: If called before __iter__.
            Any exception raised by the transformation function.
        """
        assert self.started
        return self.fn(next(self.iterator))

    def save(self):
        """Return checkpoint state from source iterator.

        Returns:
            Checkpoint state from the underlying source iterator.
        """
        return self.iterator.save()

    def load(self, data):
        """Restore checkpoint state to source iterator.

        Args:
            data: Checkpoint state to restore to source iterator.
        """
        self.iterator.load(data)


class Mixer(base.Stream):
    """Weighted probabilistic mixing of multiple data sources.

    Randomly selects one of several source streams for each element according
    to specified probability weights. This enables training on mixed datasets
    or replay buffers with controlled sampling ratios.

    The selection is deterministic given the seed and step counter, ensuring
    reproducibility across checkpoint restores. Each source maintains its own
    state independently, and only the selected source is advanced on each step.

    This is useful for:
    - Multi-task learning with different sampling ratios per task
    - Curriculum learning with time-varying mixture weights
    - Balancing data from multiple replay buffers
    - Combining online and offline data sources

    Args:
        sources: Dict mapping source names (str) to source streams. Keys are
            used for identification and checkpoint management.
        weights: Dict mapping source names to unnormalized weights (float).
            Must have same keys as sources. Weights are normalized to sum to 1.
            Higher weight = higher probability of selection.
        seed: Random seed for reproducible source selection. Default 0.

    Example:
        # Mix 75% task A, 25% task B
        sources = {
            'task_a': replay_buffer_a,
            'task_b': replay_buffer_b,
        }
        weights = {'task_a': 0.75, 'task_b': 0.25}
        stream = Mixer(sources, weights, seed=42)

        for batch in stream:
            # batch comes from task_a 75% of the time, task_b 25%
            train_step(batch)

        # Checkpointing preserves selection sequence
        state = stream.save()
        stream.load(state)  # Continues same selection pattern

    Attributes:
        keys: Sorted list of source names for deterministic ordering.
        iterators: List of source iterators in key order.
        probs: Normalized probability distribution for source selection.
        seed: Random seed for selection RNG.
        step: Current step counter for RNG seeding.
        started: Whether iteration has begun.
    """

    def __init__(self, sources, weights, seed=0):
        assert sources.keys() == weights.keys(), (sources, weights)
        self.keys = sorted(sources.keys())
        self.iterators = [iter(sources[k]) for k in self.keys]
        weights = np.array([weights[k] for k in self.keys], np.float32)
        self.probs = weights / weights.sum()
        self.seed = seed
        self.started = False
        self.step = 0

    def __iter__(self):
        """Mark iteration as started.

        Returns:
            Self as an iterator.

        Raises:
            AssertionError: If called more than once (already started).
        """
        assert not self.started
        self.started = True
        return self

    def __next__(self):
        """Select a source probabilistically and fetch its next element.

        Uses deterministic random selection based on seed and step counter
        to choose which source to sample from, then advances that source.

        Returns:
            Next element from the randomly selected source stream.

        Raises:
            AssertionError: If called before __iter__.
            StopIteration: If the selected source is exhausted.
        """
        assert self.started
        rng = np.random.default_rng(seed=[self.seed, self.step])
        self.step += 1
        index = rng.choice(len(self.keys), p=self.probs)
        return next(self.iterators[index])

    def save(self):
        """Return checkpoint state including all source states and RNG state.

        Returns:
            Dict containing 'step' (RNG counter), 'seed' (RNG seed), and
            'sources' (dict of per-source checkpoint states keyed by name).
        """
        return {
            "step": self.step,
            "seed": self.seed,
            "sources": {k: it.save() for k, it in zip(self.keys, self.iterators)},
        }

    def load(self, data):
        """Restore checkpoint state including RNG and all source states.

        Args:
            data: Dict containing 'step', 'seed', and 'sources' checkpoint data.

        Raises:
            AssertionError: If source keys don't match current configuration.
        """
        self.step = data["step"]
        self.seed = data["seed"]
        assert sorted(data["sources"].keys()) == self.keys, (data["sources"], self.keys)
        for key, iterator in zip(self.keys, self.iterators):
            iterator.load(data["sources"][key])
