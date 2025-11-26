"""Experience replay buffer for reinforcement learning.

This module provides the Replay class for storing and sampling experience
sequences. It supports both in-memory and disk-backed storage with
configurable sampling strategies.

Key features:
- Chunked storage for efficient memory usage
- Multiple sampling strategies via selectors (Uniform, Prioritized, etc.)
- Online queue mode for prioritizing recent experience
- Asynchronous disk persistence
- Thread-safe operations

Example:
    >>> replay = Replay(length=64, capacity=100000, directory="/tmp/replay")
    >>> replay.add({"obs": obs, "action": action, "reward": reward}, worker=0)
    >>> batch = replay.sample(batch=32)
    >>> replay.save()  # Persist to disk
"""

import threading
import typing
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial as bind

import elements
import numpy as np

from . import chunk as chunklib
from . import limiters, selectors


class Replay:
    """Experience replay buffer with chunked storage and flexible sampling.

    Stores experience sequences and provides efficient sampling for training.
    Supports various sampling strategies, prioritized replay, and disk persistence.

    Experiences are stored in chunks and organized into fixed-length sequences.
    The buffer maintains multiple streams (one per worker) to handle parallel
    data collection without interleaving sequences from different episodes.

    Attributes:
        length: Fixed length of sampled sequences.
        capacity: Maximum number of items (sequences) in the buffer.
        chunksize: Number of steps per storage chunk.
        name: Identifier for logging and metrics.
        sampler: Selector instance for choosing items to sample.
        online: Whether to prioritize recent experience in a queue.
        directory: Path for disk persistence (None for memory-only).
        metrics: Dict tracking samples, inserts, and updates.

    Example:
        >>> replay = Replay(length=64, capacity=100000)
        >>> for step in episode:
        ...     replay.add(step, worker=0)
        >>> for batch in replay.dataset(batch=32):
        ...     train(batch)
    """

    def __init__(
        self,
        length,
        capacity=None,
        directory=None,
        chunksize=1024,
        online=False,
        selector=None,
        save_wait=False,
        name="unnamed",
        seed=0,
    ):
        """Initialize the replay buffer.

        Args:
            length: Length of sequences to sample (number of timesteps).
            capacity: Maximum number of items. None for unlimited.
            directory: Path for disk persistence. None for memory-only.
            chunksize: Number of timesteps per storage chunk.
            online: If True, maintain a queue of recent experience.
            selector: Sampling strategy (default: Uniform). Can be Prioritized, etc.
            save_wait: If True, block on save() until writes complete.
            name: Identifier for logging and error messages.
            seed: Random seed for the selector.
        """
        self.length = length
        self.capacity = capacity
        self.chunksize = chunksize
        self.name = name

        self.sampler = selector or selectors.Uniform(seed)

        self.chunks = {}
        self.refs = {}
        self.refs_lock = threading.RLock()

        self.items = {}
        self.fifo: deque[typing.Any] = deque()
        self.itemid = 0

        self.current = {}
        self.streams = defaultdict(deque)
        self.rwlock = elements.RWLock()

        self.online = online
        if online:
            self.lengths = defaultdict(int)
            self.queue: deque[typing.Any] = deque()

        if directory:
            self.directory = elements.Path(directory)
            self.directory.mkdir()
            self.workers = ThreadPoolExecutor(16, "replay_saver")
            self.saved = set()
        else:
            self.directory = None
        self.save_wait = save_wait

        self.metrics = {"samples": 0, "inserts": 0, "updates": 0}

    def __len__(self):
        """Return the number of items (sequences) currently in the buffer."""
        return len(self.items)

    def stats(self):
        """Compute and return buffer statistics, resetting metrics.

        Returns:
            Dict with keys: items, chunks, streams, ram_gb, inserts, samples,
            updates, replay_ratio. Metrics are reset to zero after calling.
        """
        ratio = lambda x, y: x / y if y else np.nan
        m = self.metrics
        chunk_nbytes = sum(x.nbytes for x in list(self.chunks.values()))
        stats = {
            "items": len(self.items),
            "chunks": len(self.chunks),
            "streams": len(self.streams),
            "ram_gb": chunk_nbytes / (1024**3),
            "inserts": m["inserts"],
            "samples": m["samples"],
            "updates": m["updates"],
            "replay_ratio": ratio(self.length * m["samples"], m["inserts"]),
        }
        for key in self.metrics:
            self.metrics[key] = 0
        return stats

    @elements.timer.section("replay_add")
    def add(self, step, worker=0):
        """Add a single timestep to the buffer.

        Accumulates steps into sequences of self.length. Once enough steps
        are accumulated, the sequence becomes available for sampling.

        Args:
            step: Dict of arrays for this timestep (obs, action, reward, etc.).
                Keys starting with "log/" are filtered out.
            worker: Worker ID for multi-worker collection. Each worker maintains
                a separate stream to avoid interleaving episodes.
        """
        step = {k: v for k, v in step.items() if not k.startswith("log/")}
        with self.rwlock.reading:
            step = {k: np.asarray(v) for k, v in step.items()}

            if worker not in self.current:
                chunk = chunklib.Chunk(self.chunksize)
                with self.refs_lock:
                    self.refs[chunk.uuid] = 1
                self.chunks[chunk.uuid] = chunk
                self.current[worker] = (chunk.uuid, 0)

            chunkid, index = self.current[worker]
            step["stepid"] = np.frombuffer(
                bytes(chunkid) + index.to_bytes(4, "big"), np.uint8
            )
            stream = self.streams[worker]
            chunk = self.chunks[chunkid]
            assert chunk.length == index, (chunk.length, index)
            chunk.append(step)
            assert chunk.length == index + 1, (chunk.length, index + 1)
            stream.append((chunkid, index))
            with self.refs_lock:
                self.refs[chunkid] += 1

            index += 1
            if index < chunk.size:
                self.current[worker] = (chunkid, index)
            else:
                self._complete(chunk, worker)
            assert len(self.streams) == len(self.current)

            if len(stream) >= self.length:
                # Increment is not thread safe thus inaccurate but faster than locking.
                self.metrics["inserts"] += 1
                chunkid, index = stream.popleft()
                self._insert(chunkid, index)

                if self.online and self.lengths[worker] % self.length == 0:
                    self.queue.append((chunkid, index))

            if self.online:
                self.lengths[worker] += 1

    @elements.timer.section("replay_sample")
    def sample(self, batch, mode="train"):
        """Sample a batch of sequences from the buffer.

        Args:
            batch: Number of sequences to sample.
            mode: One of "train", "report", or "eval". Affects metrics tracking.

        Returns:
            Dict of batched arrays with shape [batch, length, ...].
        """
        message = f"Replay buffer {self.name} is empty"
        limiters.wait(lambda: len(self.sampler), message)
        seqs, is_online = zip(*[self._sample(mode) for _ in range(batch)])
        data = self._assemble_batch(seqs, 0, self.length)
        data = self._annotate_batch(data, is_online, True)
        return data

    @elements.timer.section("replay_update")
    def update(self, data):
        """Update stored sequences with new data (e.g., computed values).

        Used to write back computed values like priorities or value targets
        to the replay buffer. Also updates sampling priorities if the selector
        supports prioritized replay.

        Args:
            data: Dict containing "stepid" for identifying sequences,
                optional "priority" for updating sampling weights,
                and any additional data to store back in the buffer.
        """
        stepid = data.pop("stepid")
        priority = data.pop("priority", None)
        assert stepid.ndim == 3, stepid.shape
        self.metrics["updates"] += int(np.prod(stepid.shape[:-1]))
        if priority is not None:
            assert priority.ndim == 2, priority.shape
            if hasattr(self.sampler, "prioritize"):
                self.sampler.prioritize(
                    stepid.reshape((-1, stepid.shape[-1])), priority.flatten()
                )
        if data:
            for i, stepid in enumerate(stepid):
                stepid = stepid[0].tobytes()
                chunkid = elements.UUID(stepid[:-4])
                index = int.from_bytes(stepid[-4:], "big")
                values = {k: v[i] for k, v in data.items()}
                try:
                    self._setseq(chunkid, index, values)
                except KeyError:
                    pass

    def _sample(self, mode):
        assert mode in ("train", "report", "eval"), mode
        if mode == "train":
            # Increment is not thread safe thus inaccurate but faster than locking.
            self.metrics["samples"] += 1
        while True:
            try:
                if self.online and self.queue and mode == "train":
                    chunkid, index = self.queue.popleft()
                    is_online = True
                else:
                    with elements.timer.section("sample"):
                        itemid = self.sampler()
                    chunkid, index = self.items[itemid]
                    is_online = False
                seq = self._getseq(chunkid, index, concat=False)
                return seq, is_online
            except KeyError:
                continue

    def _insert(self, chunkid, index):
        while self.capacity and len(self.items) >= self.capacity:
            self._remove()
        itemid = self.itemid
        self.itemid += 1
        self.items[itemid] = (chunkid, index)
        stepids = self._getseq(chunkid, index, ["stepid"])["stepid"]
        self.sampler[itemid] = stepids
        self.fifo.append(itemid)

    def _remove(self):
        itemid = self.fifo.popleft()
        del self.sampler[itemid]
        chunkid, _index = self.items.pop(itemid)
        with self.refs_lock:
            self.refs[chunkid] -= 1
            if self.refs[chunkid] < 1:
                del self.refs[chunkid]
                chunk = self.chunks.pop(chunkid)
                if chunk.succ in self.refs:
                    self.refs[chunk.succ] -= 1

    def _getseq(self, chunkid, index, keys=None, concat=True):
        chunk = self.chunks[chunkid]
        available = chunk.length - index
        if available >= self.length:
            with elements.timer.section("get_slice"):
                seq = chunk.slice(index, self.length)
                if not concat:
                    seq = {k: [v] for k, v in seq.items()}
                return seq
        else:
            with elements.timer.section("get_compose"):
                parts = [chunk.slice(index, available)]
                remaining = self.length - available
                while remaining > 0:
                    chunk = self.chunks[chunk.succ]
                    used = min(remaining, chunk.length)
                    parts.append(chunk.slice(0, used))
                    remaining -= used
                seq = {k: [p[k] for p in parts] for k in keys or parts[0].keys()}
                if concat:
                    seq = {k: np.concatenate(v, 0) for k, v in seq.items()}
                return seq

    def _setseq(self, chunkid, index, values):
        length = len(next(iter(values.values())))
        chunk = self.chunks[chunkid]
        available = chunk.length - index
        if available >= length:
            with elements.timer.section("set_slice"):
                return chunk.update(index, length, values)
        else:
            with elements.timer.section("set_compose"):
                part = {k: v[:available] for k, v in values.items()}
                values = {k: v[available:] for k, v in values.items()}
                chunk.update(index, available, part)
                remaining = length - available
                while remaining > 0:
                    chunk = self.chunks[chunk.succ]
                    used = min(remaining, chunk.length)
                    part = {k: v[:used] for k, v in values.items()}
                    values = {k: v[used:] for k, v in values.items()}
                    chunk.update(0, used, part)
                    remaining -= used

    def dataset(self, batch, length=None, consec=None, prefix=0, report=False):
        """Generate an infinite stream of batched data.

        Yields batches continuously, splitting long sequences into consecutive
        chunks if consec > 1. Useful for training loops that need streaming data.

        Args:
            batch: Number of sequences per batch.
            length: Length of each yielded chunk (default: self.length).
            consec: Number of consecutive chunks per sampled sequence.
            prefix: Additional timesteps at the start of each chunk.
            report: If True, use "report" mode for sampling (no metrics).

        Yields:
            Dict of batched arrays with shape [batch, length+prefix, ...].
        """
        length = length or self.length
        consec = consec or (self.length - prefix) // length
        assert consec <= (self.length - prefix) // length, (
            self.length,
            length,
            consec,
            prefix,
        )
        limiters.wait(lambda: len(self.sampler), "Replay buffer is empty")
        # For performance, each batch should be consecutive in memory, rather than
        # a non-consecutive view into a longer batch. For example, this allows
        # near-instant serialization when sending over the network.
        mode = "report" if report else "train"
        while True:
            seqs, is_online = zip(*[self._sample(mode) for _ in range(batch)])
            for i in range(consec):
                offset = i * length
                data = self._assemble_batch(seqs, offset, offset + length + prefix)
                data = self._annotate_batch(data, is_online, is_first=(i == 0))
                yield data

    @elements.timer.section("assemble_batch")
    def _assemble_batch(self, seqs, start, stop):
        shape = (len(seqs), stop - start)
        data = {
            key: np.empty((*shape, *parts[0].shape[1:]), parts[0].dtype)
            for key, parts in seqs[0].items()
        }
        for n, seq in enumerate(seqs):
            st, dt = 0, 0  # Source and destination time index.
            for p in range(len(seq["stepid"])):
                partlen = len(seq["stepid"][p])
                if start < st + partlen:
                    part_start = max(0, start - st)
                    part_stop = min(stop - st, partlen)
                    num = part_stop - part_start
                    for k in data:
                        data[k][n, dt : dt + num] = seq[k][p][part_start:part_stop]
                    dt += num
                st += partlen
                if st >= stop:
                    break
        return data

    @elements.timer.section("annotate_batch")
    def _annotate_batch(self, data, is_online, is_first):
        data = data.copy()
        # if self.online:
        #   broadcasted = [[x] for x in is_online]
        #   data['is_online'] = np.full(data['is_first'].shape, broadcasted, bool)
        if "is_first" in data:
            if is_first:
                data["is_first"] = data["is_first"].copy()
                data["is_first"][:, 0] = True
            if "is_last" in data:
                # Make sure that abandoned episodes have is_last set.
                next_is_first = np.roll(data["is_first"], shift=-1, axis=1)
                next_is_first[:, -1] = False
                data["is_last"] = data["is_last"] | next_is_first
        return data

    @elements.timer.section("replay_save")
    def save(self):
        """Persist all unsaved chunks to disk.

        Completes any in-progress chunks, then saves all chunks that haven't
        been saved yet. Saving is asynchronous unless save_wait=True.

        Returns:
            None
        """
        if self.directory:
            with self.rwlock.writing:
                for worker, (chunkid, _) in self.current.items():
                    chunk = self.chunks[chunkid]
                    if chunk.length > 0:
                        self._complete(chunk, worker)
                promises = []
                for chunk in self.chunks.values():
                    if chunk.length > 0 and chunk.uuid not in self.saved:
                        self.saved.add(chunk.uuid)
                        promises.append(self.workers.submit(chunk.save, self.directory))
                if self.save_wait:
                    [promise.result() for promise in promises]
        return None

    @elements.timer.section("replay_load")
    def load(self, data=None, directory=None, amount=None):
        """Load chunks from disk into memory.

        Loads previously saved chunks from the specified directory, up to
        the specified amount. Handles corrupted chunks gracefully.

        Args:
            data: Unused, kept for API compatibility.
            directory: Path to load from (default: self.directory).
            amount: Maximum number of items to load (default: self.capacity).
        """
        directory = directory or self.directory
        amount = amount or self.capacity or np.inf
        if not directory:
            return
        revsorted = lambda x: list(reversed(sorted(list(x))))
        directory = elements.Path(directory)
        names_loaded = revsorted(x.filename for x in list(self.chunks.values()))
        names_ondisk = revsorted(x.name for x in directory.glob("*.npz"))
        names_ondisk = [x for x in names_ondisk if x not in names_loaded]
        if not names_ondisk:
            return

        numitems = self._numitems(names_loaded + names_ondisk)
        uuids = [elements.UUID(x.split("-")[1]) for x in names_ondisk]
        total = 0
        numchunks = 0
        for uuid in uuids:
            numchunks += 1
            total += numitems[uuid]
            if total >= amount:
                break

        load = bind(chunklib.Chunk.load, error="none")
        filenames = [directory / x for x in names_ondisk[:numchunks]]

        with ThreadPoolExecutor(16, "replay_loader") as pool:
            chunks = [x for x in pool.map(load, filenames) if x]

        # We need to recompute the number of items per chunk now because some
        # chunks may be corrupted and thus not available.
        # numitems = self._numitems(chunks + list(self.chunks.values()))
        numitems = self._numitems(chunks)

        with self.rwlock.writing:
            self.saved.update([chunk.uuid for chunk in chunks])
            with self.refs_lock:
                for chunk in chunks:
                    self.chunks[chunk.uuid] = chunk
                    self.refs[chunk.uuid] = 0
                for chunk in reversed(chunks):
                    amount = numitems[chunk.uuid]
                    self.refs[chunk.uuid] += amount
                    if chunk.succ in self.refs:
                        self.refs[chunk.succ] += 1
                    for index in range(amount):
                        self._insert(chunk.uuid, index)

    @elements.timer.section("complete_chunk")
    def _complete(self, chunk, worker):
        succ = chunklib.Chunk(self.chunksize)
        with self.refs_lock:
            self.refs[chunk.uuid] -= 1
            self.refs[succ.uuid] = 2
        self.chunks[succ.uuid] = succ
        self.current[worker] = (succ.uuid, 0)
        chunk.succ = succ.uuid
        return succ

    def _numitems(self, chunks):
        chunks = [x.filename if hasattr(x, "filename") else x for x in chunks]
        if not chunks:
            return 0
        chunks = list(reversed(sorted([elements.Path(x).stem for x in chunks])))
        _times, uuids, succs, lengths = zip(*[x.split("-") for x in chunks])
        uuids_list = [elements.UUID(x) for x in uuids]
        succs_list = [elements.UUID(x) for x in succs]
        lengths_dict = {k: int(v) for k, v in zip(uuids_list, lengths)}
        future: dict[typing.Any, int] = {}
        for uuid, succ in zip(uuids_list, succs_list):
            future[uuid] = lengths_dict[uuid] + future.get(succ, 0)
        numitems = {}
        for uuid, succ in zip(uuids_list, succs_list):
            numitems[uuid] = lengths_dict[uuid] + 1 - self.length + future.get(succ, 0)
        numitems = {k: np.clip(v, 0, lengths_dict[k]) for k, v in numitems.items()}
        return numitems

    def _notempty(self, reason=False):
        if reason:
            return (True, "ok") if len(self.sampler) else (False, "empty buffer")
        else:
            return bool(len(self.sampler))
