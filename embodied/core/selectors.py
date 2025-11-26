"""Replay buffer selector implementations for sampling strategies.

This module provides various selector classes that determine which sequences
are sampled from the replay buffer during training. Each selector implements
a different sampling strategy:

- Fifo: First-in-first-out queue (oldest sequences sampled first)
- Uniform: Uniform random sampling from all sequences
- Recency: Recency-weighted sampling (newer sequences more likely)
- Prioritized: Priority-based sampling using temporal difference errors
- Mixture: Combines multiple selectors with weighted selection
- SampleTree: Efficient tree structure for weighted sampling

All selectors implement a common interface:
    __call__(): Sample and return a sequence key
    __setitem__(key, stepids): Add a new sequence
    __delitem__(key): Remove a sequence
    __len__(): Return number of sequences

Some selectors (Prioritized, Mixture) also support:
    prioritize(stepids, priorities): Update priorities for specific timesteps
"""

import collections
import threading
import typing

import numpy as np


class Fifo:
    """First-in-first-out selector that samples the oldest sequence.

    Maintains sequences in insertion order using a deque. Always samples
    the oldest (first inserted) sequence. Useful for ensuring all data
    is seen before replay buffer overflow occurs.

    Attributes:
        queue: Deque storing sequence keys in insertion order.
    """

    def __init__(self):
        """Initialize an empty FIFO queue."""
        self.queue: collections.deque[typing.Any] = collections.deque()

    def __call__(self):
        """Sample the oldest sequence key.

        Returns:
            The key of the oldest sequence in the queue.

        Raises:
            IndexError: If queue is empty.
        """
        return self.queue[0]

    def __len__(self):
        """Return the number of sequences in the queue.

        Returns:
            Number of sequences currently stored.
        """
        return len(self.queue)

    def __setitem__(self, key, stepids):
        """Add a new sequence to the end of the queue.

        Args:
            key: Unique identifier for the sequence.
            stepids: List of step IDs in the sequence (not used by FIFO).
        """
        self.queue.append(key)

    def __delitem__(self, key):
        """Remove a sequence from the queue.

        Efficiently removes from the front, but slow for other positions.

        Args:
            key: Sequence key to remove.

        Raises:
            ValueError: If key is not in the queue.
        """
        if self.queue[0] == key:
            self.queue.popleft()
        else:
            # This is very slow but typically not used.
            self.queue.remove(key)


class Uniform:
    """Uniform random selector that samples all sequences with equal probability.

    Thread-safe implementation that maintains O(1) insertion, deletion, and
    sampling operations. Uses swap-with-last deletion strategy to maintain
    constant-time removal.

    Attributes:
        indices: Dictionary mapping keys to their positions in the keys list.
        keys: List of all sequence keys for efficient random access.
        rng: NumPy random number generator for reproducible sampling.
        lock: Threading lock for thread-safe operations.
    """

    def __init__(self, seed=0):
        """Initialize a uniform random selector.

        Args:
            seed: Random seed for reproducible sampling. Defaults to 0.
        """
        self.indices = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()

    def __len__(self):
        """Return the number of sequences available for sampling.

        Returns:
            Number of sequences currently stored.
        """
        return len(self.keys)

    def __call__(self):
        """Sample a random sequence key with uniform probability.

        Thread-safe operation that selects a random key from all available
        sequences with equal probability.

        Returns:
            A randomly selected sequence key.

        Raises:
            ValueError: If no sequences are available for sampling.
        """
        with self.lock:
            index = self.rng.integers(0, len(self.keys)).item()
            return self.keys[index]

    def __setitem__(self, key, stepids):
        """Add a new sequence to the selector.

        Thread-safe insertion that appends the key to the end of the list
        and records its index for O(1) deletion.

        Args:
            key: Unique identifier for the sequence.
            stepids: List of step IDs in the sequence (not used by Uniform).
        """
        with self.lock:
            self.indices[key] = len(self.keys)
            self.keys.append(key)

    def __delitem__(self, key):
        """Remove a sequence from the selector.

        Thread-safe deletion using swap-with-last strategy to maintain O(1)
        time complexity. The deleted key is swapped with the last key in the
        list to avoid shifting all subsequent elements.

        Args:
            key: Sequence key to remove.

        Raises:
            KeyError: If key is not in the selector.
        """
        with self.lock:
            index = self.indices.pop(key)
            last = self.keys.pop()
            # Only swap if there are remaining items after popping
            if index < len(self.keys):
                self.keys[index] = last
                self.indices[last] = index


class Recency:
    """Recency-weighted selector that samples newer sequences more frequently.

    Uses a hierarchical probability tree to efficiently sample sequences based
    on their age. Newer sequences have higher sampling probability according to
    the provided probability distribution. The distribution must be monotonically
    decreasing (most recent has highest probability).

    Attributes:
        uprobs: Unnormalized probability distribution over ages (decreasing).
        tree: Hierarchical probability tree for efficient sampling.
        rng: NumPy random number generator for reproducible sampling.
        step: Current global step counter for tracking insertion order.
        steps: Dictionary mapping keys to their insertion step.
        items: Dictionary mapping insertion steps to keys.
    """

    def __init__(self, uprobs, seed=0):
        """Initialize a recency-weighted selector.

        Args:
            uprobs: Unnormalized probability array where uprobs[0] is the
                probability of sampling the most recent item and uprobs[-1]
                is the probability of sampling the oldest. Must be monotonically
                decreasing (uprobs[0] >= uprobs[-1]).
            seed: Random seed for reproducible sampling. Defaults to 0.

        Raises:
            AssertionError: If uprobs is not monotonically decreasing.
        """
        assert uprobs[0] >= uprobs[-1], uprobs
        self.uprobs = uprobs
        self.tree = self._build(uprobs)
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.steps = {}
        self.items = {}

    def __len__(self):
        """Return the number of sequences available for sampling.

        Returns:
            Number of sequences currently stored.
        """
        return len(self.items)

    def __call__(self):
        """Sample a sequence key with recency-weighted probability.

        Samples an age from the probability distribution, then returns the
        sequence at that age. Retries up to 10 times if a sequence was
        recently deleted.

        Returns:
            A sequence key sampled according to recency weights.

        Raises:
            KeyError: If unable to sample a valid key after 10 retries.
        """
        for retry in range(10):
            try:
                age = self._sample(self.tree, self.rng)
                if len(self.items) < len(self.uprobs):
                    age = int(age / len(self.uprobs) * len(self.items))
                return self.items[self.step - 1 - age]
            except KeyError:
                # Item might have been deleted very recently.
                if retry < 9:
                    import time

                    time.sleep(0.01)
                else:
                    raise

    def __setitem__(self, key, stepids):
        """Add a new sequence to the selector.

        The sequence is assigned the current step counter and becomes the
        most recent item (age 0).

        Args:
            key: Unique identifier for the sequence.
            stepids: List of step IDs in the sequence (not used by Recency).
        """
        self.steps[key] = self.step
        self.items[self.step] = key
        self.step += 1

    def __delitem__(self, key):
        """Remove a sequence from the selector.

        Args:
            key: Sequence key to remove.

        Raises:
            KeyError: If key is not in the selector.
        """
        step = self.steps.pop(key)
        del self.items[step]

    def _sample(self, tree, rng, bfactor=16):
        """Sample an age index from the hierarchical probability tree.

        Traverses the tree from root to leaf, selecting children according
        to their normalized probabilities at each level.

        Args:
            tree: List of probability arrays at each tree level.
            rng: NumPy random number generator.
            bfactor: Branching factor of the tree (default 16).

        Returns:
            Sampled age index in the range [0, len(uprobs)).
        """
        path: list[int] = []
        for level, prob in enumerate(tree):
            p = prob
            for segment in path:
                p = p[segment]
            index = rng.choice(len(segment), p=p)
            path.append(index)
        index = sum(
            index * bfactor ** (len(tree) - level - 1)
            for level, index in enumerate(path)
        )
        return index

    def _build(self, uprobs, bfactor=16):
        """Build a hierarchical probability tree for efficient sampling.

        Constructs a tree where each level aggregates probabilities from the
        level below, normalized within each parent node. Enables O(log N)
        sampling complexity.

        Args:
            uprobs: Unnormalized probability distribution over ages.
            bfactor: Branching factor (children per node). Default 16.

        Returns:
            List of normalized probability arrays, one per tree level.

        Raises:
            AssertionError: If uprobs contains non-finite or negative values.
        """
        assert np.isfinite(uprobs).all(), uprobs
        assert (uprobs >= 0).all(), uprobs
        depth = int(np.ceil(np.log(len(uprobs)) / np.log(bfactor)))
        size = bfactor**depth
        uprobs = np.concatenate([uprobs, np.zeros(size - len(uprobs))])
        tree = [uprobs]
        for level in reversed(range(depth - 1)):
            tree.insert(0, tree[0].reshape((-1, bfactor)).sum(-1))
        for level, prob in enumerate(tree):
            prob = prob.reshape([bfactor] * (1 + level))
            total = prob.sum(-1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                tree[level] = np.where(total, prob / total, prob)
        return tree


class Prioritized:
    """Priority-based selector using temporal difference errors for sampling.

    Samples sequences with probability proportional to their priority scores
    (typically TD errors). Sequences with higher learning value are sampled
    more frequently. Uses a SampleTree for efficient weighted sampling.

    Priorities are assigned per timestep, and sequence priorities are aggregated
    (mean or mix of mean and max). Supports optional exponent for prioritization
    strength and zero-on-sample for one-shot learning.

    Attributes:
        exponent: Exponent applied to priorities (higher = more aggressive).
        initial: Initial priority for new timesteps.
        zero_on_sample: If True, reset priorities to zero after sampling.
        maxfrac: Fraction of max priority in aggregation (0 = mean only).
        tree: SampleTree for efficient weighted sampling.
        prios: Dictionary mapping step IDs to their current priorities.
        stepitems: Dictionary mapping step IDs to containing sequence keys.
        items: Dictionary mapping sequence keys to their step IDs.
    """

    def __init__(
        self,
        exponent=1.0,
        initial=1.0,
        zero_on_sample=False,
        maxfrac=0.0,
        branching=16,
        seed=0,
    ):
        """Initialize a prioritized experience replay selector.

        Args:
            exponent: Exponent applied to priorities before sampling. Values > 1
                increase focus on high-priority sequences. Default 1.0 (linear).
            initial: Initial priority assigned to new timesteps. Default 1.0.
            zero_on_sample: If True, reset priorities to zero after sampling
                to ensure each sequence is sampled at most once. Default False.
            maxfrac: Fraction of max priority in aggregation formula:
                priority = maxfrac * max(prios) + (1 - maxfrac) * mean(prios).
                Use 0.0 for mean-only (default), 1.0 for max-only.
            branching: Branching factor for the internal SampleTree. Default 16.
            seed: Random seed for reproducible sampling. Default 0.

        Raises:
            AssertionError: If maxfrac is not in [0, 1].
        """
        assert 0 <= maxfrac <= 1, maxfrac
        self.exponent = float(exponent)
        self.initial = float(initial)
        self.zero_on_sample = zero_on_sample
        self.maxfrac = maxfrac
        self.tree = SampleTree(branching, seed)
        self.prios = collections.defaultdict(lambda: self.initial)
        self.stepitems = collections.defaultdict(list)
        self.items = {}

    def prioritize(self, stepids, priorities):
        """Update priorities for specific timesteps and their sequences.

        Updates the priority of each timestep and recomputes the aggregated
        priority for all sequences containing those timesteps. This is typically
        called after computing TD errors during training.

        Args:
            stepids: List of timestep identifiers (bytes or arrays).
            priorities: List of priority values corresponding to stepids.
        """
        if not isinstance(stepids[0], bytes):
            stepids = [x.tobytes() for x in stepids]
        for stepid, priority in zip(stepids, priorities):
            try:
                self.prios[stepid] = priority
            except KeyError:
                print("Ignoring priority update for removed time step.")
        items = []
        for stepid in stepids:
            items += self.stepitems[stepid]
        for key in list(set(items)):
            try:
                self.tree.update(key, self._aggregate(key))
            except KeyError:
                print("Ignoring tree update for removed time step.")

    def __len__(self):
        """Return the number of sequences available for sampling.

        Returns:
            Number of sequences currently stored.
        """
        return len(self.items)

    def __call__(self):
        """Sample a sequence key with priority-weighted probability.

        Samples from the tree according to aggregated priorities. If
        zero_on_sample is enabled, resets all priorities in the sampled
        sequence to zero.

        Returns:
            A sequence key sampled according to priority weights.
        """
        key = self.tree.sample()
        if self.zero_on_sample:
            zeros = [0.0] * len(self.items[key])
            self.prioritize(self.items[key], zeros)
        return key

    def __setitem__(self, key, stepids):
        """Add a new sequence to the selector.

        Registers the sequence and its timesteps, assigns initial priorities,
        and inserts into the sampling tree with aggregated priority.

        Args:
            key: Unique identifier for the sequence.
            stepids: List of timestep identifiers in the sequence (bytes or arrays).
        """
        if not isinstance(stepids[0], bytes):
            stepids = [x.tobytes() for x in stepids]
        self.items[key] = stepids
        [self.stepitems[stepid].append(key) for stepid in stepids]
        self.tree.insert(key, self._aggregate(key))

    def __delitem__(self, key):
        """Remove a sequence from the selector.

        Removes the sequence from the tree and cleans up all associated
        timestep priorities and mappings.

        Args:
            key: Sequence key to remove.

        Raises:
            KeyError: If key is not in the selector.
        """
        self.tree.remove(key)
        stepids = self.items.pop(key)
        for stepid in stepids:
            stepitems = self.stepitems[stepid]
            stepitems.remove(key)
            if not stepitems:
                del self.stepitems[stepid]
                del self.prios[stepid]

    def _aggregate(self, key):
        """Compute aggregated priority for a sequence.

        Aggregates per-timestep priorities into a single sequence priority
        using mean or a mix of mean and max. Applies the exponent if specified.

        Args:
            key: Sequence key to aggregate priorities for.

        Returns:
            Aggregated priority value for the sequence.
        """
        # Both list comprehensions in this function are a performance bottleneck
        # because they are called very often.
        prios = [self.prios[stepid] for stepid in self.items[key]]
        if self.exponent != 1.0:
            prios = [x**self.exponent for x in prios]
        mean = sum(prios) / len(prios)
        if self.maxfrac:
            return self.maxfrac * max(prios) + (1 - self.maxfrac) * mean
        else:
            return mean


class Mixture:
    """Mixture selector that combines multiple selectors with weighted sampling.

    Samples from multiple underlying selectors according to specified fractions.
    For each sample, first selects a selector according to the fractions, then
    samples from that selector. Useful for combining different sampling
    strategies (e.g., 80% prioritized + 20% uniform).

    All sequences are registered with all underlying selectors. Priority updates
    are forwarded to selectors that support prioritization.

    Attributes:
        selectors: List of underlying selector instances.
        fractions: NumPy array of sampling probabilities for each selector.
        rng: NumPy random number generator for reproducible selector choice.
    """

    def __init__(self, selectors, fractions, seed=0):
        """Initialize a mixture of selectors.

        Args:
            selectors: Dictionary mapping names to selector instances.
            fractions: Dictionary mapping names to sampling fractions.
                Must sum to 1.0. Selectors with 0 fraction are removed.
            seed: Random seed for reproducible selector choice. Default 0.

        Raises:
            AssertionError: If selector and fraction keys don't match or
                fractions don't sum to 1.0.
        """
        assert set(selectors.keys()) == set(fractions.keys())
        assert sum(fractions.values()) == 1, fractions
        for key, frac in list(fractions.items()):
            if not frac:
                selectors.pop(key)
                fractions.pop(key)
        keys = sorted(selectors.keys())
        self.selectors = [selectors[key] for key in keys]
        self.fractions = np.array([fractions[key] for key in keys], np.float32)
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        """Sample a sequence key using the mixture distribution.

        First randomly selects a selector according to the fractions, then
        samples a key from that selector.

        Returns:
            A sequence key sampled from one of the underlying selectors.
        """
        return self.rng.choice(self.selectors, p=self.fractions)()

    def __setitem__(self, key, stepids):
        """Add a new sequence to all underlying selectors.

        Args:
            key: Unique identifier for the sequence.
            stepids: List of step IDs in the sequence.
        """
        for selector in self.selectors:
            selector[key] = stepids

    def __delitem__(self, key):
        """Remove a sequence from all underlying selectors.

        Args:
            key: Sequence key to remove.
        """
        for selector in self.selectors:
            del selector[key]

    def prioritize(self, stepids, priorities):
        """Update priorities in all selectors that support prioritization.

        Forwards priority updates to underlying selectors that have a
        prioritize() method (e.g., Prioritized selector).

        Args:
            stepids: List of timestep identifiers.
            priorities: List of priority values corresponding to stepids.
        """
        for selector in self.selectors:
            if hasattr(selector, "prioritize"):
                selector.prioritize(stepids, priorities)


class SampleTree:
    """Efficient tree structure for weighted sampling with dynamic updates.

    Implements a tree of SampleTreeNodes and SampleTreeEntries that enables
    O(log N) weighted sampling, insertion, deletion, and priority updates.
    Each node maintains the sum of its children's unnormalized probabilities
    (uprobs), and sampling traverses from root to leaf by selecting children
    proportional to their uprobs.

    The tree grows dynamically as entries are added, maintaining a balanced
    structure with configurable branching factor. Used internally by the
    Prioritized selector for efficient priority-based sampling.

    Attributes:
        branching: Maximum number of children per node.
        root: Root SampleTreeNode of the tree.
        last: Most recently inserted SampleTreeEntry.
        entries: Dictionary mapping keys to their SampleTreeEntry objects.
        rng: NumPy random number generator for reproducible sampling.
    """

    def __init__(self, branching=16, seed=0):
        """Initialize an empty sample tree.

        Args:
            branching: Maximum children per node. Must be >= 2. Default 16.
            seed: Random seed for reproducible sampling. Default 0.

        Raises:
            AssertionError: If branching < 2.
        """
        assert branching >= 2
        self.branching = branching
        self.root = SampleTreeNode()
        self.last = None
        self.entries = {}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        """Return the number of entries in the tree.

        Returns:
            Number of entries currently stored.
        """
        return len(self.entries)

    def insert(self, key, uprob):
        """Insert a new entry into the tree with specified unnormalized probability.

        Finds an appropriate leaf position for the new entry and updates all
        ancestor nodes' uprobs. Grows the tree height if necessary.

        Args:
            key: Unique identifier for the entry.
            uprob: Unnormalized probability (sampling weight) for the entry.
        """
        if not self.last:
            node = self.root
        else:
            ups = 0
            node = self.last.parent
            while node and len(node) >= self.branching:
                node = node.parent
                ups += 1
            if not node:
                node = SampleTreeNode()
                node.append(self.root)
                self.root = node
            for _ in range(ups):
                below = SampleTreeNode()
                node.append(below)
                node = below
        entry = SampleTreeEntry(key, uprob)
        node.append(entry)
        self.entries[key] = entry
        self.last = entry

    def remove(self, key):
        """Remove an entry from the tree.

        Removes the entry and moves the last entry to fill its position to
        maintain tree structure. Updates all affected ancestor nodes' uprobs.
        Shrinks tree height if necessary.

        Args:
            key: Entry key to remove.

        Raises:
            KeyError: If key is not in the tree.
        """
        entry = self.entries.pop(key)
        entry_parent = entry.parent
        last_parent = self.last.parent
        entry.parent.remove(entry)
        if entry is not self.last:
            entry_parent.append(self.last)
        node = last_parent
        ups = 0
        while node.parent and not len(node):
            above = node.parent
            above.remove(node)
            node = above
            ups += 1
        if not len(node):
            self.last = None
            return
        while isinstance(node, SampleTreeNode):
            node = node.children[-1]
        self.last = node

    def update(self, key, uprob):
        """Update the unnormalized probability of an existing entry.

        Updates the entry's uprob and recomputes all ancestor nodes' uprobs
        to maintain correct sampling probabilities.

        Args:
            key: Entry key to update.
            uprob: New unnormalized probability for the entry.

        Raises:
            KeyError: If key is not in the tree.
        """
        entry = self.entries[key]
        entry.uprob = uprob
        entry.parent.recompute()

    def sample(self):
        """Sample an entry key according to unnormalized probabilities.

        Traverses from root to leaf, selecting children at each level
        proportional to their uprobs. Handles edge cases like zero or
        infinite probabilities.

        Returns:
            The key of the sampled entry.

        Raises:
            ValueError: If tree is empty (no entries to sample).
        """
        node = self.root
        while isinstance(node, SampleTreeNode):
            uprobs = np.array([x.uprob for x in node.children])
            total = uprobs.sum()
            if not np.isfinite(total):
                finite = np.isinf(uprobs)
                probs = finite / finite.sum()
            elif total == 0:
                probs = np.ones(len(uprobs)) / len(uprobs)
            else:
                probs = uprobs / total
            choice = self.rng.choice(np.arange(len(uprobs)), p=probs)
            node = node.children[choice.item()]
        return node.key


class SampleTreeNode:
    """Internal node in a SampleTree that aggregates children's probabilities.

    A node in the tree structure that can contain either other nodes or leaf
    entries. Maintains the sum of its children's unnormalized probabilities
    (uprobs) and recursively updates parent nodes when modified.

    Uses __slots__ for memory efficiency since many nodes may exist in large
    trees.

    Attributes:
        children: List of child nodes or entries (SampleTreeNode or SampleTreeEntry).
        parent: Parent SampleTreeNode, or None if this is the root.
        uprob: Sum of children's uprobs (unnormalized probability).
    """

    __slots__ = ("children", "parent", "uprob")

    def __init__(self, parent=None):
        """Initialize a sample tree node.

        Args:
            parent: Parent node, or None for root node. Default None.
        """
        self.parent = parent
        self.children = []
        self.uprob = 0

    def __repr__(self):
        """Return string representation showing uprob and children's uprobs.

        Returns:
            String representation of the node.
        """
        return (
            f"SampleTreeNode(uprob={self.uprob}, "
            f"children={[x.uprob for x in self.children]})"
        )

    def __len__(self):
        """Return the number of children.

        Returns:
            Number of children in this node.
        """
        return len(self.children)

    def __bool__(self):
        """Return True (nodes are always truthy even if empty).

        Returns:
            Always True.
        """
        return True

    def append(self, child):
        """Add a child node or entry and recompute uprobs.

        If the child already has a parent, removes it from that parent first.
        After appending, recomputes uprobs up the tree.

        Args:
            child: SampleTreeNode or SampleTreeEntry to add as a child.
        """
        if child.parent:
            child.parent.remove(child)
        child.parent = self
        self.children.append(child)
        self.recompute()

    def remove(self, child):
        """Remove a child node or entry and recompute uprobs.

        Detaches the child from this node and recomputes uprobs up the tree.

        Args:
            child: SampleTreeNode or SampleTreeEntry to remove.

        Raises:
            ValueError: If child is not in children list.
        """
        child.parent = None
        self.children.remove(child)
        self.recompute()

    def recompute(self):
        """Recompute this node's uprob from children and propagate upward.

        Sums children's uprobs and recursively calls recompute on parent if
        one exists. This ensures the entire path to the root reflects the
        updated probabilities.
        """
        self.uprob = sum(x.uprob for x in self.children)
        self.parent and self.parent.recompute()


class SampleTreeEntry:
    """Leaf entry in a SampleTree representing a samplable item.

    A leaf node in the tree structure that stores a key (sequence identifier)
    and its unnormalized probability (uprob). When the tree is sampled, it
    traverses from root to a leaf entry and returns that entry's key.

    Uses __slots__ for memory efficiency since many entries may exist in large
    replay buffers.

    Attributes:
        key: Unique identifier for the samplable sequence.
        parent: Parent SampleTreeNode containing this entry.
        uprob: Unnormalized probability (sampling weight) for this entry.
    """

    __slots__ = ("key", "parent", "uprob")

    def __init__(self, key=None, uprob=None):
        """Initialize a sample tree entry.

        Args:
            key: Unique identifier for the sequence. Default None.
            uprob: Unnormalized probability (sampling weight). Default None.
        """
        self.parent = None
        self.key = key
        self.uprob = uprob
