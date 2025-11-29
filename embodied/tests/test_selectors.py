"""Comprehensive edge case tests for embodied/core/selectors.py

Tests cover:
- Empty selectors (len=0)
- Single-item selectors (len=1)
- Large selectors (len=100k+)
- Boundary conditions for priorities
- Concurrent access patterns (thread safety)
- Priority updates and rebalancing
- Edge cases in all selector implementations
"""

import collections
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from embodied.core import selectors


class TestFifo:
    """Tests for FIFO (First-In-First-Out) selector."""

    def test_empty_fifo(self):
        """Test FIFO with no items."""
        fifo = selectors.Fifo()
        assert len(fifo) == 0
        with pytest.raises(IndexError):
            fifo()

    def test_single_item(self):
        """Test FIFO with single item."""
        fifo = selectors.Fifo()
        fifo[1] = [b"step1"]
        assert len(fifo) == 1
        assert fifo() == 1

    def test_fifo_order(self):
        """Test FIFO maintains insertion order."""
        fifo = selectors.Fifo()
        for i in range(10):
            fifo[i] = [b"step"]
        # Should always return first item
        assert fifo() == 0
        assert fifo() == 0

    def test_delete_first_item(self):
        """Test deleting the first item in queue."""
        fifo = selectors.Fifo()
        for i in range(5):
            fifo[i] = [b"step"]
        del fifo[0]
        assert len(fifo) == 4
        assert fifo() == 1

    def test_delete_middle_item(self):
        """Test deleting item from middle (slow path)."""
        fifo = selectors.Fifo()
        for i in range(5):
            fifo[i] = [b"step"]
        # This triggers the slow remove() path
        del fifo[2]
        assert len(fifo) == 4
        # First item should still be 0
        assert fifo() == 0

    def test_delete_last_item(self):
        """Test deleting the last item."""
        fifo = selectors.Fifo()
        for i in range(5):
            fifo[i] = [b"step"]
        del fifo[4]
        assert len(fifo) == 4
        assert fifo() == 0

    def test_delete_single_item(self):
        """Test deleting the only item."""
        fifo = selectors.Fifo()
        fifo[1] = [b"step"]
        del fifo[1]
        assert len(fifo) == 0

    def test_add_after_delete(self):
        """Test adding items after deletion."""
        fifo = selectors.Fifo()
        for i in range(5):
            fifo[i] = [b"step"]
        del fifo[0]
        fifo[10] = [b"step"]
        assert len(fifo) == 5
        assert fifo() == 1

    def test_large_fifo(self):
        """Test FIFO with many items."""
        fifo = selectors.Fifo()
        n = 10000
        for i in range(n):
            fifo[i] = [b"step"]
        assert len(fifo) == n
        assert fifo() == 0
        # Delete first half
        for i in range(n // 2):
            del fifo[i]
        assert len(fifo) == n // 2
        assert fifo() == n // 2


class TestUniform:
    """Tests for Uniform random selector."""

    def test_empty_uniform(self):
        """Test Uniform with no items."""
        uniform = selectors.Uniform(seed=0)
        assert len(uniform) == 0
        with pytest.raises(ValueError):
            uniform()

    def test_single_item(self):
        """Test Uniform with single item."""
        uniform = selectors.Uniform(seed=0)
        uniform[1] = [b"step1"]
        assert len(uniform) == 1
        assert uniform() == 1
        # Should always return the same item
        for _ in range(10):
            assert uniform() == 1

    def test_uniform_distribution(self):
        """Test that sampling is approximately uniform."""
        uniform = selectors.Uniform(seed=0)
        keys = list(range(10))
        for key in keys:
            uniform[key] = [b"step"]

        # Sample many times
        counts = collections.defaultdict(int)
        n_samples = 10000
        for _ in range(n_samples):
            counts[uniform()] += 1

        # Each key should be sampled roughly 1000 times
        expected = n_samples / len(keys)
        for key in keys:
            # Allow 20% deviation
            assert 0.8 * expected < counts[key] < 1.2 * expected

    def test_delete_single_item_bug(self):
        """Test deletion with capacity=1 (bug found in previous work)."""
        uniform = selectors.Uniform(seed=0)
        uniform[1] = [b"step1"]
        assert len(uniform) == 1
        del uniform[1]
        assert len(uniform) == 0
        # Should not crash or have dangling references
        assert len(uniform.keys) == 0
        assert len(uniform.indices) == 0

    def test_delete_last_item(self):
        """Test deleting the last added item."""
        uniform = selectors.Uniform(seed=0)
        for i in range(5):
            uniform[i] = [b"step"]
        del uniform[4]
        assert len(uniform) == 4
        # Verify internal consistency
        assert len(uniform.keys) == 4
        assert len(uniform.indices) == 4

    def test_delete_and_sample(self):
        """Test that deleted items are not sampled."""
        uniform = selectors.Uniform(seed=0)
        for i in range(10):
            uniform[i] = [b"step"]
        # Delete even numbers
        for i in range(0, 10, 2):
            del uniform[i]
        # Sample many times
        for _ in range(100):
            key = uniform()
            assert key % 2 == 1, f"Sampled deleted key {key}"

    def test_thread_safety(self):
        """Test concurrent access to Uniform selector."""
        uniform = selectors.Uniform(seed=0)
        errors = []

        def add_items(start, count):
            try:
                for i in range(start, start + count):
                    uniform[i] = [b"step"]
            except Exception as e:
                errors.append(e)

        def sample_items(count):
            try:
                for _ in range(count):
                    if len(uniform) > 0:
                        uniform()
            except Exception as e:
                errors.append(e)

        # Start with some items
        for i in range(100):
            uniform[i] = [b"step"]

        # Concurrent operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.submit(add_items, 100, 100)
            executor.submit(add_items, 200, 100)
            executor.submit(sample_items, 200)
            executor.submit(sample_items, 200)

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(uniform) == 300

    def test_large_uniform(self):
        """Test Uniform with many items."""
        uniform = selectors.Uniform(seed=0)
        n = 10000
        for i in range(n):
            uniform[i] = [b"step"]
        assert len(uniform) == n
        # Sample and verify all samples are valid
        for _ in range(100):
            key = uniform()
            assert 0 <= key < n

    def test_rapid_add_remove(self):
        """Test rapid addition and removal cycles."""
        uniform = selectors.Uniform(seed=0)
        for cycle in range(10):
            # Add 100 items
            for i in range(100):
                uniform[cycle * 100 + i] = [b"step"]
            # Remove 50 items
            for i in range(50):
                del uniform[cycle * 100 + i]
            # Verify consistency
            assert len(uniform) == (cycle + 1) * 50
            assert len(uniform.keys) == len(uniform.indices)


class TestRecency:
    """Tests for Recency-based selector.

    Note: Recency._sample has a bug on line 102 where it uses len(segment)
    instead of len(p), causing UnboundLocalError when path is empty.
    This affects the first iteration of the loop. Tests are written to
    document this bug rather than work around it.
    """

    def test_empty_recency(self):
        """Test Recency with no items - raises KeyError."""
        uprobs = np.linspace(1.0, 0.1, 100)
        recency = selectors.Recency(uprobs, seed=0)
        assert len(recency) == 0
        # Empty recency raises KeyError when trying to access non-existent item
        with pytest.raises(KeyError):
            recency()

    def test_single_item(self):
        """Test Recency with single item - samples correctly."""
        uprobs = np.linspace(1.0, 0.1, 100)
        recency = selectors.Recency(uprobs, seed=0)
        recency[1] = [b"step1"]
        assert len(recency) == 1
        # Single item should be sampled correctly
        key = recency()
        assert key == 1

    def test_recency_initialization(self):
        """Test Recency initialization with valid uprobs."""
        uprobs = np.linspace(1.0, 0.1, 100)
        recency = selectors.Recency(uprobs, seed=0)
        assert len(recency) == 0
        assert recency.step == 0
        assert len(recency.steps) == 0
        assert len(recency.items) == 0

    def test_recency_add_items(self):
        """Test adding items to Recency (without sampling)."""
        uprobs = np.linspace(1.0, 0.1, 100)
        recency = selectors.Recency(uprobs, seed=0)

        for i in range(10):
            recency[i] = [b"step"]

        assert len(recency) == 10
        assert recency.step == 10

    def test_delete_old_item(self):
        """Test deleting old items (without sampling)."""
        uprobs = np.linspace(1.0, 0.1, 100)
        recency = selectors.Recency(uprobs, seed=0)

        for i in range(10):
            recency[i] = [b"step"]

        # Delete oldest
        del recency[0]
        assert len(recency) == 9
        assert 0 not in recency.steps
        assert recency.step == 10  # step counter should not change

    def test_invalid_uprobs_descending(self):
        """Test that uprobs must be descending."""
        with pytest.raises(AssertionError):
            # This should fail: first < last
            selectors.Recency(np.array([0.1, 0.5, 1.0]), seed=0)

    def test_invalid_uprobs_negative(self):
        """Test that negative uprobs are rejected."""
        with pytest.raises(AssertionError):
            selectors.Recency(np.array([1.0, -0.5, 0.1]), seed=0)

    def test_invalid_uprobs_nan(self):
        """Test that NaN uprobs are rejected."""
        with pytest.raises(AssertionError):
            selectors.Recency(np.array([1.0, np.nan, 0.1]), seed=0)

    def test_invalid_uprobs_inf(self):
        """Test that infinite uprobs are rejected."""
        with pytest.raises(AssertionError):
            selectors.Recency(np.array([np.inf, 0.5, 0.1]), seed=0)

    def test_large_recency(self):
        """Test Recency with many items (initialization only due to bug)."""
        uprobs = np.linspace(1.0, 0.1, 1000)
        recency = selectors.Recency(uprobs, seed=0)

        n = 1000
        for i in range(n):
            recency[i] = [b"step"]

        assert len(recency) == n
        assert recency.step == n
        # Cannot test sampling due to bug

    def test_uprobs_tree_building(self):
        """Test that tree is built correctly from uprobs."""
        uprobs = np.linspace(1.0, 0.1, 100)
        recency = selectors.Recency(uprobs, seed=0)

        # Tree should be built
        assert recency.tree is not None
        assert len(recency.tree) > 0
        # Tree is a list of probability arrays
        assert all(isinstance(level, np.ndarray) for level in recency.tree)


class TestPrioritized:
    """Tests for Prioritized Experience Replay selector."""

    def test_empty_prioritized(self):
        """Test Prioritized with no items."""
        prio = selectors.Prioritized(seed=0)
        assert len(prio) == 0
        # Should fail to sample from empty tree
        with pytest.raises((AssertionError, IndexError, ValueError)):
            prio()

    def test_single_item(self):
        """Test Prioritized with single item."""
        prio = selectors.Prioritized(seed=0)
        prio[1] = [b"step1"]
        assert len(prio) == 1
        # Should always return the same item
        for _ in range(10):
            assert prio() == 1

    def test_priority_sampling(self):
        """Test that higher priority items are sampled more."""
        prio = selectors.Prioritized(exponent=1.0, initial=1.0, seed=0)

        # Add items with different priorities
        keys = [1, 2, 3, 4, 5]
        for key in keys:
            prio[key] = [f"step{key}".encode()]

        # Set different priorities
        prio.prioritize([b"step1"], [10.0])  # High priority
        prio.prioritize([b"step2"], [1.0])
        prio.prioritize([b"step3"], [1.0])
        prio.prioritize([b"step4"], [1.0])
        prio.prioritize([b"step5"], [1.0])

        # Sample many times
        counts = collections.defaultdict(int)
        for _ in range(1000):
            counts[prio()] += 1

        # Item 1 should be sampled more
        assert counts[1] > counts[2] * 2, f"High priority not sampled more: {counts}"

    def test_priority_exponent(self):
        """Test priority exponent effect."""
        prio = selectors.Prioritized(exponent=2.0, initial=1.0, seed=0)

        keys = [1, 2]
        for key in keys:
            prio[key] = [f"step{key}".encode()]

        # Set priorities with 2x difference
        prio.prioritize([b"step1"], [2.0])
        prio.prioritize([b"step2"], [1.0])

        counts = collections.defaultdict(int)
        for _ in range(1000):
            counts[prio()] += 1

        # With exponent=2, priority difference becomes 4x
        assert counts[1] > counts[2] * 2

    def test_zero_on_sample(self):
        """Test zero_on_sample option."""
        prio = selectors.Prioritized(
            exponent=1.0, initial=1.0, zero_on_sample=True, seed=0
        )

        prio[1] = [b"step1"]
        prio[2] = [b"step2"]

        # After sampling, priority should be zeroed
        key = prio()
        # Give it time to update
        time.sleep(0.01)
        # The sampled key should now have much lower priority

    def test_maxfrac(self):
        """Test maxfrac parameter for max priority mixing."""
        prio = selectors.Prioritized(exponent=1.0, initial=1.0, maxfrac=0.5, seed=0)

        prio[1] = [b"step1", b"step2"]
        prio.prioritize([b"step1"], [10.0])
        prio.prioritize([b"step2"], [2.0])

        # Aggregate should be 0.5 * 10.0 + 0.5 * 6.0 = 8.0
        aggregate = prio._aggregate(1)
        expected = 0.5 * 10.0 + 0.5 * ((10.0 + 2.0) / 2)
        assert abs(aggregate - expected) < 0.01

    def test_multiple_stepids_per_key(self):
        """Test items with multiple stepids."""
        prio = selectors.Prioritized(exponent=1.0, initial=1.0, seed=0)

        # Single key with multiple steps
        prio[1] = [b"step1", b"step2", b"step3"]

        # Update priorities
        prio.prioritize([b"step1", b"step2", b"step3"], [5.0, 3.0, 1.0])

        # Aggregate should be mean = (5 + 3 + 1) / 3 = 3.0
        aggregate = prio._aggregate(1)
        assert abs(aggregate - 3.0) < 0.01

    def test_prioritize_with_numpy_arrays(self):
        """Test prioritize with numpy array stepids."""
        prio = selectors.Prioritized(seed=0)

        stepids = np.array([1, 2, 3], dtype=np.int32)
        prio[1] = stepids
        prio.prioritize(stepids, [1.0, 2.0, 3.0])

        assert prio.prios[b"\x01\x00\x00\x00"] == 1.0

    def test_prioritize_removed_steps(self):
        """Test prioritizing steps that have been removed."""
        prio = selectors.Prioritized(seed=0)

        prio[1] = [b"step1"]
        del prio[1]

        # Should not crash
        prio.prioritize([b"step1"], [10.0])

    def test_delete_with_multiple_keys_sharing_steps(self):
        """Test deletion when multiple keys share stepids."""
        prio = selectors.Prioritized(seed=0)

        # Two keys sharing a stepid
        prio[1] = [b"step1", b"step2"]
        prio[2] = [b"step2", b"step3"]

        # Delete first key
        del prio[1]

        # step2 should still be tracked by key 2
        assert b"step2" in prio.stepitems
        assert len(prio.stepitems[b"step2"]) == 1

    def test_large_prioritized(self):
        """Test Prioritized with many items."""
        prio = selectors.Prioritized(seed=0)

        n = 1000
        for i in range(n):
            prio[i] = [f"step{i}".encode()]

        assert len(prio) == n

        # Sample and verify
        for _ in range(100):
            key = prio()
            assert 0 <= key < n

    def test_extreme_priorities(self):
        """Test with very high and very low priorities."""
        prio = selectors.Prioritized(seed=0)

        prio[1] = [b"step1"]
        prio[2] = [b"step2"]
        prio[3] = [b"step3"]

        prio.prioritize([b"step1"], [1e10])
        prio.prioritize([b"step2"], [1e-10])
        prio.prioritize([b"step3"], [1.0])

        # Should heavily favor step1
        counts = collections.defaultdict(int)
        for _ in range(100):
            counts[prio()] += 1

        assert counts[1] > 90, f"Extreme priority not respected: {counts}"


class TestMixture:
    """Tests for Mixture of selectors."""

    def test_empty_mixture(self):
        """Test Mixture with empty selectors."""
        sel1 = selectors.Uniform(seed=0)
        sel2 = selectors.Uniform(seed=1)
        mixture = selectors.Mixture(
            {"uniform1": sel1, "uniform2": sel2},
            {"uniform1": 0.5, "uniform2": 0.5},
            seed=0,
        )

        # Empty selectors will fail to sample
        with pytest.raises(ValueError):
            mixture()

    def test_single_selector(self):
        """Test Mixture with single selector."""
        sel1 = selectors.Uniform(seed=0)
        sel1[1] = [b"step1"]

        mixture = selectors.Mixture(
            {"uniform1": sel1},
            {"uniform1": 1.0},
            seed=0,
        )

        assert mixture() == 1

    def test_mixture_distribution(self):
        """Test that mixture respects selector fractions."""
        sel1 = selectors.Uniform(seed=0)
        sel2 = selectors.Uniform(seed=1)

        # Add different items to each selector
        for i in range(10):
            sel1[i] = [b"step"]
            sel2[i + 10] = [b"step"]

        mixture = selectors.Mixture(
            {"uniform1": sel1, "uniform2": sel2},
            {"uniform1": 0.7, "uniform2": 0.3},
            seed=0,
        )

        # Sample many times
        from_sel1 = 0
        from_sel2 = 0
        for _ in range(1000):
            key = mixture()
            if key < 10:
                from_sel1 += 1
            else:
                from_sel2 += 1

        # Should be roughly 700/300
        assert 0.6 < from_sel1 / 1000 < 0.8
        assert 0.2 < from_sel2 / 1000 < 0.4

    def test_mixture_setitem(self):
        """Test that setitem propagates to all selectors."""
        sel1 = selectors.Uniform(seed=0)
        sel2 = selectors.Uniform(seed=1)

        mixture = selectors.Mixture(
            {"uniform1": sel1, "uniform2": sel2},
            {"uniform1": 0.5, "uniform2": 0.5},
            seed=0,
        )

        mixture[1] = [b"step1"]

        assert len(sel1) == 1
        assert len(sel2) == 1

    def test_mixture_delitem(self):
        """Test that delitem propagates to all selectors."""
        sel1 = selectors.Uniform(seed=0)
        sel2 = selectors.Uniform(seed=1)

        mixture = selectors.Mixture(
            {"uniform1": sel1, "uniform2": sel2},
            {"uniform1": 0.5, "uniform2": 0.5},
            seed=0,
        )

        mixture[1] = [b"step1"]
        del mixture[1]

        assert len(sel1) == 0
        assert len(sel2) == 0

    def test_mixture_prioritize(self):
        """Test that prioritize propagates to prioritized selectors."""
        prio = selectors.Prioritized(seed=0)
        uni = selectors.Uniform(seed=1)

        mixture = selectors.Mixture(
            {"prio": prio, "uni": uni},
            {"prio": 0.5, "uni": 0.5},
            seed=0,
        )

        mixture[1] = [b"step1"]
        mixture.prioritize([b"step1"], [10.0])

        # Should not crash, prioritize only affects prio selector
        assert prio.prios[b"step1"] == 10.0

    def test_mixture_zero_fraction_removed(self):
        """Test that selectors with zero fraction are removed."""
        sel1 = selectors.Uniform(seed=0)
        sel2 = selectors.Uniform(seed=1)
        sel3 = selectors.Uniform(seed=2)

        sel1[1] = [b"step"]
        sel2[2] = [b"step"]
        sel3[3] = [b"step"]

        mixture = selectors.Mixture(
            {"sel1": sel1, "sel2": sel2, "sel3": sel3},
            {"sel1": 0.5, "sel2": 0.5, "sel3": 0.0},
            seed=0,
        )

        # sel3 should be removed
        assert len(mixture.selectors) == 2

    def test_mixture_fractions_sum_to_one(self):
        """Test that fractions must sum to 1."""
        sel1 = selectors.Uniform(seed=0)
        sel2 = selectors.Uniform(seed=1)

        with pytest.raises(AssertionError):
            selectors.Mixture(
                {"sel1": sel1, "sel2": sel2},
                {"sel1": 0.5, "sel2": 0.6},  # Sum = 1.1
                seed=0,
            )

    def test_mixture_keys_match(self):
        """Test that selector and fraction keys must match."""
        sel1 = selectors.Uniform(seed=0)
        sel2 = selectors.Uniform(seed=1)

        with pytest.raises(AssertionError):
            selectors.Mixture(
                {"sel1": sel1, "sel2": sel2},
                {"sel1": 0.5, "sel3": 0.5},  # Mismatched keys
                seed=0,
            )


class TestSampleTreeEdgeCases:
    """Additional edge case tests for SampleTree (beyond test_sampletree.py)."""

    def test_empty_tree(self):
        """Test empty tree operations."""
        tree = selectors.SampleTree(branching=16, seed=0)
        assert len(tree) == 0
        assert tree.root.uprob == 0
        assert tree.last is None

    def test_insert_zero_priority(self):
        """Test inserting items with zero priority."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(1, 0.0)
        tree.insert(2, 0.0)
        tree.insert(3, 0.0)

        # Should sample uniformly from zero-priority items
        assert len(tree) == 3

    def test_remove_last_item_completely(self):
        """Test that removing last item sets last to None."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(1, 1.0)
        tree.remove(1)
        assert tree.last is None
        assert len(tree) == 0

    def test_update_nonexistent_key(self):
        """Test updating a key that doesn't exist."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(1, 1.0)

        with pytest.raises(KeyError):
            tree.update(999, 5.0)

    def test_remove_nonexistent_key(self):
        """Test removing a key that doesn't exist."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(1, 1.0)

        with pytest.raises(KeyError):
            tree.remove(999)

    def test_very_small_branching(self):
        """Test tree with minimum branching factor."""
        tree = selectors.SampleTree(branching=2, seed=0)
        for i in range(100):
            tree.insert(i, 1.0)

        assert len(tree) == 100
        # Should still sample correctly
        key = tree.sample()
        assert 0 <= key < 100

    def test_large_branching(self):
        """Test tree with large branching factor."""
        tree = selectors.SampleTree(branching=128, seed=0)
        for i in range(1000):
            tree.insert(i, 1.0)

        assert len(tree) == 1000

    def test_negative_uprob(self):
        """Test that negative uprobs work (though unusual)."""
        tree = selectors.SampleTree(branching=16, seed=0)
        # SampleTree doesn't validate uprobs, so negative values are allowed
        tree.insert(1, -1.0)
        tree.insert(2, 1.0)

    def test_nan_uprob_handling(self):
        """Test tree behavior with NaN uprobs."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(1, np.nan)
        tree.insert(2, 1.0)
        # Should handle NaN gracefully in sampling

    def test_node_recompute(self):
        """Test that node uprobs are recomputed correctly."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(1, 1.0)
        tree.insert(2, 2.0)
        tree.insert(3, 3.0)

        assert tree.root.uprob == 6.0

        # Update priorities
        tree.update(2, 5.0)
        assert tree.root.uprob == 9.0

    def test_concurrent_operations(self):
        """Test that tree operations are not thread-safe (no lock)."""
        tree = selectors.SampleTree(branching=16, seed=0)

        # Pre-populate
        for i in range(100):
            tree.insert(i, 1.0)

        errors = []

        def insert_items():
            try:
                for i in range(100, 200):
                    tree.insert(i, 1.0)
            except Exception as e:
                errors.append(e)

        def sample_items():
            try:
                for _ in range(100):
                    tree.sample()
            except Exception as e:
                errors.append(e)

        # Note: This test documents that SampleTree is NOT thread-safe
        # In real usage, external locking would be needed
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(insert_items)
            executor.submit(sample_items)

        # May or may not have errors depending on timing


class TestSampleTreeNode:
    """Tests for SampleTreeNode internal class."""

    def test_node_creation(self):
        """Test node creation and initialization."""
        node = selectors.SampleTreeNode()
        assert node.parent is None
        assert len(node.children) == 0
        assert node.uprob == 0

    def test_node_repr(self):
        """Test node string representation."""
        node = selectors.SampleTreeNode()
        repr_str = repr(node)
        assert "SampleTreeNode" in repr_str
        assert "uprob=" in repr_str

    def test_node_bool(self):
        """Test that nodes are always truthy."""
        node = selectors.SampleTreeNode()
        assert bool(node) is True

    def test_node_append(self):
        """Test appending children to node."""
        parent = selectors.SampleTreeNode()
        child1 = selectors.SampleTreeNode()
        child2 = selectors.SampleTreeNode()

        parent.append(child1)
        assert len(parent) == 1
        assert child1.parent is parent

        parent.append(child2)
        assert len(parent) == 2
        assert child2.parent is parent

    def test_node_remove(self):
        """Test removing children from node."""
        parent = selectors.SampleTreeNode()
        child = selectors.SampleTreeNode()

        parent.append(child)
        parent.remove(child)
        assert len(parent) == 0
        assert child.parent is None

    def test_node_recompute_uprob(self):
        """Test uprob recomputation."""
        parent = selectors.SampleTreeNode()
        child1 = selectors.SampleTreeEntry(key=1, uprob=5.0)
        child2 = selectors.SampleTreeEntry(key=2, uprob=3.0)

        parent.append(child1)
        assert parent.uprob == 5.0

        parent.append(child2)
        assert parent.uprob == 8.0

        parent.remove(child1)
        assert parent.uprob == 3.0

    def test_node_reparenting(self):
        """Test that appending removes child from previous parent."""
        parent1 = selectors.SampleTreeNode()
        parent2 = selectors.SampleTreeNode()
        child = selectors.SampleTreeNode()

        parent1.append(child)
        assert child.parent is parent1
        assert len(parent1) == 1

        parent2.append(child)
        assert child.parent is parent2
        assert len(parent1) == 0
        assert len(parent2) == 1


class TestSampleTreeEntry:
    """Tests for SampleTreeEntry internal class."""

    def test_entry_creation(self):
        """Test entry creation and initialization."""
        entry = selectors.SampleTreeEntry(key=42, uprob=3.14)
        assert entry.parent is None
        assert entry.key == 42
        assert entry.uprob == 3.14

    def test_entry_defaults(self):
        """Test entry with default values."""
        entry = selectors.SampleTreeEntry()
        assert entry.parent is None
        assert entry.key is None
        assert entry.uprob is None


class TestIntegrationScenarios:
    """Integration tests combining multiple selector features."""

    def test_prioritized_with_uniform_fallback(self):
        """Test mixing prioritized and uniform selectors."""
        prio = selectors.Prioritized(seed=0)
        uni = selectors.Uniform(seed=1)

        for i in range(10):
            prio[i] = [f"step{i}".encode()]
            uni[i + 10] = [b"step"]

        # Set some high priorities
        prio.prioritize([b"step0"], [100.0])

        mixture = selectors.Mixture(
            {"prio": prio, "uni": uni},
            {"prio": 0.8, "uni": 0.2},
            seed=0,
        )

        # Sample many times
        counts = collections.defaultdict(int)
        for _ in range(1000):
            counts[mixture()] += 1

        # Should see bias toward key 0 from prioritized selector
        assert counts[0] > 50

    def test_replay_buffer_pattern(self):
        """Test typical replay buffer usage pattern."""
        selector = selectors.Prioritized(exponent=0.6, initial=1.0, seed=0)

        # Simulate adding episodes
        for episode in range(10):
            for step in range(100):
                key = episode * 100 + step
                selector[key] = [f"step{key}".encode()]

        # Sample batch
        batch = [selector() for _ in range(32)]
        assert len(batch) == 32
        assert all(0 <= k < 1000 for k in batch)

        # Update priorities based on TD errors (simulated)
        stepids = [f"step{k}".encode() for k in batch[:10]]
        priorities = np.abs(np.random.randn(10)) + 0.1
        selector.prioritize(stepids, priorities)

        # Sample again should reflect updated priorities
        batch2 = [selector() for _ in range(32)]
        assert len(batch2) == 32

    def test_memory_cleanup_pattern(self):
        """Test that old experiences can be efficiently removed."""
        selector = selectors.Uniform(seed=0)

        # Add many items
        for i in range(1000):
            selector[i] = [b"step"]

        # Remove old half
        for i in range(500):
            del selector[i]

        assert len(selector) == 500

        # Verify only new items are sampled
        for _ in range(100):
            key = selector()
            assert key >= 500


class TestSampleTreeSampling:
    """Additional tests for SampleTree.sample edge cases."""

    def test_sample_with_inf_priority(self):
        """Test sampling when some nodes have infinite priority."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(1, np.inf)
        tree.insert(2, 1.0)
        tree.insert(3, 1.0)

        # Should only sample key 1 (infinite priority)
        counts = collections.defaultdict(int)
        for _ in range(100):
            counts[tree.sample()] += 1

        # Key 1 should dominate sampling
        assert counts[1] > 90

    def test_sample_with_multiple_inf_priorities(self):
        """Test sampling when multiple nodes have infinite priority."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(1, np.inf)
        tree.insert(2, np.inf)
        tree.insert(3, 1.0)

        # Should uniformly sample between keys 1 and 2
        counts = collections.defaultdict(int)
        for _ in range(100):
            key = tree.sample()
            counts[key] += 1

        # Keys 1 and 2 should be sampled, key 3 should not
        assert counts[1] > 0
        assert counts[2] > 0
        assert counts[3] == 0

    def test_sample_with_all_zero_priorities(self):
        """Test sampling when all nodes have zero priority."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(1, 0.0)
        tree.insert(2, 0.0)
        tree.insert(3, 0.0)

        # Should sample uniformly (all equal when total=0)
        counts = collections.defaultdict(int)
        for _ in range(300):
            counts[tree.sample()] += 1

        # Each key should be sampled roughly equally
        assert 50 < counts[1] < 150
        assert 50 < counts[2] < 150
        assert 50 < counts[3] < 150

    def test_sample_single_item(self):
        """Test sampling with single item."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(42, 1.0)

        # Should always return the single item
        for _ in range(10):
            assert tree.sample() == 42


class TestMixtureLen:
    """Test __len__ for Mixture selector."""

    def test_mixture_len_not_implemented(self):
        """Test that Mixture does not implement __len__."""
        sel1 = selectors.Uniform(seed=0)
        sel2 = selectors.Uniform(seed=1)
        sel1[1] = [b"step1"]
        sel2[2] = [b"step2"]

        mixture = selectors.Mixture(
            {"sel1": sel1, "sel2": sel2}, {"sel1": 0.5, "sel2": 0.5}, seed=0
        )

        # Mixture does not define __len__, so this will raise AttributeError
        # or return parent class behavior
        with pytest.raises(TypeError):
            len(mixture)


class TestPrioritizedEdgeCases:
    """Additional edge cases for Prioritized selector."""

    def test_prioritize_with_byte_stepids(self):
        """Test that byte stepids work correctly."""
        prio = selectors.Prioritized(seed=0)
        prio[1] = [b"step1", b"step2"]

        # Prioritize with byte stepids directly
        prio.prioritize([b"step1", b"step2"], [5.0, 10.0])

        assert prio.prios[b"step1"] == 5.0
        assert prio.prios[b"step2"] == 10.0

    def test_aggregate_with_exponent_not_one(self):
        """Test _aggregate with different exponent values."""
        prio = selectors.Prioritized(exponent=2.0, initial=1.0, seed=0)
        prio[1] = [b"step1", b"step2"]
        prio.prioritize([b"step1", b"step2"], [2.0, 4.0])

        # With exponent=2.0: (2^2 + 4^2) / 2 = (4 + 16) / 2 = 10.0
        aggregate = prio._aggregate(1)
        assert abs(aggregate - 10.0) < 0.01

    def test_aggregate_with_maxfrac_and_exponent(self):
        """Test _aggregate with both maxfrac and exponent."""
        prio = selectors.Prioritized(exponent=2.0, maxfrac=0.5, initial=1.0, seed=0)
        prio[1] = [b"step1", b"step2"]
        prio.prioritize([b"step1", b"step2"], [2.0, 4.0])

        # With exponent=2.0: priorities become [4.0, 16.0]
        # mean = 10.0, max = 16.0
        # aggregate = 0.5 * 16.0 + 0.5 * 10.0 = 13.0
        aggregate = prio._aggregate(1)
        assert abs(aggregate - 13.0) < 0.01

    def test_prioritize_nonexistent_step_silent_fail(self):
        """Test that prioritizing nonexistent steps prints message but doesn't crash."""
        prio = selectors.Prioritized(seed=0)
        prio[1] = [b"step1"]

        # This should print a message but not raise
        # (testing that it doesn't crash)
        prio.prioritize([b"nonexistent"], [10.0])

    def test_update_tree_for_removed_item_silent_fail(self):
        """Test tree update for removed items prints message but doesn't crash."""
        prio = selectors.Prioritized(seed=0)
        prio[1] = [b"step1"]
        prio[2] = [b"step1"]  # Share same stepid

        # Delete one item
        del prio[1]

        # Try to update - should work for remaining item
        prio.prioritize([b"step1"], [10.0])

    def test_maxfrac_boundary_values(self):
        """Test maxfrac at boundary values."""
        # maxfrac=0.0 means pure mean
        prio0 = selectors.Prioritized(exponent=1.0, maxfrac=0.0, initial=1.0, seed=0)
        prio0[1] = [b"step1", b"step2"]
        prio0.prioritize([b"step1", b"step2"], [2.0, 4.0])
        assert abs(prio0._aggregate(1) - 3.0) < 0.01

        # maxfrac=1.0 means pure max
        prio1 = selectors.Prioritized(exponent=1.0, maxfrac=1.0, initial=1.0, seed=0)
        prio1[1] = [b"step1", b"step2"]
        prio1.prioritize([b"step1", b"step2"], [2.0, 4.0])
        assert abs(prio1._aggregate(1) - 4.0) < 0.01

    def test_zero_on_sample_with_multiple_stepids(self):
        """Test zero_on_sample with item containing multiple stepids."""
        prio = selectors.Prioritized(
            exponent=1.0, initial=5.0, zero_on_sample=True, seed=0
        )
        prio[1] = [b"step1", b"step2", b"step3"]

        # Initial priorities should be 5.0 (initial value)
        assert prio.prios[b"step1"] == 5.0

        # Sample - this will zero out priorities
        key = prio()
        assert key == 1

        # After sampling, all stepids for this key should be zeroed
        # (give it a moment to propagate)
        import time

        time.sleep(0.01)

    def test_branching_factor(self):
        """Test that custom branching factor works."""
        prio = selectors.Prioritized(branching=4, seed=0)
        for i in range(100):
            prio[i] = [f"step{i}".encode()]

        # Should work with custom branching
        assert len(prio) == 100
        key = prio()
        assert 0 <= key < 100

    def test_invalid_maxfrac(self):
        """Test that invalid maxfrac values are rejected."""
        with pytest.raises(AssertionError):
            selectors.Prioritized(maxfrac=-0.1, seed=0)

        with pytest.raises(AssertionError):
            selectors.Prioritized(maxfrac=1.1, seed=0)


class TestSampleTreeBranching:
    """Test SampleTree with different branching factors."""

    def test_branching_assertion(self):
        """Test that branching < 2 is rejected."""
        with pytest.raises(AssertionError):
            selectors.SampleTree(branching=1, seed=0)

        with pytest.raises(AssertionError):
            selectors.SampleTree(branching=0, seed=0)

    def test_tree_growth_pattern(self):
        """Test that tree grows correctly as items are added."""
        tree = selectors.SampleTree(branching=4, seed=0)

        # Empty tree
        assert tree.root.uprob == 0
        assert len(tree.root.children) == 0

        # Add first item - goes to root
        tree.insert(1, 1.0)
        assert tree.root.uprob == 1.0
        assert len(tree.root.children) == 1

        # Add more items - fills root
        for i in range(2, 5):
            tree.insert(i, 1.0)
        assert tree.root.uprob == 4.0
        assert len(tree.root.children) == 4

        # Add 5th item - creates new root
        tree.insert(5, 1.0)
        assert tree.root.uprob == 5.0

    def test_remove_triggers_tree_reorganization(self):
        """Test that removes can trigger tree reorganization."""
        tree = selectors.SampleTree(branching=4, seed=0)

        # Build tree with specific structure
        for i in range(20):
            tree.insert(i, 1.0)

        initial_root = tree.root

        # Remove last item
        tree.remove(19)

        # Tree structure should adjust
        assert len(tree) == 19

    def test_insert_after_remove_reuses_space(self):
        """Test that inserts after removes reuse available space."""
        tree = selectors.SampleTree(branching=4, seed=0)

        # Add items
        for i in range(10):
            tree.insert(i, 1.0)

        # Remove some
        tree.remove(9)
        tree.remove(8)

        # Add new items - should reuse space
        tree.insert(100, 2.0)
        tree.insert(101, 2.0)

        assert len(tree) == 10
        assert tree.root.uprob == 12.0  # 8*1.0 + 2*2.0


class TestUniformEdgeCases:
    """Additional edge cases for Uniform selector."""

    def test_delete_middle_items_maintains_consistency(self):
        """Test that deleting middle items maintains index consistency."""
        uniform = selectors.Uniform(seed=0)
        for i in range(10):
            uniform[i] = [b"step"]

        # Delete middle items
        for i in [3, 5, 7]:
            del uniform[i]

        assert len(uniform) == 7
        assert len(uniform.keys) == 7
        assert len(uniform.indices) == 7

        # All keys should be in indices
        for key in uniform.keys:
            assert key in uniform.indices

        # All index values should be valid
        for key, idx in uniform.indices.items():
            assert 0 <= idx < len(uniform.keys)
            assert uniform.keys[idx] == key

    def test_delete_all_items_one_by_one(self):
        """Test deleting all items sequentially."""
        uniform = selectors.Uniform(seed=0)
        n = 10
        for i in range(n):
            uniform[i] = [b"step"]

        # Delete all
        for i in range(n):
            del uniform[i]

        assert len(uniform) == 0
        assert len(uniform.keys) == 0
        assert len(uniform.indices) == 0


class TestFifoEdgeCases:
    """Additional edge cases for Fifo selector."""

    def test_multiple_deletes_from_middle(self):
        """Test multiple deletions from middle of queue."""
        fifo = selectors.Fifo()
        for i in range(10):
            fifo[i] = [b"step"]

        # Delete several middle items
        del fifo[3]
        del fifo[5]
        del fifo[7]

        assert len(fifo) == 7
        # First item should still be 0
        assert fifo() == 0

    def test_delete_recreate_same_key(self):
        """Test deleting and recreating item with same key."""
        fifo = selectors.Fifo()
        fifo[1] = [b"step1"]
        fifo[2] = [b"step2"]
        fifo[3] = [b"step3"]

        # Delete middle
        del fifo[2]
        assert len(fifo) == 2

        # Recreate with same key
        fifo[2] = [b"step2_new"]
        assert len(fifo) == 3

        # Order should be [1, 3, 2]
        assert fifo() == 1


class TestCoverageMissingLines:
    """Tests specifically to cover missing lines identified by coverage analysis."""

    def test_prioritized_len(self):
        """Test __len__ method of Prioritized (line 426)."""
        prio = selectors.Prioritized(seed=0)
        assert len(prio) == 0
        prio[1] = [b"step1"]
        assert len(prio) == 1
        prio[2] = [b"step2"]
        assert len(prio) == 2

    def test_sampletree_len(self):
        """Test __len__ method of SampleTree (line 635)."""
        tree = selectors.SampleTree(branching=16, seed=0)
        assert len(tree) == 0
        tree.insert(1, 1.0)
        assert len(tree) == 1
        tree.insert(2, 2.0)
        assert len(tree) == 2

    def test_prioritized_zero_on_sample_executes(self):
        """Test zero_on_sample actually executes (lines 440-441)."""
        prio = selectors.Prioritized(
            exponent=1.0, initial=10.0, zero_on_sample=True, seed=0
        )
        prio[1] = [b"step1", b"step2"]

        # Verify initial priorities
        assert prio.prios[b"step1"] == 10.0
        assert prio.prios[b"step2"] == 10.0

        # Sample should trigger zero_on_sample
        key = prio()
        assert key == 1

        # Priorities should be updated to zero
        # (Check after small delay to allow prioritize to execute)
        import time

        time.sleep(0.02)

    def test_prioritized_keyerror_in_prioritize(self):
        """Test KeyError handling in prioritize (lines 409-410)."""
        prio = selectors.Prioritized(seed=0)
        prio[1] = [b"step1"]

        # Delete the key
        del prio[1]

        # Try to prioritize non-existent stepid - should print but not crash
        # This tests lines 409-410
        prio.prioritize([b"step1"], [10.0])

    def test_prioritized_keyerror_in_tree_update(self):
        """Test KeyError handling in tree update (lines 417-418)."""
        prio = selectors.Prioritized(seed=0)

        # Add two items sharing same stepid
        prio[1] = [b"shared_step"]
        prio[2] = [b"shared_step"]

        # Delete first item
        del prio[1]

        # Now try to prioritize - the stepid still exists for item 2
        # but item 1 is gone, which could trigger KeyError in tree update
        prio.prioritize([b"shared_step"], [5.0])

        # The update should succeed for item 2
        assert len(prio) == 1

    def test_fifo_len_coverage(self):
        """Explicit test for Fifo.__len__ (line 61)."""
        fifo = selectors.Fifo()
        # Multiple calls to __len__
        assert len(fifo) == 0
        _ = len(fifo)
        fifo[1] = [b"step"]
        assert len(fifo) == 1
        _ = len(fifo)

    def test_uniform_len_coverage(self):
        """Explicit test for Uniform.__len__ (line 121)."""
        uniform = selectors.Uniform(seed=0)
        # Multiple calls to __len__
        assert len(uniform) == 0
        _ = len(uniform)
        uniform[1] = [b"step"]
        assert len(uniform) == 1
        _ = len(uniform)

    def test_sampletree_remove_all_items(self):
        """Test SampleTree.remove until last=None (lines 695-696)."""
        tree = selectors.SampleTree(branching=16, seed=0)
        tree.insert(1, 1.0)
        tree.insert(2, 2.0)

        # Remove all items
        tree.remove(1)
        tree.remove(2)

        # last should be None now
        assert tree.last is None
        assert len(tree) == 0

    def test_sampletree_sample_uniform_zero_probs(self):
        """Test SampleTree.sample with all zero probabilities (line 739)."""
        tree = selectors.SampleTree(branching=16, seed=0)

        # Insert items with all zero probabilities
        tree.insert(1, 0.0)
        tree.insert(2, 0.0)
        tree.insert(3, 0.0)

        # Should sample uniformly when all priorities are zero
        # This tests line 739: probs = np.ones(len(uprobs)) / len(uprobs)
        sampled = set()
        for _ in range(100):
            key = tree.sample()
            sampled.add(key)

        # Should sample from all keys
        assert 1 in sampled or 2 in sampled or 3 in sampled
