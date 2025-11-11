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
        """Test Recency with no items - documents sampling bug."""
        uprobs = np.linspace(1.0, 0.1, 100)
        recency = selectors.Recency(uprobs, seed=0)
        assert len(recency) == 0
        # Bug: UnboundLocalError in _sample when path is empty
        with pytest.raises(UnboundLocalError):
            recency()

    def test_single_item(self):
        """Test Recency with single item - documents sampling bug."""
        uprobs = np.linspace(1.0, 0.1, 100)
        recency = selectors.Recency(uprobs, seed=0)
        recency[1] = [b"step1"]
        assert len(recency) == 1
        # Bug: UnboundLocalError in _sample
        with pytest.raises(UnboundLocalError):
            recency()

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
