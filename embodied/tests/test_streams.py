import functools
import queue
import threading
import time

import elements
import numpy as np
import pytest

import embodied


class TestStateless:
    """Tests for Stateless stream wrapper."""

    def test_basic_function(self):
        """Test wrapping a basic callable function."""
        counter = [0]

        def count_fn():
            counter[0] += 1
            return counter[0]

        stream = embodied.core.streams.Stateless(count_fn)
        assert next(stream) == 1
        assert next(stream) == 2
        assert next(stream) == 3

    def test_with_args(self):
        """Test function with arguments."""

        def add_fn(a, b):
            return a + b

        stream = embodied.core.streams.Stateless(add_fn, 10, 5)
        assert next(stream) == 15
        assert next(stream) == 15

    def test_with_kwargs(self):
        """Test function with keyword arguments."""

        def multiply_fn(x, factor=2):
            return x * factor

        stream = embodied.core.streams.Stateless(multiply_fn, 5, factor=3)
        assert next(stream) == 15

    def test_with_iterator(self):
        """Test wrapping an iterator object."""
        data = iter([1, 2, 3, 4, 5])
        stream = embodied.core.streams.Stateless(data)
        assert next(stream) == 1
        assert next(stream) == 2
        assert next(stream) == 3

    def test_iter_returns_self(self):
        """Test __iter__ returns self."""
        stream = embodied.core.streams.Stateless(lambda: 1)
        assert iter(stream) is stream

    def test_save_returns_none(self):
        """Test save returns None (stateless)."""
        stream = embodied.core.streams.Stateless(lambda: 1)
        assert stream.save() is None

    def test_load_does_nothing(self):
        """Test load does nothing (stateless)."""
        stream = embodied.core.streams.Stateless(lambda: 1)
        stream.load({"anything": "here"})
        assert next(stream) == 1

    def test_generator_function(self):
        """Test wrapping a generator."""

        def gen():
            yield 1
            yield 2
            yield 3

        stream = embodied.core.streams.Stateless(gen())
        assert next(stream) == 1
        assert next(stream) == 2
        assert next(stream) == 3


class TestPrefetch:
    """Tests for Prefetch stream for async data loading."""

    def test_basic_prefetch(self):
        """Test basic prefetching without transform."""

        class SimpleSource:
            def __init__(self):
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.count += 1
                return self.count

            def save(self):
                return {"count": self.count}

            def load(self, state):
                self.count = state["count"]

        source = SimpleSource()
        stream = embodied.core.streams.Prefetch(source, amount=2)

        # Start iteration
        iter(stream)
        time.sleep(0.1)  # Give worker time to prefetch

        assert next(stream) == 1
        assert next(stream) == 2

        # Test save
        state = stream.save()
        assert state is not None
        assert "count" in state

        # Test load after started
        stream.load(state)
        time.sleep(0.05)

        # Continue consuming
        result = next(stream)
        assert result > 0


class TestConsec:
    """Tests for Consec stream that creates consecutive chunks."""

    def test_basic_chunking(self):
        """Test basic consecutive chunking."""

        class SimpleSource:
            def __init__(self):
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                batch_size = 2
                seq_len = 9  # 3 chunks of length 3
                data = {
                    "is_first": np.zeros((batch_size, seq_len), dtype=bool),
                    "value": np.arange(seq_len)[None, :].repeat(batch_size, axis=0),
                }
                data["is_first"][:, 0] = True
                self.count += 1
                return data

            def save(self):
                return {"count": self.count}

            def load(self, state):
                self.count = state["count"]

        source = SimpleSource()
        stream = embodied.core.streams.Consec(
            source, length=3, consec=3, prefix=0, strict=True
        )
        it = iter(stream)

        chunk1 = next(it)
        assert chunk1["value"].shape == (2, 3)
        assert (chunk1["value"][0] == [0, 1, 2]).all()
        assert (chunk1["consec"] == 0).all()

        chunk2 = next(it)
        assert chunk2["value"].shape == (2, 3)
        assert (chunk2["value"][0] == [3, 4, 5]).all()
        assert (chunk2["consec"] == 1).all()

        chunk3 = next(it)
        assert chunk3["value"].shape == (2, 3)
        assert (chunk3["value"][0] == [6, 7, 8]).all()
        assert (chunk3["consec"] == 2).all()

    def test_with_prefix(self):
        """Test chunking with prefix."""

        class SimpleSource:
            def __init__(self):
                pass

            def __iter__(self):
                return self

            def __next__(self):
                batch_size = 1
                seq_len = 11  # 3 chunks of length 3 with prefix 2
                data = {
                    "is_first": np.zeros((batch_size, seq_len), dtype=bool),
                    "value": np.arange(seq_len)[None, :],
                }
                data["is_first"][:, 0] = True
                return data

            def save(self):
                return {}

            def load(self, state):
                pass

        source = SimpleSource()
        stream = embodied.core.streams.Consec(
            source, length=3, consec=3, prefix=2, strict=True
        )
        it = iter(stream)

        chunk1 = next(it)
        assert chunk1["value"].shape == (1, 5)  # length + prefix
        assert (chunk1["value"][0] == [0, 1, 2, 3, 4]).all()

        chunk2 = next(it)
        assert (chunk2["value"][0] == [3, 4, 5, 6, 7]).all()

        chunk3 = next(it)
        assert (chunk3["value"][0] == [6, 7, 8, 9, 10]).all()

    def test_non_strict_mode(self):
        """Test non-strict mode allows extra data."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                batch_size = 1
                seq_len = 15  # More than needed
                data = {
                    "is_first": np.zeros((batch_size, seq_len), dtype=bool),
                    "value": np.arange(seq_len)[None, :],
                }
                data["is_first"][:, 0] = True
                return data

            def save(self):
                return {}

            def load(self, state):
                pass

        source = SimpleSource()
        stream = embodied.core.streams.Consec(
            source, length=3, consec=3, prefix=0, strict=False
        )
        it = iter(stream)

        # Should work even though source has more data than needed
        chunk1 = next(it)
        assert chunk1["value"].shape == (1, 3)

    def test_strict_mode_assertion(self):
        """Test strict mode raises assertion on size mismatch."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                batch_size = 1
                seq_len = 15  # More than needed
                data = {
                    "is_first": np.zeros((batch_size, seq_len), dtype=bool),
                    "value": np.arange(seq_len)[None, :],
                }
                return data

            def save(self):
                return {}

            def load(self, state):
                pass

        source = SimpleSource()
        stream = embodied.core.streams.Consec(
            source, length=3, consec=3, prefix=0, strict=True
        )
        it = iter(stream)

        with pytest.raises(AssertionError):
            next(it)

    def test_insufficient_data_assertion(self):
        """Test assertion when source doesn't have enough data."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                batch_size = 1
                seq_len = 5  # Not enough for 3 chunks of length 3
                data = {
                    "is_first": np.zeros((batch_size, seq_len), dtype=bool),
                    "value": np.arange(seq_len)[None, :],
                }
                return data

            def save(self):
                return {}

            def load(self, state):
                pass

        source = SimpleSource()
        stream = embodied.core.streams.Consec(
            source, length=3, consec=3, prefix=0, strict=False
        )
        it = iter(stream)

        with pytest.raises(AssertionError):
            next(it)

    def test_contiguous_flag(self):
        """Test contiguous array creation."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                batch_size = 2
                seq_len = 9
                data = {
                    "is_first": np.zeros((batch_size, seq_len), dtype=bool),
                    "value": np.arange(seq_len)[None, :].repeat(batch_size, axis=0),
                }
                data["is_first"][:, 0] = True
                return data

            def save(self):
                return {}

            def load(self, state):
                pass

        source = SimpleSource()
        stream = embodied.core.streams.Consec(
            source, length=3, consec=3, contiguous=True
        )
        it = iter(stream)

        chunk = next(it)
        # Check that arrays are contiguous
        assert chunk["value"].flags["C_CONTIGUOUS"]
        assert chunk["is_first"].flags["C_CONTIGUOUS"]

    def test_save_load(self):
        """Test save and load functionality."""

        class SimpleSource:
            def __init__(self):
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                batch_size = 1
                seq_len = 9
                data = {
                    "is_first": np.zeros((batch_size, seq_len), dtype=bool),
                    "value": np.full((batch_size, seq_len), self.count),
                }
                data["is_first"][:, 0] = True
                self.count += 1
                return data

            def save(self):
                return {"count": self.count}

            def load(self, state):
                self.count = state["count"]

        source = SimpleSource()
        stream = embodied.core.streams.Consec(source, length=3, consec=3)
        it = iter(stream)

        next(it)
        next(it)  # Now index is 2

        state = stream.save()
        assert state["index"] == 2
        assert "source" in state

        # Reset and load
        stream2 = embodied.core.streams.Consec(SimpleSource(), length=3, consec=3)
        stream2.load(state)
        assert stream2.index == 2

    def test_index_reset(self):
        """Test that index resets after full cycle."""

        class SimpleSource:
            def __init__(self):
                self.call_count = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.call_count += 1
                batch_size = 1
                seq_len = 9
                data = {
                    "is_first": np.zeros((batch_size, seq_len), dtype=bool),
                    "value": np.full((batch_size, seq_len), self.call_count),
                }
                data["is_first"][:, 0] = True
                return data

            def save(self):
                return {}

            def load(self, state):
                pass

        source = SimpleSource()
        stream = embodied.core.streams.Consec(source, length=3, consec=3)
        it = iter(stream)

        next(it)  # index 0
        next(it)  # index 1
        next(it)  # index 2
        next(it)  # index resets to 0, new batch from source

        # Source should have been called twice
        assert source.call_count == 2


class TestZip:
    """Tests for Zip stream that combines multiple sources."""

    def test_basic_zip(self):
        """Test basic zipping of two sources."""

        class SimpleSource:
            def __init__(self, offset):
                self.offset = offset
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                result = {"value": np.array([[self.count + self.offset]])}
                self.count += 1
                return result

            def save(self):
                return {"count": self.count}

            def load(self, state):
                self.count = state["count"]

        source1 = SimpleSource(0)
        source2 = SimpleSource(100)
        stream = embodied.core.streams.Zip([source1, source2])
        it = iter(stream)

        result = next(it)
        assert result["value"].shape == (2, 1)
        assert result["value"][0, 0] == 0
        assert result["value"][1, 0] == 100

        result = next(it)
        assert result["value"][0, 0] == 1
        assert result["value"][1, 0] == 101

    def test_multiple_sources(self):
        """Test zipping more than two sources."""

        class SimpleSource:
            def __init__(self, value):
                self.value = value

            def __iter__(self):
                return self

            def __next__(self):
                return {"data": np.array([[self.value]])}

            def save(self):
                return {}

            def load(self, state):
                pass

        sources = [SimpleSource(i * 10) for i in range(4)]
        stream = embodied.core.streams.Zip(sources)
        it = iter(stream)

        result = next(it)
        assert result["data"].shape == (4, 1)
        assert (result["data"][:, 0] == [0, 10, 20, 30]).all()

    def test_assertion_on_single_source(self):
        """Test that single source raises assertion."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                return {"value": np.array([[1]])}

        with pytest.raises(AssertionError):
            embodied.core.streams.Zip([SimpleSource()])

    def test_save_load(self):
        """Test save and load functionality."""

        class SimpleSource:
            def __init__(self):
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                result = {"value": np.array([[self.count]])}
                self.count += 1
                return result

            def save(self):
                return {"count": self.count}

            def load(self, state):
                self.count = state["count"]

        sources = [SimpleSource() for _ in range(3)]
        stream = embodied.core.streams.Zip(sources)
        it = iter(stream)

        next(it)
        next(it)

        state = stream.save()
        assert len(state) == 3

        # Load into new stream
        new_sources = [SimpleSource() for _ in range(3)]
        new_stream = embodied.core.streams.Zip(new_sources)
        new_stream.started = True
        new_stream.iterators = [iter(s) for s in new_sources]
        new_stream.load(state)

        result = next(new_stream)
        assert (result["value"][:, 0] == [2, 2, 2]).all()

    def test_iter_assertion(self):
        """Test assertion when iterating twice."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                return {"value": np.array([[1]])}

            def save(self):
                return {}

            def load(self, state):
                pass

        stream = embodied.core.streams.Zip([SimpleSource(), SimpleSource()])
        iter(stream)

        with pytest.raises(AssertionError):
            iter(stream)

    def test_complex_tree_structure(self):
        """Test zipping with complex nested structures."""

        class ComplexSource:
            def __init__(self, offset):
                self.offset = offset

            def __iter__(self):
                return self

            def __next__(self):
                return {
                    "obs": {
                        "image": np.array([[self.offset]]),
                        "state": np.array([[self.offset * 2]]),
                    },
                    "reward": np.array([[self.offset * 3]]),
                }

            def save(self):
                return {}

            def load(self, state):
                pass

        sources = [ComplexSource(i) for i in range(2)]
        stream = embodied.core.streams.Zip(sources)
        it = iter(stream)

        result = next(it)
        assert result["obs"]["image"].shape == (2, 1)
        assert (result["obs"]["image"][:, 0] == [0, 1]).all()
        assert (result["obs"]["state"][:, 0] == [0, 2]).all()
        assert (result["reward"][:, 0] == [0, 3]).all()


class TestMap:
    """Tests for Map stream that applies transformations."""

    def test_basic_map(self):
        """Test basic mapping function."""

        class SimpleSource:
            def __init__(self):
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.count += 1
                return {"value": self.count}

            def save(self):
                return {"count": self.count}

            def load(self, state):
                self.count = state["count"]

        source = SimpleSource()
        stream = embodied.core.streams.Map(
            source, lambda x: {**x, "value": x["value"] * 2}
        )
        it = iter(stream)

        result = next(it)
        assert result["value"] == 2

        result = next(it)
        assert result["value"] == 4

    def test_map_with_args(self):
        """Test map with additional arguments."""

        class SimpleSource:
            def __init__(self):
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.count += 1
                return {"value": self.count}

            def save(self):
                return {}

            def load(self, state):
                pass

        def add_offset(data, offset):
            return {**data, "value": data["value"] + offset}

        source = SimpleSource()
        stream = embodied.core.streams.Map(source, add_offset, 10)
        it = iter(stream)

        result = next(it)
        assert result["value"] == 11

    def test_map_with_kwargs(self):
        """Test map with keyword arguments."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                return {"value": 5}

            def save(self):
                return {}

            def load(self, state):
                pass

        def multiply(data, factor=1):
            return {**data, "value": data["value"] * factor}

        source = SimpleSource()
        stream = embodied.core.streams.Map(source, multiply, factor=3)
        it = iter(stream)

        result = next(it)
        assert result["value"] == 15

    def test_save_load(self):
        """Test save and load functionality."""

        class SimpleSource:
            def __init__(self):
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.count += 1
                return {"value": self.count}

            def save(self):
                return {"count": self.count}

            def load(self, state):
                self.count = state["count"]

        source = SimpleSource()
        stream = embodied.core.streams.Map(source, lambda x: x)
        it = iter(stream)

        next(it)
        next(it)

        state = stream.save()
        assert state["count"] == 2

        # Load into new stream
        new_source = SimpleSource()
        new_stream = embodied.core.streams.Map(new_source, lambda x: x)
        new_stream.started = True
        new_stream.iterator = iter(new_source)
        new_stream.load(state)

        result = next(new_stream)
        assert result["value"] == 3

    def test_iter_assertion(self):
        """Test assertion when iterating twice."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                return {"value": 1}

            def save(self):
                return {}

            def load(self, state):
                pass

        stream = embodied.core.streams.Map(SimpleSource(), lambda x: x)
        iter(stream)

        with pytest.raises(AssertionError):
            iter(stream)

    def test_next_assertion(self):
        """Test assertion when calling next before iter."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                return {"value": 1}

            def save(self):
                return {}

            def load(self, state):
                pass

        stream = embodied.core.streams.Map(SimpleSource(), lambda x: x)

        with pytest.raises(AssertionError):
            next(stream)

    def test_complex_transformation(self):
        """Test complex data transformation."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                return {
                    "obs": np.array([1, 2, 3]),
                    "reward": np.array([0.5]),
                }

            def save(self):
                return {}

            def load(self, state):
                pass

        def normalize(data):
            return {
                "obs": data["obs"] / 10.0,
                "reward": data["reward"] * 2,
            }

        source = SimpleSource()
        stream = embodied.core.streams.Map(source, normalize)
        it = iter(stream)

        result = next(it)
        assert (result["obs"] == [0.1, 0.2, 0.3]).all()
        assert result["reward"] == 1.0


class TestMixer:
    """Tests for Mixer stream that mixes multiple sources."""

    def test_basic_mixing(self):
        """Test basic mixing with equal weights."""

        class SimpleSource:
            def __init__(self, value):
                self.value = value

            def __iter__(self):
                return self

            def __next__(self):
                return {"source": self.value}

            def save(self):
                return {}

            def load(self, state):
                pass

        sources = {"a": SimpleSource(1), "b": SimpleSource(2)}
        weights = {"a": 1.0, "b": 1.0}

        stream = embodied.core.streams.Mixer(sources, weights, seed=42)
        iter(stream)

        # Collect samples
        samples = [next(stream)["source"] for _ in range(100)]

        # Should have mix of both sources
        assert 1 in samples
        assert 2 in samples

    def test_weighted_mixing(self):
        """Test mixing with different weights."""

        class SimpleSource:
            def __init__(self, value):
                self.value = value

            def __iter__(self):
                return self

            def __next__(self):
                return {"source": self.value}

            def save(self):
                return {}

            def load(self, state):
                pass

        sources = {"a": SimpleSource(1), "b": SimpleSource(2)}
        weights = {"a": 9.0, "b": 1.0}  # 90% from a, 10% from b

        stream = embodied.core.streams.Mixer(sources, weights, seed=42)
        iter(stream)

        samples = [next(stream)["source"] for _ in range(1000)]

        # Count occurrences
        count_a = samples.count(1)
        count_b = samples.count(2)

        # Should be roughly 9:1 ratio (with some tolerance)
        assert count_a > count_b
        assert count_a / count_b > 5  # At least 5:1 ratio

    def test_multiple_sources(self):
        """Test mixing more than two sources."""

        class SimpleSource:
            def __init__(self, value):
                self.value = value

            def __iter__(self):
                return self

            def __next__(self):
                return {"source": self.value}

            def save(self):
                return {}

            def load(self, state):
                pass

        sources = {f"src{i}": SimpleSource(i) for i in range(5)}
        weights = {f"src{i}": 1.0 for i in range(5)}

        stream = embodied.core.streams.Mixer(sources, weights, seed=0)
        iter(stream)

        samples = [next(stream)["source"] for _ in range(500)]

        # All sources should be represented
        for i in range(5):
            assert i in samples

    def test_deterministic_with_seed(self):
        """Test that same seed produces same sequence."""

        class SimpleSource:
            def __init__(self, value):
                self.value = value
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.count += 1
                return {"source": self.value, "count": self.count}

            def save(self):
                return {"count": self.count}

            def load(self, state):
                self.count = state["count"]

        # First stream
        sources1 = {"a": SimpleSource(1), "b": SimpleSource(2)}
        weights1 = {"a": 1.0, "b": 1.0}
        stream1 = embodied.core.streams.Mixer(sources1, weights1, seed=123)
        iter(stream1)

        samples1 = [next(stream1)["source"] for _ in range(50)]

        # Second stream with same seed
        sources2 = {"a": SimpleSource(1), "b": SimpleSource(2)}
        weights2 = {"a": 1.0, "b": 1.0}
        stream2 = embodied.core.streams.Mixer(sources2, weights2, seed=123)
        iter(stream2)

        samples2 = [next(stream2)["source"] for _ in range(50)]

        # Should produce same sequence
        assert samples1 == samples2

    def test_save_load(self):
        """Test save and load functionality."""

        class SimpleSource:
            def __init__(self, value):
                self.value = value
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.count += 1
                return {"source": self.value, "count": self.count}

            def save(self):
                return {"count": self.count}

            def load(self, state):
                self.count = state["count"]

        sources = {"a": SimpleSource(1), "b": SimpleSource(2)}
        weights = {"a": 1.0, "b": 1.0}
        stream = embodied.core.streams.Mixer(sources, weights, seed=42)
        iter(stream)

        # Get some samples
        for _ in range(10):
            next(stream)

        state = stream.save()
        assert state["step"] == 10
        assert state["seed"] == 42
        assert "sources" in state
        assert len(state["sources"]) == 2

        # Test load
        new_sources = {"a": SimpleSource(1), "b": SimpleSource(2)}
        new_stream = embodied.core.streams.Mixer(new_sources, weights, seed=42)
        iter(new_stream)
        new_stream.load(state)

        assert new_stream.step == 10
        assert new_stream.seed == 42

    def test_assertion_on_key_mismatch(self):
        """Test assertion when source and weight keys don't match."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                return {"value": 1}

        sources = {"a": SimpleSource(), "b": SimpleSource()}
        weights = {"a": 1.0, "c": 1.0}  # Mismatched keys

        with pytest.raises(AssertionError):
            embodied.core.streams.Mixer(sources, weights)

    def test_iter_assertion(self):
        """Test assertion when calling next before setting started."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                return {"value": 1}

            def save(self):
                return {}

            def load(self, state):
                pass

        sources = {"a": SimpleSource(), "b": SimpleSource()}
        weights = {"a": 1.0, "b": 1.0}
        stream = embodied.core.streams.Mixer(sources, weights)

        # started is False by default
        with pytest.raises(AssertionError):
            next(stream)

    def test_probability_normalization(self):
        """Test that weights are normalized to probabilities."""

        class SimpleSource:
            def __init__(self, value):
                self.value = value

            def __iter__(self):
                return self

            def __next__(self):
                return {"source": self.value}

            def save(self):
                return {}

            def load(self, state):
                pass

        sources = {"a": SimpleSource(1), "b": SimpleSource(2)}
        weights = {"a": 200.0, "b": 200.0}  # Large weights should normalize to 0.5 each

        stream = embodied.core.streams.Mixer(sources, weights, seed=0)
        assert abs(stream.probs.sum() - 1.0) < 1e-6

    def test_iter_returns_self(self):
        """Test __iter__ returns self."""

        class SimpleSource:
            def __iter__(self):
                return self

            def __next__(self):
                return {"value": 1}

            def save(self):
                return {}

            def load(self, state):
                pass

        sources = {"a": SimpleSource(), "b": SimpleSource()}
        weights = {"a": 1.0, "b": 1.0}
        stream = embodied.core.streams.Mixer(sources, weights)

        assert iter(stream) is stream


class TestStreamIntegration:
    """Integration tests combining multiple stream types."""

    def test_stateless_with_map(self):
        """Test Stateless with Map transformation."""
        counter = [0]

        def count_fn():
            counter[0] += 1
            return {"count": counter[0]}

        stateless = embodied.core.streams.Stateless(count_fn)
        mapped = embodied.core.streams.Map(
            stateless, lambda x: {**x, "squared": x["count"] ** 2}
        )

        iter(mapped)
        result = next(mapped)
        assert result["count"] == 1
        assert result["squared"] == 1

        result = next(mapped)
        assert result["count"] == 2
        assert result["squared"] == 4
