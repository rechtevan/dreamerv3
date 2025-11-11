"""
Tests for embodied.core.chunk - Chunked storage for replay buffer

Coverage target: 90%+ (from 52.56%)

Tests cover:
- Basic operations: repr, comparison, properties
- Data manipulation: append, update, slice
- Persistence: save, load, error handling
- Edge cases: empty chunks, full chunks, boundary conditions
"""

import io
import tempfile
import time

import elements
import numpy as np
import pytest

from embodied.core.chunk import Chunk


class TestChunkInit:
    """Test Chunk initialization"""

    def test_init_default_size(self):
        """Test Chunk initializes with default size"""
        c = Chunk()
        assert c.size == 1024
        assert c.length == 0
        assert c.data is None
        assert c.saved is False
        assert int(c.succ) == 0

    def test_init_custom_size(self):
        """Test Chunk initializes with custom size"""
        c = Chunk(size=512)
        assert c.size == 512
        assert c.length == 0

    def test_init_creates_unique_uuid(self):
        """Test each Chunk gets unique UUID"""
        c1 = Chunk()
        c2 = Chunk()
        assert c1.uuid != c2.uuid
        assert isinstance(c1.uuid, elements.UUID)

    def test_init_creates_timestamp(self):
        """Test Chunk records creation timestamp"""
        before = elements.timestamp(millis=True)
        c = Chunk()
        after = elements.timestamp(millis=True)
        assert before <= c.time <= after


class TestChunkProperties:
    """Test Chunk properties and magic methods"""

    def test_repr_format(self):
        """Test __repr__ returns expected format"""
        c = Chunk(size=10)
        repr_str = repr(c)
        assert repr_str.startswith("Chunk(")
        assert repr_str.endswith(")")
        assert c.filename in repr_str

    def test_lt_comparison_by_time(self):
        """Test chunks compare by timestamp"""
        c1 = Chunk()
        time.sleep(0.002)  # Ensure different millisecond timestamps
        c2 = Chunk()
        assert c1 < c2
        assert not (c2 < c1)

    def test_filename_format(self):
        """Test filename format is correct"""
        c = Chunk()
        filename = c.filename
        parts = filename.split("-")

        # Format: {time}-{uuid}-{succ}-{length}.npz
        assert len(parts) == 4
        assert parts[0] == c.time
        assert parts[1] == str(c.uuid)
        assert parts[2] == str(c.succ)
        assert parts[3] == f"{c.length}.npz"

    def test_filename_with_chunk_successor(self):
        """Test filename when successor is another Chunk"""
        c1 = Chunk()
        c2 = Chunk()
        c1.succ = c2

        filename = c1.filename
        parts = filename.split("-")
        # Should use c2.uuid, not str(c2)
        assert parts[2] == str(c2.uuid)

    def test_nbytes_empty_chunk(self):
        """Test nbytes returns 0 for empty chunk"""
        c = Chunk()
        assert c.nbytes == 0

    def test_nbytes_with_data(self):
        """Test nbytes calculates total size correctly"""
        c = Chunk(size=10)
        step = {
            "obs": np.zeros((64, 64, 3), dtype=np.uint8),
            "action": np.zeros(12, dtype=np.float32),
        }
        c.append(step)

        expected = (10 * 64 * 64 * 3 * 1) + (10 * 12 * 4)  # uint8 + float32
        assert c.nbytes == expected


class TestChunkDataManipulation:
    """Test data append, update, and slice operations"""

    def test_append_first_step(self):
        """Test appending first step initializes data arrays"""
        c = Chunk(size=10)
        step = {"obs": np.ones((64, 64, 3), dtype=np.uint8)}

        c.append(step)

        assert c.length == 1
        assert c.data is not None
        assert "obs" in c.data
        assert c.data["obs"].shape == (10, 64, 64, 3)
        assert c.data["obs"].dtype == np.uint8
        assert np.array_equal(c.data["obs"][0], step["obs"])

    def test_append_multiple_steps(self):
        """Test appending multiple steps"""
        c = Chunk(size=10)

        for i in range(5):
            step = {"value": np.array([i])}
            c.append(step)

        assert c.length == 5
        for i in range(5):
            assert c.data["value"][i] == i

    def test_append_respects_size_limit(self):
        """Test cannot append beyond chunk size"""
        c = Chunk(size=3)

        for i in range(3):
            c.append({"x": np.array([i])})

        assert c.length == 3

        # Should raise assertion error
        with pytest.raises(AssertionError):
            c.append({"x": np.array([99])})

    def test_update_single_index(self):
        """Test updating data at specific index"""
        c = Chunk(size=10)

        for i in range(5):
            c.append({"value": np.array([i])})

        # Update index 2
        c.update(2, 1, {"value": np.array([[99]])})

        assert c.data["value"][2] == 99
        # Other indices unchanged
        assert c.data["value"][1] == 1
        assert c.data["value"][3] == 3

    def test_update_range(self):
        """Test updating multiple consecutive indices"""
        c = Chunk(size=10)

        for i in range(5):
            c.append({"value": np.array([i])})

        # Update indices 1-3
        c.update(1, 3, {"value": np.array([[10], [11], [12]])})

        assert c.data["value"][0] == 0  # Unchanged
        assert c.data["value"][1] == 10
        assert c.data["value"][2] == 11
        assert c.data["value"][3] == 12
        assert c.data["value"][4] == 4  # Unchanged

    def test_update_bounds_checking(self):
        """Test update validates index and length bounds"""
        c = Chunk(size=10)
        c.append({"value": np.array([0])})

        # Index must be >= 0
        with pytest.raises(AssertionError):
            c.update(-1, 1, {"value": np.array([[1]])})

        # Index must be <= length
        with pytest.raises(AssertionError):
            c.update(2, 1, {"value": np.array([[1]])})

        # Index + length must be <= length
        with pytest.raises(AssertionError):
            c.update(0, 2, {"value": np.array([[1], [2]])})

    def test_slice_basic(self):
        """Test slicing returns correct subset"""
        c = Chunk(size=10)

        for i in range(5):
            c.append({"step": np.array([i])})

        sliced = c.slice(1, 3)

        assert "step" in sliced
        assert len(sliced["step"]) == 3
        assert sliced["step"][0] == 1
        assert sliced["step"][1] == 2
        assert sliced["step"][2] == 3

    def test_slice_bounds_checking(self):
        """Test slice validates bounds"""
        c = Chunk(size=10)
        c.append({"x": np.array([0])})

        # Index must be >= 0
        with pytest.raises(AssertionError):
            c.slice(-1, 1)

        # index + length must be <= length
        with pytest.raises(AssertionError):
            c.slice(0, 2)


class TestChunkPersistence:
    """Test save and load functionality"""

    def test_save_creates_file(self):
        """Test save creates NPZ file with correct name"""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = Chunk(size=10)
            c.append({"obs": np.ones((64, 64, 3), dtype=np.uint8)})

            c.save(tmpdir)

            filepath = elements.Path(tmpdir) / c.filename
            assert filepath.exists()
            assert c.saved is True

    def test_save_compression(self):
        """Test save uses compression"""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = Chunk(size=10)
            # Create compressible data (all zeros)
            c.append({"obs": np.zeros((64, 64, 3), dtype=np.uint8)})

            c.save(tmpdir)

            filepath = elements.Path(tmpdir) / c.filename
            file_size = filepath.size

            # Compressed file should be much smaller than uncompressed
            # 10 * 64 * 64 * 3 = 122,880 bytes uncompressed
            # Should compress to <10KB for zeros
            assert file_size < 10 * 1024

    def test_save_only_saved_length(self):
        """Test save only writes used portion of arrays"""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = Chunk(size=100)

            # Only append 3 items
            for i in range(3):
                c.append({"value": np.array([i])})

            c.save(tmpdir)

            # Load and verify only 3 items saved
            filepath = elements.Path(tmpdir) / c.filename
            with filepath.open("rb") as f:
                loaded_data = np.load(f)
                assert loaded_data["value"].shape == (3, 1)

    def test_save_prevents_double_save(self):
        """Test save asserts if already saved"""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = Chunk(size=10)
            c.append({"x": np.array([1])})

            c.save(tmpdir)

            # Second save should fail
            with pytest.raises(AssertionError):
                c.save(tmpdir)

    def test_save_with_logging(self, capsys):
        """Test save prints message when log=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = Chunk(size=10)
            c.append({"x": np.array([1])})

            c.save(tmpdir, log=True)

            captured = capsys.readouterr()
            assert "Saved chunk:" in captured.out
            assert c.filename in captured.out

    def test_load_basic(self):
        """Test load reads saved chunk correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a chunk
            c1 = Chunk(size=10)
            for i in range(5):
                c1.append({"obs": np.array([i])})
            c1.save(tmpdir)

            # Load it back
            filepath = elements.Path(tmpdir) / c1.filename
            c2 = Chunk.load(filepath)

            assert c2.length == 5
            assert c2.size == 5  # Size set to length on load
            assert c2.saved is True
            assert c2.uuid == c1.uuid
            assert c2.succ == c1.succ
            assert c2.time == c1.time
            assert np.array_equal(c2.data["obs"], c1.data["obs"][:5])

    def test_load_parses_filename(self):
        """Test load extracts metadata from filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            c1 = Chunk(size=10)
            c1.append({"x": np.array([1])})
            c1.save(tmpdir)

            filepath = elements.Path(tmpdir) / c1.filename
            c2 = Chunk.load(filepath)

            # Verify filename parsing (line 78)
            time_val, uuid_val, succ_val, length_val = c1.filename.split(".")[0].split(
                "-"
            )
            assert c2.time == time_val
            assert c2.uuid == elements.UUID(uuid_val)
            assert c2.succ == elements.UUID(succ_val)
            assert c2.length == int(length_val)

    def test_load_error_raise(self):
        """Test load raises on corrupted file with error='raise'"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted file
            bad_file = elements.Path(tmpdir) / "123-uuid1-uuid2-5.npz"
            bad_file.write(b"corrupted data", mode="wb")

            with pytest.raises((ValueError, OSError)):
                Chunk.load(bad_file, error="raise")

    def test_load_error_none(self, capsys):
        """Test load returns None on error with error='none'"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = elements.Path(tmpdir) / "123-uuid1-uuid2-5.npz"
            bad_file.write(b"corrupted data", mode="wb")

            result = Chunk.load(bad_file, error="none")

            assert result is None

            # Should print error message
            captured = capsys.readouterr()
            assert "Error loading chunk" in captured.out

    def test_load_error_validation(self):
        """Test load validates error parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            c = Chunk(size=10)
            c.append({"x": np.array([1])})
            c.save(tmpdir)

            filepath = elements.Path(tmpdir) / c.filename

            # Should assert on invalid error value
            with pytest.raises(AssertionError):
                Chunk.load(filepath, error="invalid")

    def test_round_trip_preserves_data(self):
        """Test save then load preserves all data exactly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chunk with varied data types
            c1 = Chunk(size=10)
            np.random.seed(42)  # For reproducibility
            for i in range(5):
                c1.append(
                    {
                        "obs": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                        "action": np.random.randn(12).astype(np.float32),
                        "reward": np.array([i * 0.1], dtype=np.float32),
                    }
                )

            c1.save(tmpdir)

            filepath = elements.Path(tmpdir) / c1.filename
            c2 = Chunk.load(filepath)

            # Verify exact data match
            for key in c1.data:
                assert np.array_equal(c2.data[key], c1.data[key][: c1.length])


class TestChunkEdgeCases:
    """Test boundary conditions and error cases"""

    def test_full_chunk_behavior(self):
        """Test chunk at exactly size capacity"""
        c = Chunk(size=3)

        for i in range(3):
            c.append({"x": np.array([i])})

        assert c.length == c.size
        assert c.length == 3

        # Should be able to slice full chunk
        sliced = c.slice(0, 3)
        assert len(sliced["x"]) == 3

    def test_chunk_with_multiple_keys(self):
        """Test chunk handles multiple data keys correctly"""
        c = Chunk(size=10)

        step = {
            "obs": np.zeros((64, 64, 3), dtype=np.uint8),
            "action": np.zeros(12, dtype=np.float32),
            "reward": np.array([1.5], dtype=np.float32),
            "done": np.array([False], dtype=bool),
        }

        c.append(step)

        assert len(c.data) == 4
        assert all(key in c.data for key in step.keys())

    def test_chunk_with_complex_shapes(self):
        """Test chunk handles various array shapes"""
        c = Chunk(size=5)

        step = {
            "scalar": np.array([1.0]),
            "vector": np.array([1, 2, 3]),
            "matrix": np.array([[1, 2], [3, 4]]),
            "tensor": np.zeros((4, 4, 4)),
        }

        c.append(step)

        assert c.data["scalar"].shape == (5, 1)
        assert c.data["vector"].shape == (5, 3)
        assert c.data["matrix"].shape == (5, 2, 2)
        assert c.data["tensor"].shape == (5, 4, 4, 4)

    def test_uuid_uniqueness(self):
        """Test chunk UUIDs are globally unique"""
        chunks = [Chunk() for _ in range(100)]
        uuids = [c.uuid for c in chunks]

        # All UUIDs should be unique
        assert len(set(uuids)) == 100

    def test_timestamp_ordering(self):
        """Test chunks created later have later timestamps"""
        chunks = []
        for _ in range(10):
            chunks.append(Chunk())
            time.sleep(0.002)

        # Timestamps should be monotonically increasing
        for i in range(len(chunks) - 1):
            assert chunks[i].time < chunks[i + 1].time
