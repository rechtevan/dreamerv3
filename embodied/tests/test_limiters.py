"""
Tests for embodied.core.limiters - Rate limiting utilities

Coverage goal: 90% (from 20.75%)

Tests cover:
- wait() function: predicate waiting with notifications
- SamplesPerInsert: Rate limiting for insert/sample operations
- save/load functionality
- want_insert/want_sample logic
- Thread safety
"""

import threading
import time
from unittest import mock

import pytest

from embodied.core import limiters


class TestWaitFunction:
    """Test wait utility function"""

    def test_wait_returns_immediately_if_predicate_true(self):
        """Test wait returns 0 if predicate is already true"""
        predicate = lambda: True

        duration = limiters.wait(predicate, "Test message")

        assert duration == 0

    def test_wait_blocks_until_predicate_true(self):
        """Test wait blocks until predicate becomes true"""
        flag = [False]
        predicate = lambda: flag[0]

        def set_flag_after_delay():
            time.sleep(0.1)
            flag[0] = True

        thread = threading.Thread(target=set_flag_after_delay)
        thread.start()

        start = time.time()
        duration = limiters.wait(predicate, "Test message", sleep=0.01)
        elapsed = time.time() - start

        thread.join()
        assert duration >= 0.1
        assert elapsed >= 0.1

    def test_wait_uses_custom_sleep_interval(self):
        """Test wait respects custom sleep interval"""
        call_count = [0]

        def predicate():
            call_count[0] += 1
            return call_count[0] > 5

        duration = limiters.wait(predicate, "Test message", sleep=0.02)

        # Should have called predicate multiple times with 0.02s sleep
        assert call_count[0] > 5

    def test_wait_prints_notification_after_timeout(self):
        """Test wait prints notification after notify interval"""
        flag = [False]
        predicate = lambda: flag[0]

        def set_flag_after_delay():
            time.sleep(0.15)
            flag[0] = True

        thread = threading.Thread(target=set_flag_after_delay)
        thread.start()

        with mock.patch("builtins.print") as mock_print:
            limiters.wait(
                predicate, "Waiting", info="test_info", sleep=0.01, notify=0.1
            )

        thread.join()
        # Should have printed notification after 0.1s
        assert mock_print.call_count >= 1

    def test_wait_returns_elapsed_time(self):
        """Test wait returns the time spent waiting"""
        flag = [False]
        predicate = lambda: flag[0]

        def set_flag_after_delay():
            time.sleep(0.1)
            flag[0] = True

        thread = threading.Thread(target=set_flag_after_delay)
        thread.start()

        duration = limiters.wait(predicate, "Test message")

        thread.join()
        assert duration >= 0.1
        assert duration < 0.2  # Should not take too long


class TestSamplesPerInsertInit:
    """Test SamplesPerInsert initialization"""

    def test_init_basic(self):
        """Test SamplesPerInsert initializes correctly"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=2.0, tolerance=1.0, minsize=10
        )

        assert limiter.samples_per_insert == 2.0
        assert limiter.minsize == 10
        assert limiter.avail == -10  # -minsize
        assert limiter.min_avail == -1.0  # -tolerance
        assert limiter.max_avail == 2.0  # tolerance * samples_per_insert
        assert limiter.size == 0

    def test_init_asserts_minsize_ge_1(self):
        """Test SamplesPerInsert asserts minsize >= 1"""
        with pytest.raises(AssertionError):
            limiters.SamplesPerInsert(samples_per_insert=1.0, tolerance=1.0, minsize=0)

    def test_init_has_lock(self):
        """Test SamplesPerInsert initializes with threading lock"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=1.0, minsize=5
        )

        assert hasattr(limiter, "lock")
        # Check it has lock methods
        assert hasattr(limiter.lock, "acquire")
        assert hasattr(limiter.lock, "release")


class TestSamplesPerInsertSaveLoad:
    """Test SamplesPerInsert save/load functionality"""

    def test_save(self):
        """Test save returns size and avail"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=1.0, minsize=5
        )
        limiter.size = 10
        limiter.avail = 5

        data = limiter.save()

        assert data == {"size": 10, "avail": 5}

    def test_load(self):
        """Test load restores size and avail"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=1.0, minsize=5
        )

        limiter.load({"size": 20, "avail": 10})

        assert limiter.size == 20
        assert limiter.avail == 10


class TestSamplesPerInsertWantInsert:
    """Test SamplesPerInsert want_insert logic"""

    def test_want_insert_when_below_minsize(self):
        """Test want_insert returns True when size < minsize"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=1.0, minsize=10
        )
        limiter.size = 5

        assert limiter.want_insert() is True

    def test_want_insert_when_samples_per_insert_zero(self):
        """Test want_insert returns True when samples_per_insert <= 0"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=0, tolerance=1.0, minsize=10
        )
        limiter.size = 15

        assert limiter.want_insert() is True

    def test_want_insert_when_samples_per_insert_negative(self):
        """Test want_insert returns True when samples_per_insert < 0"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=-1.0, tolerance=1.0, minsize=10
        )
        limiter.size = 15

        assert limiter.want_insert() is True

    def test_want_insert_when_below_max_avail(self):
        """Test want_insert returns True when avail < max_avail"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=2.0, tolerance=1.0, minsize=10
        )
        limiter.size = 15
        limiter.avail = 1.0  # Below max_avail (2.0)

        assert limiter.want_insert() is True

    def test_want_insert_when_at_max_avail(self):
        """Test want_insert returns False when avail >= max_avail"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=2.0, tolerance=1.0, minsize=10
        )
        limiter.size = 15
        limiter.avail = 2.0  # At max_avail

        assert limiter.want_insert() is False

    def test_want_insert_when_above_max_avail(self):
        """Test want_insert returns False when avail > max_avail"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=2.0, tolerance=1.0, minsize=10
        )
        limiter.size = 15
        limiter.avail = 3.0  # Above max_avail

        assert limiter.want_insert() is False


class TestSamplesPerInsertWantSample:
    """Test SamplesPerInsert want_sample logic"""

    def test_want_sample_when_below_minsize(self):
        """Test want_sample returns False when size < minsize"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=1.0, minsize=10
        )
        limiter.size = 5

        assert limiter.want_sample() is False

    def test_want_sample_when_samples_per_insert_zero(self):
        """Test want_sample returns True when samples_per_insert <= 0"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=0, tolerance=1.0, minsize=10
        )
        limiter.size = 15

        assert limiter.want_sample() is True

    def test_want_sample_when_samples_per_insert_negative(self):
        """Test want_sample returns True when samples_per_insert < 0"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=-1.0, tolerance=1.0, minsize=10
        )
        limiter.size = 15

        assert limiter.want_sample() is True

    def test_want_sample_when_above_min_avail(self):
        """Test want_sample returns True when avail > min_avail"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=2.0, tolerance=1.0, minsize=10
        )
        limiter.size = 15
        limiter.avail = 0.0  # Above min_avail (-1.0)

        assert limiter.want_sample() is True

    def test_want_sample_when_at_min_avail(self):
        """Test want_sample returns False when avail <= min_avail"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=2.0, tolerance=1.0, minsize=10
        )
        limiter.size = 15
        limiter.avail = -1.0  # At min_avail

        assert limiter.want_sample() is False

    def test_want_sample_when_below_min_avail(self):
        """Test want_sample returns False when avail < min_avail"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=2.0, tolerance=1.0, minsize=10
        )
        limiter.size = 15
        limiter.avail = -2.0  # Below min_avail

        assert limiter.want_sample() is False


class TestSamplesPerInsertInsert:
    """Test SamplesPerInsert insert method"""

    def test_insert_increments_size(self):
        """Test insert increments size"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=1.0, minsize=10
        )

        limiter.insert()

        assert limiter.size == 1

    def test_insert_increments_avail_when_above_minsize(self):
        """Test insert increments avail when size >= minsize"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=2.0, tolerance=1.0, minsize=10
        )
        limiter.size = 10
        limiter.avail = 0

        limiter.insert()

        assert limiter.size == 11
        assert limiter.avail == 2.0  # samples_per_insert added

    def test_insert_does_not_increment_avail_when_below_minsize(self):
        """Test insert does not increment avail when size < minsize"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=2.0, tolerance=1.0, minsize=10
        )
        limiter.size = 5
        limiter.avail = -10

        limiter.insert()

        assert limiter.size == 6
        assert limiter.avail == -10  # Not changed

    def test_insert_thread_safe(self):
        """Test insert is thread-safe"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=1.0, minsize=10
        )

        def insert_many():
            for _ in range(100):
                limiter.insert()

        threads = [threading.Thread(target=insert_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have exactly 1000 inserts
        assert limiter.size == 1000


class TestSamplesPerInsertSample:
    """Test SamplesPerInsert sample method"""

    def test_sample_decrements_avail(self):
        """Test sample decrements avail"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=1.0, minsize=10
        )
        limiter.avail = 5.0

        limiter.sample()

        assert limiter.avail == 4.0

    def test_sample_thread_safe(self):
        """Test sample is thread-safe"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=1.0, minsize=10
        )
        limiter.avail = 1000.0

        def sample_many():
            for _ in range(100):
                limiter.sample()

        threads = [threading.Thread(target=sample_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have exactly 1000 samples
        assert limiter.avail == 0.0


class TestSamplesPerInsertIntegration:
    """Test SamplesPerInsert integration scenarios"""

    def test_insert_sample_workflow(self):
        """Test typical insert/sample workflow"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=2.0, tolerance=1.0, minsize=10
        )

        # Initially can't sample (below minsize)
        assert limiter.want_sample() is False

        # Insert 15 times (10 to reach minsize, 5 more to get avail > min_avail)
        # After 10: size=10, avail=-10+2.0=-8.0 (only 1 insert adds to avail)
        # After 11: avail=-8.0+2.0=-6.0
        # After 12: avail=-6.0+2.0=-4.0
        # After 13: avail=-4.0+2.0=-2.0
        # After 14: avail=-2.0+2.0=0.0
        # After 15: avail=0.0+2.0=2.0, now 2.0 > -1.0 (min_avail)
        for _ in range(15):
            limiter.insert()

        # Now can sample (avail=2.0 > min_avail=-1.0)
        assert limiter.want_sample() is True

        # Sample twice
        limiter.sample()  # avail = 1.0
        limiter.sample()  # avail = 0.0
        # Still 0.0 > -1.0, can sample
        assert limiter.want_sample() is True

    def test_rate_limiting_blocks_insert(self):
        """Test rate limiting blocks inserts when avail too high"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=0.5, minsize=5
        )

        # Insert past minsize to build up avail
        for _ in range(10):
            limiter.insert()

        # Should have built up avail = 5 * 1.0 = 5.0
        # max_avail = 0.5 * 1.0 = 0.5
        # So want_insert should be False
        assert limiter.want_insert() is False

    def test_rate_limiting_blocks_sample(self):
        """Test rate limiting blocks samples when avail too low"""
        limiter = limiters.SamplesPerInsert(
            samples_per_insert=1.0, tolerance=0.5, minsize=5
        )

        # Insert to minsize
        for _ in range(5):
            limiter.insert()

        # Sample more than we have avail
        for _ in range(10):
            limiter.sample()

        # Should have avail = 5 * 1.0 - 10 = -5.0
        # min_avail = -0.5
        # So want_sample should be False
        assert limiter.want_sample() is False
