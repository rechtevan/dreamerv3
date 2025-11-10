"""
Tests for embodied.core.clock - Time-based trigger mechanisms

Coverage goal: 90% (from 13.68%)

Tests cover:
- LocalClock: Time-based triggering logic
- GlobalClock: Single-host mode (without CLIENT)
- Special modes: off (0), always (-1), regular intervals
- first parameter handling
- skip parameter handling

Note: GlobalClock multihost mode and setup() require portal infrastructure
and are better tested through integration tests.
"""

import time
from unittest import mock

import pytest

from embodied.core import clock


class TestLocalClock:
    """Test LocalClock time-based triggering"""

    def test_init_default(self):
        """Test LocalClock initializes with defaults"""
        c = clock.LocalClock(every=5.0)

        assert c.every == 5.0
        assert c.prev is None
        assert c.first is False

    def test_init_with_first(self):
        """Test LocalClock initializes with first=True"""
        c = clock.LocalClock(every=5.0, first=True)

        assert c.first is True

    def test_first_call_without_first_flag(self):
        """Test first call returns False when first=False"""
        c = clock.LocalClock(every=5.0, first=False)

        result = c()

        assert result is False
        assert c.prev is not None

    def test_first_call_with_first_flag(self):
        """Test first call returns True when first=True"""
        c = clock.LocalClock(every=5.0, first=True)

        result = c()

        assert result is True
        assert c.prev is not None

    def test_every_zero_always_false(self):
        """Test every=0 means off (always returns False)"""
        c = clock.LocalClock(every=0)

        # Multiple calls should all return False
        assert c() is False
        assert c() is False
        assert c() is False

    def test_every_negative_always_true(self):
        """Test every<0 means always (always returns True)"""
        c = clock.LocalClock(every=-1)

        # Multiple calls should all return True
        assert c() is True
        assert c() is True
        assert c() is True

    def test_skip_parameter(self):
        """Test skip=True always returns False"""
        c = clock.LocalClock(every=-1)  # Would normally always return True

        result = c(skip=True)

        assert result is False

    def test_triggers_after_interval(self):
        """Test clock triggers after specified interval"""
        c = clock.LocalClock(every=0.1, first=False)

        # First call sets prev
        assert c() is False

        # Before interval elapses
        time.sleep(0.05)
        assert c() is False

        # After interval elapses
        time.sleep(0.1)
        assert c() is True

    def test_does_not_trigger_before_interval(self):
        """Test clock does not trigger before interval"""
        c = clock.LocalClock(every=1.0, first=False)

        # First call
        assert c() is False

        # Immediate second call
        assert c() is False

    def test_resets_prev_after_trigger(self):
        """Test clock resets prev time after triggering"""
        c = clock.LocalClock(every=0.1, first=True)

        # First call triggers
        first_time = time.time()
        assert c() is True
        first_prev = c.prev

        # Wait and trigger again
        time.sleep(0.15)
        second_time = time.time()
        assert c() is True
        second_prev = c.prev

        # prev should have been updated
        assert second_prev > first_prev
        assert second_prev >= first_prev + 0.1

    def test_multiple_intervals(self):
        """Test clock works across multiple intervals"""
        c = clock.LocalClock(every=0.1, first=False)

        # First call
        assert c() is False

        # Trigger first time
        time.sleep(0.12)
        assert c() is True

        # No trigger immediately after
        assert c() is False

        # Trigger second time
        time.sleep(0.12)
        assert c() is True


class TestGlobalClockSingleHost:
    """Test GlobalClock in single-host mode (no CLIENT)"""

    def test_init_without_client(self):
        """Test GlobalClock initializes in single-host mode"""
        # Ensure CLIENT is None
        original_client = clock.CLIENT
        clock.CLIENT = None

        c = clock.GlobalClock(every=5.0)

        assert c.multihost is False
        assert hasattr(c, "clock")
        assert isinstance(c.clock, clock.LocalClock)

        # Restore
        clock.CLIENT = original_client

    def test_call_without_client(self):
        """Test GlobalClock delegates to LocalClock in single-host mode"""
        # Ensure CLIENT is None
        original_client = clock.CLIENT
        clock.CLIENT = None

        c = clock.GlobalClock(every=-1)
        result = c()

        assert result is True

        # Restore
        clock.CLIENT = original_client

    def test_first_parameter_without_client(self):
        """Test GlobalClock passes first parameter to LocalClock"""
        original_client = clock.CLIENT
        clock.CLIENT = None

        c = clock.GlobalClock(every=5.0, first=True)
        result = c()

        # First call should trigger with first=True
        assert result is True

        clock.CLIENT = original_client

    def test_skip_parameter_without_client(self):
        """Test GlobalClock passes skip parameter to LocalClock"""
        original_client = clock.CLIENT
        clock.CLIENT = None

        c = clock.GlobalClock(every=-1)  # Would normally always trigger
        result = c(skip=True)

        assert result is False

        clock.CLIENT = original_client

    def test_every_zero_without_client(self):
        """Test GlobalClock handles every=0 (off) in single-host mode"""
        original_client = clock.CLIENT
        clock.CLIENT = None

        c = clock.GlobalClock(every=0)
        result = c()

        assert result is False

        clock.CLIENT = original_client

    def test_every_negative_without_client(self):
        """Test GlobalClock handles every<0 (always) in single-host mode"""
        original_client = clock.CLIENT
        clock.CLIENT = None

        c = clock.GlobalClock(every=-1)
        result = c()

        assert result is True

        clock.CLIENT = original_client


class TestGlobalClockMultihost:
    """Test GlobalClock multihost mode behavior"""

    def test_init_with_client(self):
        """Test GlobalClock initializes in multihost mode"""
        original_client = clock.CLIENT
        original_replica = clock.REPLICA

        # Mock CLIENT
        mock_client = mock.Mock()
        mock_future = mock.Mock()
        mock_future.result.return_value = 42
        mock_client.create.return_value = mock_future
        clock.CLIENT = mock_client
        clock.REPLICA = 0

        c = clock.GlobalClock(every=5.0)

        assert c.multihost is True
        assert c.clockid == 42
        mock_client.create.assert_called_once_with(0, 5.0)

        # Restore
        clock.CLIENT = original_client
        clock.REPLICA = original_replica

    def test_init_with_first_in_multihost(self):
        """Test GlobalClock first parameter in multihost mode"""
        original_client = clock.CLIENT
        original_replica = clock.REPLICA

        mock_client = mock.Mock()
        mock_future = mock.Mock()
        mock_future.result.return_value = 42
        mock_client.create.return_value = mock_future
        clock.CLIENT = mock_client
        clock.REPLICA = 0

        c = clock.GlobalClock(every=5.0, first=True)

        # skip_next should be False with first=True
        assert c.skip_next is False

        clock.CLIENT = original_client
        clock.REPLICA = original_replica

    def test_call_with_client(self):
        """Test GlobalClock calls remote should method"""
        original_client = clock.CLIENT
        original_replica = clock.REPLICA

        mock_client = mock.Mock()
        mock_create_future = mock.Mock()
        mock_create_future.result.return_value = 42
        mock_should_future = mock.Mock()
        mock_should_future.result.return_value = True
        mock_client.create.return_value = mock_create_future
        mock_client.should.return_value = mock_should_future
        clock.CLIENT = mock_client
        clock.REPLICA = 0

        # Create with first=True so skip_next=False
        c = clock.GlobalClock(every=5.0, first=True)
        result = c()

        assert result is True
        mock_client.should.assert_called_once_with(0, 42, False)

        clock.CLIENT = original_client
        clock.REPLICA = original_replica

    def test_skip_next_logic_in_multihost(self):
        """Test GlobalClock skip_next logic in multihost mode"""
        original_client = clock.CLIENT
        original_replica = clock.REPLICA

        mock_client = mock.Mock()
        mock_create_future = mock.Mock()
        mock_create_future.result.return_value = 42
        mock_should_future = mock.Mock()
        mock_should_future.result.return_value = False
        mock_client.create.return_value = mock_create_future
        mock_client.should.return_value = mock_should_future
        clock.CLIENT = mock_client
        clock.REPLICA = 0

        # Create with first=False (skip_next=True)
        c = clock.GlobalClock(every=5.0, first=False)
        assert c.skip_next is True

        # First call should set skip=True and reset skip_next
        result = c()
        mock_client.should.assert_called_with(0, 42, True)
        assert c.skip_next is False

        # Second call should set skip=False
        c()
        mock_client.should.assert_called_with(0, 42, False)

        clock.CLIENT = original_client
        clock.REPLICA = original_replica


class TestSetup:
    """Test setup function (integration-level, basic validation only)"""

    def test_setup_skips_when_replicas_le_1(self):
        """Test setup does nothing when replicas <= 1"""
        original_client = clock.CLIENT
        clock.CLIENT = None

        # Should not raise any errors
        clock.setup(
            is_server=False, replica=0, replicas=1, port=12345, addr="localhost:12345"
        )

        # CLIENT should still be None
        assert clock.CLIENT is None

        clock.CLIENT = original_client

    def test_setup_asserts_client_none(self):
        """Test setup asserts CLIENT is None"""
        original_client = clock.CLIENT
        clock.CLIENT = "not_none"

        with pytest.raises(AssertionError):
            clock.setup(
                is_server=False,
                replica=0,
                replicas=2,
                port=12345,
                addr="localhost:12345",
            )

        clock.CLIENT = original_client


# Note: _start_server is integration-level code that requires portal infrastructure
# and threading synchronization. It's better tested through integration tests with
# actual portal servers and multiple clients.
