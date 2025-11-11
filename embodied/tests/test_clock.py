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

    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock._start_server")
    def test_setup_multihost_as_server(self, mock_start_server, mock_portal):
        """Test setup initializes server in multihost mode"""
        original_client = clock.CLIENT
        original_replica = clock.REPLICA
        clock.CLIENT = None
        clock.REPLICA = None

        # Mock portal client
        mock_client_instance = mock.Mock()
        mock_portal.Client.return_value = mock_client_instance

        # Call setup as server
        clock.setup(
            is_server=True,
            replica=0,
            replicas=2,
            port=12345,
            addr="localhost:12345",
        )

        # Verify _start_server was called
        mock_start_server.assert_called_once_with(12345, 2)

        # Verify client created and connected
        mock_portal.Client.assert_called_once_with("localhost:12345", "ClockClient")
        mock_client_instance.connect.assert_called_once()

        # Verify globals set
        assert mock_client_instance == clock.CLIENT
        assert clock.REPLICA == 0

        clock.CLIENT = original_client
        clock.REPLICA = original_replica

    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock._start_server")
    def test_setup_multihost_as_client(self, mock_start_server, mock_portal):
        """Test setup does not start server when not is_server"""
        original_client = clock.CLIENT
        original_replica = clock.REPLICA
        clock.CLIENT = None
        clock.REPLICA = None

        # Mock portal client
        mock_client_instance = mock.Mock()
        mock_portal.Client.return_value = mock_client_instance

        # Call setup as non-server
        clock.setup(
            is_server=False,
            replica=1,
            replicas=2,
            port=12345,
            addr="localhost:12345",
        )

        # Verify _start_server was NOT called
        mock_start_server.assert_not_called()

        # Verify client created and connected
        mock_portal.Client.assert_called_once_with("localhost:12345", "ClockClient")
        mock_client_instance.connect.assert_called_once()

        # Verify globals set
        assert mock_client_instance == clock.CLIENT
        assert clock.REPLICA == 1

        clock.CLIENT = original_client
        clock.REPLICA = original_replica

    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock._start_server")
    def test_setup_prints_connection_info(self, mock_start_server, mock_portal, capsys):
        """Test setup prints port and address information"""
        original_client = clock.CLIENT
        original_replica = clock.REPLICA
        clock.CLIENT = None
        clock.REPLICA = None

        # Mock portal client
        mock_client_instance = mock.Mock()
        mock_portal.Client.return_value = mock_client_instance

        # Call setup
        clock.setup(
            is_server=False,
            replica=0,
            replicas=2,
            port=12345,
            addr="localhost:12345",
        )

        # Check stdout for debug prints
        captured = capsys.readouterr()
        assert "CLOCK PORT: 12345" in captured.out
        assert "CLOCK ADDR: localhost:12345" in captured.out

        clock.CLIENT = original_client
        clock.REPLICA = original_replica


class TestStartServer:
    """Test _start_server function (mock-based testing)"""

    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock.threading.Barrier")
    def test_start_server_creates_portal_server(self, mock_barrier, mock_portal):
        """Test _start_server creates and configures portal server"""
        # Mock server
        mock_server_instance = mock.Mock()
        mock_portal.Server.return_value = mock_server_instance

        # Call _start_server
        clock._start_server(port=12345, replicas=2)

        # Verify server created
        mock_portal.Server.assert_called_once_with(12345, "ClockServer")

        # Verify server methods bound
        assert mock_server_instance.bind.call_count == 2
        bind_calls = mock_server_instance.bind.call_args_list

        # Check create binding
        assert bind_calls[0][0][0] == "create"
        assert bind_calls[0][1]["workers"] == 2

        # Check should binding
        assert bind_calls[1][0][0] == "should"
        assert bind_calls[1][1]["workers"] == 2

        # Verify server started
        mock_server_instance.start.assert_called_once_with(block=False)

        # Verify barriers created (2 barriers for replicas=2)
        assert mock_barrier.call_count == 2
        mock_barrier.assert_any_call(2)

    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock.threading.Barrier")
    def test_start_server_with_different_replicas(self, mock_barrier, mock_portal):
        """Test _start_server handles different replica counts"""
        mock_server_instance = mock.Mock()
        mock_portal.Server.return_value = mock_server_instance

        # Test with 4 replicas
        clock._start_server(port=54321, replicas=4)

        # Verify correct replica count used
        bind_calls = mock_server_instance.bind.call_args_list
        assert bind_calls[0][1]["workers"] == 4
        assert bind_calls[1][1]["workers"] == 4

        # Verify barriers created with correct count
        mock_barrier.assert_any_call(4)

    @mock.patch("embodied.core.clock.time")
    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock.threading.Barrier")
    def test_start_server_create_handler(self, mock_barrier, mock_portal, mock_time):
        """Test _start_server create handler function"""
        # Mock server
        mock_server_instance = mock.Mock()
        mock_portal.Server.return_value = mock_server_instance

        # Mock barriers - do nothing (no actual synchronization in test)
        mock_receive_barrier = mock.Mock()
        mock_respond_barrier = mock.Mock()
        mock_barrier.side_effect = [mock_receive_barrier, mock_respond_barrier]

        # Mock time
        mock_time.time.return_value = 1000.0

        # Call _start_server to capture the create handler
        clock._start_server(port=12345, replicas=2)

        # Get the create handler from the first bind call
        bind_calls = mock_server_instance.bind.call_args_list
        create_handler = bind_calls[0][0][1]

        # Simulate both replicas calling create with same every value
        # Replica 1 calls first
        create_handler(replica=1, every=5.0)
        # Replica 0 calls (coordinator)
        result = create_handler(replica=0, every=5.0)

        # Should wait on barriers
        assert mock_receive_barrier.wait.call_count >= 1
        assert mock_respond_barrier.wait.call_count >= 1

        # Should return clockid 0 (first clock)
        assert result == 0

        # Test creating another clock (both replicas call again)
        create_handler(replica=1, every=10.0)
        result2 = create_handler(replica=0, every=10.0)
        assert result2 == 1

    @mock.patch("embodied.core.clock.time")
    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock.threading.Barrier")
    def test_start_server_should_handler_every_zero(
        self, mock_barrier, mock_portal, mock_time
    ):
        """Test _start_server should handler with every=0 (off)"""
        mock_server_instance = mock.Mock()
        mock_portal.Server.return_value = mock_server_instance

        mock_receive_barrier = mock.Mock()
        mock_respond_barrier = mock.Mock()
        mock_barrier.side_effect = [mock_receive_barrier, mock_respond_barrier]

        mock_time.time.return_value = 1000.0

        clock._start_server(port=12345, replicas=2)

        bind_calls = mock_server_instance.bind.call_args_list
        create_handler = bind_calls[0][0][1]
        should_handler = bind_calls[1][0][1]

        # Create clock with every=0 (both replicas)
        create_handler(replica=1, every=0)
        clockid = create_handler(replica=0, every=0)

        # Test should with every=0 (both replicas, should always return False)
        should_handler(replica=1, clockid=clockid, skip=False)
        result = should_handler(replica=0, clockid=clockid, skip=False)
        assert result is False

    @mock.patch("embodied.core.clock.time")
    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock.threading.Barrier")
    def test_start_server_should_handler_every_negative(
        self, mock_barrier, mock_portal, mock_time
    ):
        """Test _start_server should handler with every<0 (always)"""
        mock_server_instance = mock.Mock()
        mock_portal.Server.return_value = mock_server_instance

        mock_receive_barrier = mock.Mock()
        mock_respond_barrier = mock.Mock()
        mock_barrier.side_effect = [mock_receive_barrier, mock_respond_barrier]

        mock_time.time.return_value = 1000.0

        clock._start_server(port=12345, replicas=2)

        bind_calls = mock_server_instance.bind.call_args_list
        create_handler = bind_calls[0][0][1]
        should_handler = bind_calls[1][0][1]

        # Create clock with every=-1 (both replicas)
        create_handler(replica=1, every=-1)
        clockid = create_handler(replica=0, every=-1)

        # Test should with every=-1 (both replicas, should always return True if not skipped)
        should_handler(replica=1, clockid=clockid, skip=False)
        result = should_handler(replica=0, clockid=clockid, skip=False)
        assert result is True

    @mock.patch("embodied.core.clock.time")
    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock.threading.Barrier")
    def test_start_server_should_handler_skip(
        self, mock_barrier, mock_portal, mock_time
    ):
        """Test _start_server should handler respects skip parameter"""
        mock_server_instance = mock.Mock()
        mock_portal.Server.return_value = mock_server_instance

        mock_receive_barrier = mock.Mock()
        mock_respond_barrier = mock.Mock()
        mock_barrier.side_effect = [mock_receive_barrier, mock_respond_barrier]

        mock_time.time.return_value = 1000.0

        clock._start_server(port=12345, replicas=2)

        bind_calls = mock_server_instance.bind.call_args_list
        create_handler = bind_calls[0][0][1]
        should_handler = bind_calls[1][0][1]

        # Create clock with every=-1 (both replicas, would normally always trigger)
        create_handler(replica=1, every=-1)
        clockid = create_handler(replica=0, every=-1)

        # Test should with skip=True (both replicas, should return False even with every=-1)
        should_handler(replica=1, clockid=clockid, skip=True)
        result = should_handler(replica=0, clockid=clockid, skip=True)
        assert result is False

    @mock.patch("embodied.core.clock.time")
    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock.threading.Barrier")
    def test_start_server_should_handler_time_elapsed(
        self, mock_barrier, mock_portal, mock_time
    ):
        """Test _start_server should handler triggers after time elapsed"""
        mock_server_instance = mock.Mock()
        mock_portal.Server.return_value = mock_server_instance

        mock_receive_barrier = mock.Mock()
        mock_respond_barrier = mock.Mock()
        mock_barrier.side_effect = [mock_receive_barrier, mock_respond_barrier]

        # Start at time 1000
        mock_time.time.return_value = 1000.0

        clock._start_server(port=12345, replicas=2)

        bind_calls = mock_server_instance.bind.call_args_list
        create_handler = bind_calls[0][0][1]
        should_handler = bind_calls[1][0][1]

        # Create clock with every=5.0 (both replicas)
        create_handler(replica=1, every=5.0)
        clockid = create_handler(replica=0, every=5.0)

        # First call at 1002 (both replicas), not enough time elapsed
        mock_time.time.return_value = 1002.0
        should_handler(replica=1, clockid=clockid, skip=False)
        result = should_handler(replica=0, clockid=clockid, skip=False)
        assert result is False

        # Second call at 1005 (both replicas), enough time elapsed
        mock_time.time.return_value = 1005.0
        should_handler(replica=1, clockid=clockid, skip=False)
        result = should_handler(replica=0, clockid=clockid, skip=False)
        assert result is True

    @mock.patch("embodied.core.clock.time")
    @mock.patch("embodied.core.clock.portal")
    @mock.patch("embodied.core.clock.threading.Barrier")
    def test_start_server_should_handler_updates_prev_time(
        self, mock_barrier, mock_portal, mock_time
    ):
        """Test _start_server should handler updates prev time after trigger"""
        mock_server_instance = mock.Mock()
        mock_portal.Server.return_value = mock_server_instance

        mock_receive_barrier = mock.Mock()
        mock_respond_barrier = mock.Mock()
        mock_barrier.side_effect = [mock_receive_barrier, mock_respond_barrier]

        mock_time.time.return_value = 1000.0

        clock._start_server(port=12345, replicas=2)

        bind_calls = mock_server_instance.bind.call_args_list
        create_handler = bind_calls[0][0][1]
        should_handler = bind_calls[1][0][1]

        # Create clock with every=5.0 (both replicas)
        create_handler(replica=1, every=5.0)
        clockid = create_handler(replica=0, every=5.0)

        # Trigger at 1005 (both replicas)
        mock_time.time.return_value = 1005.0
        should_handler(replica=1, clockid=clockid, skip=False)
        result = should_handler(replica=0, clockid=clockid, skip=False)
        assert result is True

        # Immediately check again (both replicas) - should not trigger (prev was updated)
        mock_time.time.return_value = 1006.0
        should_handler(replica=1, clockid=clockid, skip=False)
        result = should_handler(replica=0, clockid=clockid, skip=False)
        assert result is False

        # Trigger again after interval (both replicas)
        mock_time.time.return_value = 1010.0
        should_handler(replica=1, clockid=clockid, skip=False)
        result = should_handler(replica=0, clockid=clockid, skip=False)
        assert result is True


class TestGlobalClockMultihostSkipParameter:
    """Test GlobalClock multihost mode skip parameter handling"""

    def test_explicit_skip_parameter_in_multihost(self):
        """Test explicit skip parameter overrides skip_next"""
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

        # Create with first=True (skip_next=False)
        c = clock.GlobalClock(every=5.0, first=True)

        # Call with explicit skip=True
        result = c(skip=True)

        # Should pass skip=True to remote
        mock_client.should.assert_called_with(0, 42, True)

        clock.CLIENT = original_client
        clock.REPLICA = original_replica

    def test_skip_parameter_false_in_multihost(self):
        """Test explicit skip=False parameter"""
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

        # Create with first=False (skip_next=True)
        c = clock.GlobalClock(every=5.0, first=False)
        assert c.skip_next is True

        # First call should still set skip=True due to skip_next
        c()
        mock_client.should.assert_called_with(0, 42, True)

        # Second call with explicit skip=False
        c(skip=False)
        mock_client.should.assert_called_with(0, 42, False)

        clock.CLIENT = original_client
        clock.REPLICA = original_replica


class TestLocalClockStepParameter:
    """Test LocalClock step parameter (currently unused but part of API)"""

    def test_step_parameter_accepted(self):
        """Test LocalClock accepts step parameter without error"""
        c = clock.LocalClock(every=-1)

        # Should not raise error with step parameter
        result = c(step=100)
        assert result is True

    def test_step_and_skip_parameters(self):
        """Test step and skip parameters together"""
        c = clock.LocalClock(every=-1)

        # skip should take precedence
        result = c(step=100, skip=True)
        assert result is False


class TestGlobalClockStepParameter:
    """Test GlobalClock step parameter in single-host mode"""

    def test_step_parameter_without_client(self):
        """Test GlobalClock passes step parameter to LocalClock"""
        original_client = clock.CLIENT
        clock.CLIENT = None

        c = clock.GlobalClock(every=-1)

        # Should not raise error with step parameter
        result = c(step=100)
        assert result is True

        clock.CLIENT = original_client

    def test_step_parameter_with_client(self):
        """Test GlobalClock accepts step parameter in multihost mode"""
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

        c = clock.GlobalClock(every=5.0, first=True)

        # Should not raise error with step parameter
        result = c(step=100)
        assert result is True

        clock.CLIENT = original_client
        clock.REPLICA = original_replica


# Note: _start_server is integration-level code that requires portal infrastructure
# and threading synchronization. The create() and should() helper functions inside
# _start_server are tested through the mock-based tests above which verify they
# are properly bound to the portal server.
