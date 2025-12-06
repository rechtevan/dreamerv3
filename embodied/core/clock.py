"""Clock utilities for time-based and rate-limiting operations.

This module provides clock abstractions for controlling the frequency of periodic
operations in both single-process and multi-host distributed training scenarios.

Classes:
    GlobalClock: Synchronized clock for multi-host distributed training that
        coordinates timing decisions across multiple replicas using a central server.
    LocalClock: Simple time-based clock for single-process rate limiting without
        synchronization overhead.

The GlobalClock automatically falls back to LocalClock when running in single-host
mode (replicas <= 1), providing a unified interface for both scenarios.

Typical usage:
    # Create a clock that triggers every 60 seconds
    save_clock = GlobalClock(every=60.0)

    # Check if it's time to perform the action
    if save_clock():
        save_checkpoint()
"""

import threading
import time
import typing

import portal


CLIENT = None
REPLICA = None


def setup(is_server, replica, replicas, port, addr):
    """Initialize global clock infrastructure for multi-host training.

    Sets up the client-server architecture for synchronized clocks across replicas.
    When replicas > 1, starts the clock server (if is_server=True) and connects
    all clients to enable distributed clock synchronization.

    Args:
        is_server: Whether this process should start the clock server.
        replica: The replica ID for this process (0-indexed).
        replicas: Total number of replicas in the distributed setup.
        port: Port number for the clock server.
        addr: Server address for client connections.
    """
    global CLIENT, REPLICA
    assert CLIENT is None
    if replicas <= 1:
        return
    print("CLOCK PORT:", port)
    print("CLOCK ADDR:", addr)
    if is_server:
        _start_server(port, replicas)
    client = portal.Client(addr, "ClockClient")
    client.connect()
    CLIENT = client
    REPLICA = replica


def _start_server(port, replicas):
    """Start the central clock synchronization server.

    Creates a portal server that coordinates timing decisions across all replicas.
    The server maintains clock state and ensures all replicas make synchronized
    decisions about when periodic operations should execute.

    Args:
        port: Port number to bind the server to.
        replicas: Total number of client replicas that will connect.
    """
    clocks: list[list[float]] = []
    requests: list[typing.Any] = []
    result: list[int | None] = [None]
    receive = threading.Barrier(replicas)
    respond = threading.Barrier(replicas)

    def create(replica, every):
        """Create a new synchronized clock across all replicas.

        Args:
            replica: The replica ID making the request.
            every: Time interval in seconds between clock triggers.

        Returns:
            Clock ID for future synchronization requests.
        """
        requests.append(every)
        receive.wait()
        if replica == 0:
            assert len(requests) == replicas, (len(requests), replicas)
            assert all(x == every for x in requests)
            clockid = len(clocks)
            clocks.append([float(every), time.time()])
            result[0] = clockid
            requests.clear()
        respond.wait()
        return result[0]

    def should(replica, clockid, skip):
        """Check if the clock should trigger for this timestep.

        Synchronizes the decision across all replicas, ensuring they all agree
        on whether the periodic operation should execute.

        Args:
            replica: The replica ID making the request.
            clockid: The clock ID to check.
            skip: Whether this replica wants to skip execution.

        Returns:
            True if the clock should trigger, False otherwise.
        """
        requests.append((clockid, skip))
        receive.wait()
        if replica == 0:
            assert len(requests) == replicas, (len(requests), replicas)
            clockids, skips = zip(*requests)
            assert all(x == clockid for x in clockids)
            every, prev = clocks[clockid]
            now = time.time()
            if every == 0:
                decision = False
            elif every < 0:
                decision = True
            elif now >= prev + every:
                clocks[clockid][1] = now
                decision = True
            else:
                decision = False
            decision = decision and not any(skips)
            result[0] = decision
            requests.clear()
        respond.wait()
        return result[0]

    server = portal.Server(port, "ClockServer")
    server.bind("create", create, workers=replicas)
    server.bind("should", should, workers=replicas)
    server.start(block=False)


class GlobalClock:
    """Synchronized clock for multi-host distributed training.

    Coordinates timing decisions across multiple replicas using a central server.
    Automatically falls back to LocalClock when running in single-host mode.
    All replicas must agree on whether a periodic operation should execute.

    The clock supports three modes via the 'every' parameter:
        - every > 0: Trigger at the specified time interval (seconds)
        - every < 0: Always trigger (every call returns True)
        - every = 0: Never trigger (every call returns False)

    Attributes:
        multihost: Whether multi-host synchronization is active.
        clockid: The server-assigned clock ID (multi-host mode only).
        skip_next: Whether to skip the first trigger (multi-host mode only).
        clock: The fallback LocalClock instance (single-host mode only).

    Example:
        # Create a clock that triggers every 300 seconds
        checkpoint_clock = GlobalClock(every=300.0)

        # In training loop
        if checkpoint_clock():
            save_checkpoint()
    """

    def __init__(self, every, first=False):
        """Initialize a GlobalClock.

        Args:
            every: Time interval in seconds between triggers. Use negative values
                to always trigger, or 0 to never trigger.
            first: Whether to trigger on the very first call. Defaults to False.
        """
        self.multihost = bool(CLIENT)
        if self.multihost:
            assert CLIENT is not None, "CLIENT must be initialized in multihost mode"
            self.clockid = CLIENT.create(REPLICA, every).result()
            self.skip_next = not first
        else:
            self.clock = LocalClock(every, first)

    def __call__(self, step=None, skip=None):
        """Check if the clock should trigger for this timestep.

        In multi-host mode, synchronizes the decision across all replicas via the
        central server. In single-host mode, uses local time tracking.

        Args:
            step: Training step number (reserved for future use, currently unused).
            skip: Whether to skip triggering regardless of timing. If True, the
                clock will not trigger even if the time interval has elapsed.

        Returns:
            True if the clock should trigger (time interval elapsed and not skipped),
            False otherwise.
        """
        if self.multihost:
            assert CLIENT is not None, "CLIENT must be initialized in multihost mode"
            if self.skip_next:
                self.skip_next = False
                skip = True
            return CLIENT.should(REPLICA, self.clockid, bool(skip)).result()
        else:
            return self.clock(step, skip)


class LocalClock:
    """Simple time-based clock for single-process rate limiting.

    Tracks time intervals locally without synchronization overhead, suitable for
    single-host training or when independent timing decisions are acceptable.

    The clock supports three modes via the 'every' parameter:
        - every > 0: Trigger at the specified time interval (seconds)
        - every < 0: Always trigger (every call returns True)
        - every = 0: Never trigger (every call returns False)

    Attributes:
        every: Time interval in seconds between triggers.
        prev: Timestamp of the previous trigger (None until first call).
        first: Whether to trigger on the very first call.

    Example:
        # Create a clock that triggers every 10 seconds
        log_clock = LocalClock(every=10.0, first=True)

        # In training loop
        if log_clock():
            log_metrics()
    """

    def __init__(self, every, first=False):
        """Initialize a LocalClock.

        Args:
            every: Time interval in seconds between triggers. Use negative values
                to always trigger, or 0 to never trigger.
            first: Whether to trigger on the very first call. Defaults to False.
        """
        self.every = every
        self.prev = None
        self.first = first

    def __call__(self, step=None, skip=None):
        """Check if the clock should trigger for this timestep.

        Uses local wall-clock time to determine if the specified interval has elapsed
        since the last trigger.

        Args:
            step: Training step number (reserved for future use, currently unused).
            skip: Whether to skip triggering regardless of timing. If True, the
                clock will not trigger even if the time interval has elapsed.

        Returns:
            True if the clock should trigger (time interval elapsed and not skipped),
            False otherwise.
        """
        if skip:
            return False
        if self.every == 0:  # Zero means off
            return False
        if self.every < 0:  # Negative means always
            return True
        now = time.time()
        if self.prev is None:
            self.prev = now
            return self.first
        if now >= self.prev + self.every:
            self.prev = now
            return True
        return False
