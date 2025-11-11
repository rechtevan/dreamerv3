"""Rate limiting and synchronization utilities for replay buffers.

This module provides utilities for controlling the rate of inserts and samples
in replay buffers, ensuring balanced data collection and training. The main
component is SamplesPerInsert, which maintains a desired ratio between sampling
and insertion operations to prevent buffer overflow or starvation.
"""

import threading
import time


def wait(predicate, message, info=None, sleep=0.01, notify=60):
    """Wait until a predicate becomes true, periodically logging progress.

    Polls the predicate at regular intervals and logs a status message if the
    wait exceeds the notification threshold. Useful for debugging blocking
    operations in data pipelines and training loops.

    Args:
        predicate: Callable returning bool. Wait continues until this returns True.
        message: Status message to log during wait (duration will be appended).
        info: Optional additional information to include in log messages. Can be
            a string, callable returning a string, or any object with __str__.
        sleep: Time in seconds to sleep between predicate checks. Default 0.01s.
        notify: Minimum seconds between log notifications. Default 60s. Set to
            float('inf') to disable periodic notifications.

    Returns:
        Total time waited in seconds (float). Returns 0 if predicate was
        already True on first check.
    """
    if predicate():
        return 0
    start = last_notify = time.time()
    while not predicate():
        now = time.time()
        if now - last_notify > notify:
            dur = now - start
            print(f"{message} {dur:.1f}s: {info}")
            last_notify = time.time()
        time.sleep(sleep)
    return time.time() - start


class SamplesPerInsert:
    """Rate limiter maintaining a target ratio of samples to inserts.

    Controls the balance between sampling from and inserting into a replay buffer
    by tracking available "credits" for each operation. Ensures the buffer builds
    up to a minimum size before allowing samples, then maintains approximately
    samples_per_insert samples for each insert operation.

    This prevents:
    - Sampling from an empty or too-small buffer (cold start problem)
    - Inserting too fast relative to sampling (buffer overflow)
    - Sampling too fast relative to inserting (buffer starvation)

    The tolerance parameter allows some flexibility in the ratio to avoid
    excessive blocking while still maintaining approximate balance.

    Typical usage:
        limiter = SamplesPerInsert(samples_per_insert=8, tolerance=2, minsize=1000)

        # Producer thread
        if limiter.want_insert():
            buffer.insert(data)
            limiter.insert()

        # Consumer thread
        if limiter.want_sample():
            data = buffer.sample()
            limiter.sample()

    Attributes:
        samples_per_insert: Target ratio of samples to inserts.
        minsize: Minimum buffer size before allowing any samples.
        avail: Current available credits for sampling (can be negative).
        size: Current buffer size (number of inserts).
        lock: Thread lock for atomic operations.
    """

    def __init__(self, samples_per_insert, tolerance, minsize):
        """Initialize the rate limiter.

        Args:
            samples_per_insert: Target number of samples allowed per insert.
                Set to 0 or negative to disable rate limiting (unlimited sampling).
                Typical values: 8-32 for replay buffers with train_ratio parameter.
            tolerance: Flexibility factor multiplied by samples_per_insert to set
                bounds on available credits. Higher tolerance = less blocking but
                looser ratio guarantees. Typical values: 1-4.
            minsize: Minimum buffer size before allowing any samples. Must be >= 1.
                Typical values: 1000-10000 for prefill/burn-in period.

        Raises:
            AssertionError: If minsize < 1.
        """
        assert minsize >= 1
        self.samples_per_insert = samples_per_insert
        self.minsize = minsize
        self.avail = -minsize
        self.min_avail = -tolerance
        self.max_avail = tolerance * samples_per_insert
        self.size = 0
        self.lock = threading.Lock()

    def save(self):
        """Save the current state for checkpointing.

        Returns:
            Dictionary containing 'size' and 'avail' keys for state persistence.
            Can be passed to load() to restore this limiter's state.
        """
        return {"size": self.size, "avail": self.avail}

    def load(self, data):
        """Restore state from a checkpoint.

        Args:
            data: Dictionary with 'size' and 'avail' keys, typically from save().
                Restores the limiter to continue from a previous training run.
        """
        self.size = data["size"]
        self.avail = data["avail"]

    def want_insert(self):
        """Check if an insert operation should proceed.

        Determines whether the producer should insert more data based on current
        available credits. Inserts are allowed when:
        - Buffer size is below minsize (build-up phase), OR
        - Rate limiting is disabled (samples_per_insert <= 0), OR
        - Available credits are below the maximum threshold

        This prevents inserting too fast relative to sampling, which could cause
        memory issues or waste computation on data that won't be sampled soon.

        Returns:
            True if insert should proceed, False if producer should wait/block.
        """
        # if self.samples_per_insert <= 0 or self.size < self.minsize:
        #   return True, 'ok'
        # if self.avail >= self.max_avail:
        #   return False, f'rate limited: {self.avail:.3f} >= {self.max_avail:.3f}'
        # return True, 'ok'

        if self.size < self.minsize:
            return True
        if self.samples_per_insert <= 0:
            return True
        if self.avail < self.max_avail:
            return True
        return False

    def want_sample(self):
        """Check if a sample operation should proceed.

        Determines whether the consumer should sample data based on buffer size
        and available credits. Samples are allowed when:
        - Buffer size meets minsize requirement, AND
        - Rate limiting is disabled (samples_per_insert <= 0), OR
        - Available credits are above the minimum threshold

        This prevents sampling from an empty/small buffer (cold start) or sampling
        too fast relative to inserting (buffer starvation).

        Returns:
            True if sample should proceed, False if consumer should wait/block.
        """
        # if self.size < self.minsize:
        #   return False, f'too empty: {self.size} < {self.minsize}'
        # if self.samples_per_insert > 0 and self.avail <= self.min_avail:
        #   return False, f'rate limited: {self.avail:.3f} <= {self.min_avail:.3f}'
        # return True, 'ok'

        if self.size < self.minsize:
            return False
        if self.samples_per_insert <= 0:
            return True
        if self.min_avail < self.avail:
            return True
        return False

    def insert(self):
        """Record that an insert operation occurred.

        Increments buffer size and adds sampling credits (after minsize reached).
        Must be called after successfully inserting data into the buffer.

        Thread-safe: Uses internal lock for atomic updates.
        """
        with self.lock:
            self.size += 1
            if self.size >= self.minsize:
                self.avail += self.samples_per_insert

    # def remove(self):
    #   with self.lock:
    #     self.size -= 1

    def sample(self):
        """Record that a sample operation occurred.

        Decrements available sampling credits. Must be called after successfully
        sampling data from the buffer.

        Thread-safe: Uses internal lock for atomic updates.
        """
        with self.lock:
            self.avail -= 1
