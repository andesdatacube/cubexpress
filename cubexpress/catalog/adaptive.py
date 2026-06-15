"""adaptive: AIMD worker control for concurrent GEE batch calls.

Shared by the discover and metrics batch engines. GEE rate-limits bursts of
concurrent requests ("Too Many Requests" / quota errors). A fixed worker count
either underuses the quota (too low) or trips the limit (too high). AIMD
(Additive Increase, Multiplicative Decrease) finds the right level live:

  - a run of successes nudges the worker count UP by one (additive),
  - a rate-limit error cuts it in HALF (multiplicative),

so the pool settles just under whatever GEE currently allows, on any network or
quota, without the user guessing.

This handles RATE-LIMIT only. Volume/memory errors (a batch too big for the
server) are a different axis, handled by the shrink-and-retry layer.
"""

from __future__ import annotations


# Substrings that mark a GEE rate-limit / quota error (vs a volume/memory one).
_RATE_LIMIT_MARKERS = (
    "too many requests",
    "rate limit",
    "quota",
    "user rate",
    "429",
)


def is_rate_limit_error(exc: Exception) -> bool:
    """True if the exception looks like a GEE rate-limit / quota error.

    Rate-limit errors mean 'slow down' (fewer workers); they are distinct from
    volume/memory errors ('User memory limit exceeded'), which mean 'smaller
    batch' and are handled elsewhere.

    Args:
        exc: the exception raised by a batch call.

    Returns:
        True if the message matches a known rate-limit marker.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


class AdaptiveWorkers:
    """Tracks a worker count that adapts to GEE rate-limit feedback (AIMD).

    Start at `initial`. After `success_streak_to_grow` consecutive successful
    rounds, increase by one (additive), capped at `max_workers`. On a rate-limit
    signal, halve (multiplicative), floored at 1, and reset the success streak.

    The count changes BETWEEN rounds; a round reads `.current` to size its pool.

    Args:
        initial: starting worker count (what the user requested).
        max_workers: ceiling for growth (default: 2x initial).
        min_workers: floor for shrink (default 1).
        success_streak_to_grow: successful rounds before growing by one.
    """

    def __init__(
        self,
        initial: int,
        max_workers: int | None = None,
        min_workers: int = 1,
        success_streak_to_grow: int = 3,
    ):
        if initial < 1:
            raise ValueError(f"initial must be >= 1, got {initial}")
        self._current = initial
        self._max = max_workers if max_workers is not None else initial * 2
        self._min = max(1, min_workers)
        self._streak_target = success_streak_to_grow
        self._streak = 0

    @property
    def current(self) -> int:
        """The worker count to use for the next round."""
        return self._current

    def on_success(self) -> None:
        """Record a successful round; grow by one after enough in a row."""
        self._streak += 1
        if self._streak >= self._streak_target and self._current < self._max:
            self._current += 1
            self._streak = 0

    def on_rate_limit(self) -> None:
        """Record a rate-limit signal; halve the worker count, reset the streak."""
        self._current = max(self._min, self._current // 2)
        self._streak = 0