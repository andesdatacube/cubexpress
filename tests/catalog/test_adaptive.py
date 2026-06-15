import pytest

from cubexpress.catalog.adaptive import AdaptiveWorkers, is_rate_limit_error


# --- is_rate_limit_error ---

def test_detects_too_many_requests():
    assert is_rate_limit_error(RuntimeError("Too Many Requests"))


def test_detects_quota():
    assert is_rate_limit_error(RuntimeError("Quota exceeded for ..."))


def test_detects_429():
    assert is_rate_limit_error(RuntimeError("HttpError 429"))


def test_memory_error_is_not_rate_limit():
    """Volume/memory errors are a different axis; not rate-limit."""
    assert not is_rate_limit_error(RuntimeError("User memory limit exceeded."))


def test_timeout_is_not_rate_limit():
    assert not is_rate_limit_error(RuntimeError("Computation timed out."))


# --- AdaptiveWorkers ---

def test_starts_at_initial():
    aw = AdaptiveWorkers(initial=8)
    assert aw.current == 8


def test_grows_after_streak():
    aw = AdaptiveWorkers(initial=8, success_streak_to_grow=3)
    aw.on_success(); aw.on_success()
    assert aw.current == 8          # not yet
    aw.on_success()
    assert aw.current == 9          # grew after 3


def test_growth_capped_at_max():
    aw = AdaptiveWorkers(initial=2, max_workers=3, success_streak_to_grow=1)
    aw.on_success()
    assert aw.current == 3
    aw.on_success()
    assert aw.current == 3          # capped


def test_rate_limit_halves():
    aw = AdaptiveWorkers(initial=8)
    aw.on_rate_limit()
    assert aw.current == 4
    aw.on_rate_limit()
    assert aw.current == 2


def test_shrink_floored_at_min():
    aw = AdaptiveWorkers(initial=2, min_workers=1)
    aw.on_rate_limit()
    assert aw.current == 1
    aw.on_rate_limit()
    assert aw.current == 1          # floored


def test_rate_limit_resets_streak():
    aw = AdaptiveWorkers(initial=8, success_streak_to_grow=3)
    aw.on_success(); aw.on_success()   # streak 2
    aw.on_rate_limit()                 # resets
    aw.on_success(); aw.on_success()   # streak 2 again, not 4
    assert aw.current == 4             # halved once, no growth yet


def test_initial_must_be_positive():
    with pytest.raises(ValueError, match="initial must be"):
        AdaptiveWorkers(initial=0)