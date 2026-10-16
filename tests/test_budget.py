import pytest
from atomic_gdn2_model.budget import WallClockBudget


def test_clock_warmup_and_cosine_floor():
    b = WallClockBudget(100., 1000.)
    assert b.learning_rate(0, 3e-4, .01, .1, now=100.) == 0
    assert b.learning_rate(100, 3e-4, .01, .1, now=110.) == pytest.approx(3e-4)
    assert b.learning_rate(1000, 3e-4, .01, .1, now=1100.) == pytest.approx(3e-5)
    assert b.learning_rate(1000, 3e-4, .01, .1, now=1200.) == pytest.approx(3e-5)


def test_compile_time_cannot_skip_first_100_update_warmup():
    b = WallClockBudget(0., 1000.)
    assert b.learning_rate(0, 1., .01, .1, now=200.) == 0
    assert b.learning_rate(10, 1., .01, .1, now=200.) == pytest.approx(
        .1 * b.learning_rate(100, 1., .01, .1, now=200.))


def test_monotonic_expiry():
    b = WallClockBudget(10., 100.)
    assert b.remaining(now=60.) == 50.
    assert not b.expired(now=109.)
    assert b.expired(now=110.)
    assert b.remaining(now=500.) == 0.


def test_finalization_reserve_is_inside_deadline(monkeypatch):
    monkeypatch.setattr('atomic_gdn2_model.budget.time.time', lambda: 100.)
    monkeypatch.setattr('atomic_gdn2_model.budget.time.monotonic', lambda: 10.)
    b = WallClockBudget.from_deadline(1000., 100.)
    assert b.duration == 800.
    with pytest.raises(TimeoutError):
        WallClockBudget.from_deadline(150., 100.)
