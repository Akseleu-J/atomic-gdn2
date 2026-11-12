"""Monotonic training deadline and a wall-time learning-rate envelope."""
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class WallClockBudget:
    started: float
    duration: float

    @classmethod
    def from_deadline(cls, deadline_unix, reserve_seconds):
        remaining = deadline_unix - time.time() - reserve_seconds
        if remaining <= 0:
            raise TimeoutError('No training time remains after the finalization reserve')
        return cls(time.monotonic(), remaining)

    def remaining(self, now=None):
        return max(0., self.duration - ((time.monotonic() if now is None else now) - self.started))

    def expired(self, now=None):
        return self.remaining(now) <= 0.

    def learning_rate(self, step, peak, warmup_ratio, min_ratio, now=None):
        elapsed = max(0., (time.monotonic() if now is None else now) - self.started)
        warm_seconds = max(1e-6, self.duration * warmup_ratio)
        # Compilation may consume the time warmup: still ramp over the first 100 updates.
        warm = min(1., max(0., step / 100.), elapsed / warm_seconds)
        progress = min(1., max(0., (elapsed - warm_seconds) / max(1e-6, self.duration - warm_seconds)))
        cosine = min_ratio + (1 - min_ratio) * .5 * (1 + math.cos(math.pi * progress))
        return peak * warm * cosine
