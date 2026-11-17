"""Bounded, nonblocking preparation motion; never used to limit human teleoperation."""
import numpy as np


class LeaderAlignment:
    def __init__(self, current, target, *, now, speed=.15, tolerance=.08, settle=.5, timeout=60.):
        self.target = np.asarray(target, dtype=float).copy()
        self.reference = np.asarray(current, dtype=float).copy()
        if self.target.shape != (7,) or self.reference.shape != (7,) or not np.isfinite([self.target, self.reference]).all():
            raise ValueError("Alignment needs seven finite joint angles")
        self.speed, self.tolerance, self.settle, self.timeout = speed, tolerance, settle, timeout
        self.started = self.last = now
        self.settled_since = None
        self.done = False
        self.error = float(np.max(np.abs(self.target-self.reference)))

    def update(self, q, dq, now):
        q, dq = np.asarray(q), np.asarray(dq)
        if not np.isfinite([q, dq]).all():
            raise ValueError("Invalid alignment feedback")
        if now-self.started > self.timeout:
            raise RuntimeError("FACTR alignment timed out; support arm and inspect obstruction/calibration")
        dt = max(0., min(now-self.last, .01))
        self.last = now
        self.reference += np.clip(self.target-self.reference, -self.speed*dt, self.speed*dt)
        # Never accumulate a large PD error when the operator blocks the leader.
        self.reference = np.clip(self.reference, q-.12, q+.12)
        self.error = float(np.max(np.abs(self.target-q)))
        settled = self.error <= self.tolerance and np.max(np.abs(dq)) <= .15
        self.settled_since = (self.settled_since if self.settled_since is not None else now) if settled else None
        self.done = self.settled_since is not None and now-self.settled_since >= self.settle
        return self.reference-q

    def status(self):
        return {"state": "COMPLETE" if self.done else "ALIGNING", "error_rad": self.error,
                "target": self.target.tolist(), "reference": self.reference.tolist()}
