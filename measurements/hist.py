"""Streaming distribution accumulator -- percentiles without holding the data.

The multi-gigabyte passes checkpoint their state between invocations, so an
accumulator that stores every observation would mean a 150 MB pickle written
several times.  This keeps a bucketed histogram instead: exact count, sum, min
and max, plus a bucket counter at a stated resolution.  Reported percentiles
are therefore accurate to one bucket width, which is written into every
result so no number is quoted at a precision it does not have.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


class Dist:
    def __init__(self, name: str, bucket: int = 1000) -> None:
        self.name = name
        self.bucket = bucket
        self.buckets: Counter = Counter()
        self.n = 0
        self.total = 0
        self.lo: int | None = None
        self.hi: int | None = None

    def add(self, values: np.ndarray) -> None:
        if not len(values):
            return
        v = np.asarray(values, dtype=np.int64)
        self.n += int(v.size)
        self.total += int(v.sum())
        mn, mx = int(v.min()), int(v.max())
        self.lo = mn if self.lo is None else min(self.lo, mn)
        self.hi = mx if self.hi is None else max(self.hi, mx)
        idx, cnt = np.unique(v // self.bucket, return_counts=True)
        self.buckets.update(dict(zip(idx.tolist(), cnt.tolist())))

    def quantile(self, q: float) -> float:
        """Lower edge of the bucket containing the q-th percentile."""
        if not self.n:
            return float("nan")
        target = q / 100.0 * self.n
        seen = 0
        for b in sorted(self.buckets):
            seen += self.buckets[b]
            if seen >= target:
                return float(b * self.bucket)
        return float(self.hi or 0)

    def count_over(self, threshold: int) -> int:
        """Exact only when ``threshold`` lands on a bucket edge."""
        edge = threshold // self.bucket
        return sum(c for b, c in self.buckets.items() if b > edge)

    def row(self, pcts=(50, 75, 90, 95, 99, 99.9)) -> dict:
        d = {"metric": self.name, "n": self.n, "bucket_bytes": self.bucket,
             "min": self.lo, "max": self.hi,
             "mean": (self.total / self.n) if self.n else float("nan")}
        for p in pcts:
            d[f"p{p}"] = self.quantile(p)
        return d


class Corr:
    """Exact Pearson correlation and residual stats from running sums."""

    def __init__(self) -> None:
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.y_gt_x = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        if not len(x):
            return
        xf, yf = x.astype(float), y.astype(float)
        self.n += len(xf)
        self.sx += float(xf.sum()); self.sy += float(yf.sum())
        self.sxx += float((xf * xf).sum()); self.syy += float((yf * yf).sum())
        self.sxy += float((xf * yf).sum())
        self.y_gt_x += int((yf > xf).sum())

    def r(self) -> float:
        n = self.n
        if n < 2:
            return float("nan")
        cov = self.sxy - self.sx * self.sy / n
        vx = self.sxx - self.sx * self.sx / n
        vy = self.syy - self.sy * self.sy / n
        return cov / ((vx * vy) ** 0.5) if vx > 0 and vy > 0 else float("nan")
