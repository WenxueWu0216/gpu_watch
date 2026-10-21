"""Robust statistics used for peer comparison."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Scale factor so the modified z-score is comparable to a standard z-score for
# normally distributed data (Iglewicz & Hoaglin, 1993).
MAD_TO_Z = 0.6745


def median(values: Sequence[float]) -> float:
    """Median of a non-empty sequence."""
    return float(statistics.median(values))


def mad(values: Sequence[float], center: float | None = None) -> float:
    """Median absolute deviation (unscaled)."""
    c = median(values) if center is None else center
    return median([abs(v - c) for v in values])


@dataclass(frozen=True)
class PeerStats:
    """Result of a robust peer comparison for one key."""

    value: float
    median: float
    mad: float
    z: float

    @property
    def rel(self) -> float:
        """Relative deviation from the peer median."""
        return (self.value - self.median) / self.median if self.median else 0.0


def robust_z(values: Mapping[str, float], mad_floor_frac: float) -> dict[str, PeerStats]:
    """Modified z-score of each value against the median of all values.

    ``z = 0.6745 * (x - median) / max(MAD, mad_floor_frac * |median|)``.

    The MAD floor prevents a near-identical healthy fleet (MAD ~ 0) from turning
    a negligible absolute difference into a huge z-score; it sets the minimum
    practically meaningful deviation.

    Args:
        values: key -> observation (e.g. node -> windowed median step time).
        mad_floor_frac: lower bound on MAD as a fraction of the median.

    Returns:
        key -> :class:`PeerStats`. Empty if fewer than 3 values (MAD is not
        meaningful for 1-2 points).
    """
    if len(values) < 3:
        return {}
    vals = list(values.values())
    med = median(vals)
    m = mad(vals, med)
    denom = max(m, mad_floor_frac * abs(med), 1e-12)
    return {k: PeerStats(v, med, m, MAD_TO_Z * (v - med) / denom) for k, v in values.items()}
