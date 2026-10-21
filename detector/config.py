"""Detector thresholds. Every value is documented; defaults live in ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

# XIDs that indicate hardware / driver faults requiring node action (NVIDIA
# "XID Errors" guide). 13/31/43/45 are commonly triggered by application bugs
# and are reported as warnings instead.
DEFAULT_XID_CRITICAL = (48, 61, 62, 63, 64, 74, 79, 92, 94, 95, 119, 120)


@dataclass(frozen=True)
class DetectorConfig:
    """Tunable thresholds.

    Attributes:
        window: scrapes per sliding window for hardware/network rules.
        step_window: scrapes per sliding window for step-time / utilization peer
            comparison; longer than ``window`` so that benign multi-scrape
            slowdowns (checkpoint upload, GC) shorter than ~half the window are
            absorbed by the median.
        persistence: consecutive anomalous evaluations before an event fires.
        resolve_after: consecutive clean evaluations before an event resolves.
        z_threshold: modified z-score cut-off (3.5 per Iglewicz & Hoaglin).
        step_mad_floor: MAD floor for step time as a fraction of the median.
        step_min_rel: straggler must also be this much slower than the median.
        step_critical_rel: relative slowdown that escalates a straggler to critical.
        pairwise_min_rel: 2-node fallback - slower node must exceed the other by this.
        util_mad_floor: MAD floor for utilization (fraction of median).
        thermal_temp_c: temperature at which a clock drop without throttle bits counts.
        thermal_critical_temp_c: temperature that escalates thermal events.
        thermal_clock_drop: fractional SM clock deficit vs. node peers (fallback rule).
        thermal_persistence: persistence for thermal events.
        sbe_window_delta: correctable ECC errors within one window that raise a warning.
        net_mad_floor: MAD floor for NIC xmit-wait (fraction of median).
        net_min_rel: node xmit-wait must exceed the median by this fraction.
        net_critical_rel: relative excess that escalates a slow link to critical.
        busbw_drop: fractional drop of cluster all-reduce busbw vs. baseline.
        busbw_baseline_scrapes: history length used as the busbw baseline.
        xid_critical: XID codes treated as critical hardware faults.
    """

    window: int = 6
    step_window: int = 12
    persistence: int = 3
    resolve_after: int = 6
    z_threshold: float = 3.5
    step_mad_floor: float = 0.01
    step_min_rel: float = 0.05
    step_critical_rel: float = 0.25
    pairwise_min_rel: float = 0.10
    util_mad_floor: float = 0.02
    thermal_temp_c: float = 87.0
    thermal_critical_temp_c: float = 92.0
    thermal_clock_drop: float = 0.10
    thermal_persistence: int = 3
    sbe_window_delta: int = 10
    net_mad_floor: float = 0.05
    net_min_rel: float = 0.30
    net_critical_rel: float = 2.0
    busbw_drop: float = 0.25
    busbw_baseline_scrapes: int = 60
    xid_critical: tuple[int, ...] = DEFAULT_XID_CRITICAL

    @staticmethod
    def load(path: str | Path | None = None, **overrides: Any) -> DetectorConfig:
        """Load YAML (default: ``detector/config.yaml``) with keyword overrides."""
        data: dict[str, Any] = {}
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}
        data.update(overrides)
        known = {f.name for f in fields(DetectorConfig)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown detector config keys: {sorted(unknown)}")
        if "xid_critical" in data:
            data["xid_critical"] = tuple(int(x) for x in data["xid_critical"])
        return DetectorConfig(**data)
