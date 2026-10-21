"""Structured detector output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    """Event severity, ordered INFO < WARNING < CRITICAL."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric rank for comparisons."""
        return {"info": 0, "warning": 1, "critical": 2}[self.value]


class EventKind(StrEnum):
    """What the detector believes is wrong."""

    STRAGGLER = "straggler"
    XID = "xid_error"
    ECC = "ecc_error"
    GPU_MISSING = "gpu_missing"
    THERMAL = "thermal_throttle"
    SLOW_LINK = "nccl_slow_link"
    BUSBW_DEGRADED = "nccl_busbw_degraded"


@dataclass(frozen=True)
class Evidence:
    """One observation supporting an event.

    Attributes:
        metric: metric name (DCGM field or ``gpuwatch_*``) or derived statistic.
        value: observed value.
        threshold: the threshold it was compared against, if any.
        labels: identifying labels (node / gpu).
        note: short explanation of the comparison.
    """

    metric: str
    value: float
    threshold: float | None = None
    labels: dict[str, str] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class Event:
    """A detector event (``state`` is ``firing`` or ``resolved``)."""

    ts: float
    kind: EventKind
    severity: Severity
    node: str | None
    gpu: int | None
    summary: str
    evidence: tuple[Evidence, ...] = ()
    state: str = "firing"

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict."""
        d = asdict(self)
        d["kind"] = self.kind.value
        d["severity"] = self.severity.value
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d

    def to_json(self) -> str:
        """Single-line JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Event:
        """Inverse of :meth:`to_dict`."""
        return Event(
            ts=float(d["ts"]),
            kind=EventKind(d["kind"]),
            severity=Severity(d["severity"]),
            node=d.get("node"),
            gpu=d.get("gpu"),
            summary=d.get("summary", ""),
            evidence=tuple(Evidence(**e) for e in d.get("evidence", [])),
            state=d.get("state", "firing"),
        )
