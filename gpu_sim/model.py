"""Data model shared by the simulator, the detector and the bring-up pipeline.

The field names intentionally mirror DCGM field identifiers (see ``metrics.py``)
so that a :class:`Snapshot` can be produced either by the simulator or by
querying Prometheus for real ``dcgm-exporter`` series.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Bits of DCGM_FI_DEV_CLOCK_THROTTLE_REASONS (identical to NVML
# nvmlClocksThrottleReasons*). Only the ones GPUWatch reasons about are named.
THROTTLE_GPU_IDLE = 0x1
THROTTLE_SW_POWER_CAP = 0x4
THROTTLE_HW_SLOWDOWN = 0x8
THROTTLE_SW_THERMAL = 0x20
THROTTLE_HW_THERMAL = 0x40
THROTTLE_HW_POWER_BRAKE = 0x80
THERMAL_THROTTLE_MASK = THROTTLE_SW_THERMAL | THROTTLE_HW_THERMAL


@dataclass(frozen=True)
class GPUSample:
    """One scrape of one GPU (DCGM-style gauges and counters)."""

    node: str
    gpu: int
    util_pct: float
    mem_used_mib: float
    mem_total_mib: float
    temp_c: float
    power_w: float
    sm_clock_mhz: float
    throttle_reasons: int
    ecc_sbe_total: int
    ecc_dbe_total: int
    xid: int
    nvlink_gbps: float
    available: bool = True


@dataclass(frozen=True)
class NodeSample:
    """One scrape of node-level / training-job metrics for one node.

    ``step_time_s`` is the rank-local time-to-barrier of the training step
    (forward + backward + input pipeline, excluding time spent waiting in the
    gradient all-reduce). It is ``None`` when the job emitted no step during
    the scrape interval (e.g. the job is hung).

    ``net_xmit_wait`` is the rate of InfiniBand ``port_xmit_wait`` ticks per
    second summed over the node's ports (time the NIC had data queued but
    could not transmit). Its absolute unit is hardware specific; the detector
    only compares it across peer nodes.

    ``nccl_busbw_gb_s`` is the all-reduce bus bandwidth (GB/s, nccl-tests
    definition) measured by the training job on this node.
    """

    node: str
    step_time_s: float | None
    net_tx_gbps: float
    net_xmit_wait: float
    nccl_busbw_gb_s: float | None


@dataclass(frozen=True)
class LogLine:
    """A log excerpt (kernel ring buffer, NCCL, or training log)."""

    ts: float
    node: str
    source: str
    message: str


@dataclass(frozen=True)
class Snapshot:
    """Everything observed about the cluster at one scrape timestamp."""

    ts: float
    gpus: tuple[GPUSample, ...]
    nodes: tuple[NodeSample, ...]
    logs: tuple[LogLine, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain JSON-compatible types."""
        return {
            "ts": self.ts,
            "gpus": [asdict(g) for g in self.gpus],
            "nodes": [asdict(n) for n in self.nodes],
            "logs": [asdict(log) for log in self.logs],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Snapshot:
        """Inverse of :meth:`to_dict`."""
        return Snapshot(
            ts=float(d["ts"]),
            gpus=tuple(GPUSample(**g) for g in d.get("gpus", [])),
            nodes=tuple(NodeSample(**n) for n in d.get("nodes", [])),
            logs=tuple(LogLine(**log) for log in d.get("logs", [])),
        )

    def node_names(self) -> list[str]:
        """Sorted unique node names present in this snapshot."""
        return sorted({n.node for n in self.nodes} | {g.node for g in self.gpus})
