"""Snapshot sources: simulator, JSONL file, and Prometheus (live)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from gpu_sim import metrics as m
from gpu_sim.cluster import ClusterSim
from gpu_sim.model import GPUSample, NodeSample, Snapshot


class SnapshotSource(Protocol):
    """Anything that yields snapshots in time order."""

    def __iter__(self) -> Iterator[Snapshot]:
        """Yield snapshots."""
        ...


@dataclass
class SimSource:
    """Drive a :class:`ClusterSim` for ``duration_s`` simulated seconds."""

    sim: ClusterSim
    duration_s: float

    def __iter__(self) -> Iterator[Snapshot]:
        """Yield simulated snapshots."""
        return self.sim.run(self.duration_s)


@dataclass
class JsonlSource:
    """Replay snapshots written by ``python -m gpu_sim run``."""

    path: Path

    def __iter__(self) -> Iterator[Snapshot]:
        """Yield snapshots from the file."""
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    yield Snapshot.from_dict(json.loads(line))


# PromQL used to assemble a snapshot. GPU queries must return series labelled
# with Hostname and gpu (dcgm-exporter defaults); node queries are aggregated
# to Hostname. Per-rank step times are reduced with max(): a node is as slow
# as its slowest rank.
DEFAULT_GPU_QUERIES: dict[str, str] = {
    "util_pct": m.GPU_UTIL,
    "mem_used_mib": m.FB_USED,
    "mem_free_mib": m.FB_FREE,
    "temp_c": m.GPU_TEMP,
    "power_w": m.POWER_USAGE,
    "sm_clock_mhz": m.SM_CLOCK,
    "throttle_reasons": m.THROTTLE_REASONS,
    "ecc_sbe_total": m.ECC_SBE_TOTAL,
    "ecc_dbe_total": m.ECC_DBE_TOTAL,
    "xid": m.XID_ERRORS,
    "nvlink_gbps": m.NVLINK_BW,
}
DEFAULT_NODE_QUERIES: dict[str, str] = {
    "step_time_s": f"max by ({m.LABEL_NODE}) ({m.STEP_TIME})",
    "net_tx_gbps": f"sum by ({m.LABEL_NODE}) ({m.NET_TX})",
    "net_xmit_wait": f"sum by ({m.LABEL_NODE}) ({m.NET_XMIT_WAIT})",
    "nccl_busbw_gb_s": f"min by ({m.LABEL_NODE}) ({m.NCCL_BUSBW})",
}


@dataclass
class PrometheusSource:
    """Build snapshots from Prometheus instant queries.

    Modes:
        * follow (``start`` is None): query "now" every ``interval_s`` forever
          (or ``max_polls`` times).
        * backfill (``start``/``end`` set): evaluate at ``start, start+step, ...``
          for offline replay of an incident window.

    The DCGM ``FB_USED + FB_FREE`` pair is converted to used/total. A GPU that
    reports an XID but no utilization is marked unavailable (fallen off bus).
    """

    url: str
    interval_s: float = 10.0
    start: float | None = None
    end: float | None = None
    max_polls: int | None = None
    timeout_s: float = 10.0
    gpu_queries: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_GPU_QUERIES))
    node_queries: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_NODE_QUERIES))
    session: Any = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time

    def _query(self, promql: str, ts: float) -> list[dict[str, Any]]:
        sess = self.session or requests
        resp = sess.get(
            f"{self.url.rstrip('/')}/api/v1/query",
            params={"query": promql, "time": f"{ts:.3f}"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            raise RuntimeError(f"prometheus query failed: {body}")
        result: list[dict[str, Any]] = body["data"]["result"]
        return result

    def snapshot_at(self, ts: float) -> Snapshot:
        """Evaluate all queries at ``ts`` and assemble a :class:`Snapshot`."""
        gpu_vals: dict[tuple[str, int], dict[str, float]] = {}
        for fld, q in self.gpu_queries.items():
            for series in self._query(q, ts):
                lbl = series["metric"]
                if m.LABEL_NODE not in lbl or m.LABEL_GPU not in lbl:
                    continue
                key = (lbl[m.LABEL_NODE], int(lbl[m.LABEL_GPU]))
                gpu_vals.setdefault(key, {})[fld] = float(series["value"][1])
        node_vals: dict[str, dict[str, float]] = {}
        for fld, q in self.node_queries.items():
            for series in self._query(q, ts):
                node = series["metric"].get(m.LABEL_NODE)
                if node:
                    node_vals.setdefault(node, {})[fld] = float(series["value"][1])
        gpus = []
        for (node, gpu), v in sorted(gpu_vals.items()):
            used, free = v.get("mem_used_mib", 0.0), v.get("mem_free_mib", 0.0)
            gpus.append(
                GPUSample(
                    node=node,
                    gpu=gpu,
                    util_pct=v.get("util_pct", 0.0),
                    mem_used_mib=used,
                    mem_total_mib=used + free,
                    temp_c=v.get("temp_c", 0.0),
                    power_w=v.get("power_w", 0.0),
                    sm_clock_mhz=v.get("sm_clock_mhz", 0.0),
                    throttle_reasons=int(v.get("throttle_reasons", 0)),
                    ecc_sbe_total=int(v.get("ecc_sbe_total", 0)),
                    ecc_dbe_total=int(v.get("ecc_dbe_total", 0)),
                    xid=int(v.get("xid", 0)),
                    nvlink_gbps=v.get("nvlink_gbps", 0.0),
                    available="util_pct" in v,
                )
            )
        nodes = []
        for node, v in sorted(node_vals.items()):
            nodes.append(
                NodeSample(
                    node=node,
                    step_time_s=v.get("step_time_s"),
                    net_tx_gbps=v.get("net_tx_gbps", 0.0),
                    net_xmit_wait=v.get("net_xmit_wait", 0.0),
                    nccl_busbw_gb_s=v.get("nccl_busbw_gb_s"),
                )
            )
        return Snapshot(ts=ts, gpus=tuple(gpus), nodes=tuple(nodes))

    def __iter__(self) -> Iterator[Snapshot]:
        """Yield snapshots (follow or backfill mode)."""
        if self.start is not None:
            end = self.end if self.end is not None else self.clock()
            ts = self.start
            while ts <= end:
                yield self.snapshot_at(ts)
                ts += self.interval_s
            return
        polls = 0
        while self.max_polls is None or polls < self.max_polls:
            began = self.clock()
            yield self.snapshot_at(began)
            polls += 1
            self.sleep(max(0.0, self.interval_s - (self.clock() - began)))
