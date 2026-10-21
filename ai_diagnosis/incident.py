"""Assemble an incident: the alert plus correlated metrics and log excerpts.

Every evidence item gets a stable ID (``A*`` alert evidence, ``M*`` metrics,
``L*`` log lines). The LLM may only make claims that cite these IDs.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from detector.events import Event
from gpu_sim import metrics as m
from gpu_sim.model import GPUSample, Snapshot


@dataclass(frozen=True)
class EvidenceItem:
    """One citable piece of evidence."""

    id: str
    kind: str  # "alert" | "metric" | "log"
    text: str


@dataclass
class Incident:
    """An alert with its correlated context."""

    id: str
    alert: Event
    items: list[EvidenceItem] = field(default_factory=list)

    def by_id(self) -> dict[str, EvidenceItem]:
        """Evidence items keyed by ID."""
        return {i.id: i for i in self.items}

    def evidence_sha(self) -> str:
        """Hash of the evidence text (detects stale replay fixtures)."""
        h = hashlib.sha256("\n".join(i.text for i in self.items).encode())
        return h.hexdigest()[:16]

    def render(self) -> str:
        """Evidence block as given to the model, one item per line."""
        a = self.alert
        where = f"node={a.node}" + (f" gpu={a.gpu}" if a.gpu is not None else "")
        head = [
            f"ALERT {a.kind.value} severity={a.severity.value} {where} t={a.ts:.0f}s",
            f"summary: {a.summary}",
            "",
            "EVIDENCE:",
        ]
        return "\n".join(head + [f"[{i.id}] ({i.kind}) {i.text}" for i in self.items])


def _series(vals: Sequence[float]) -> str:
    if not vals:
        return "no samples"
    return (
        f"first={vals[0]:.5g} last={vals[-1]:.5g} min={min(vals):.5g} "
        f"max={max(vals):.5g} n={len(vals)}"
    )


def build_incident(
    event: Event,
    history: Sequence[Snapshot],
    window_s: float = 300.0,
    max_logs: int = 8,
) -> Incident:
    """Correlate ``event`` with metrics/logs from ``history`` in ``[ts-window, ts]``.

    For a GPU-scoped alert the GPU's own telemetry plus same-node peer medians
    are included. For a node-scoped alert, node/job metrics vs. the cluster
    median are included, plus any GPU on that node that shows thermal
    throttling, ECC or XID activity in the window. Kernel/NCCL log lines from
    the node are attached; routine training logs are omitted.
    """
    snaps = [s for s in history if event.ts - window_s <= s.ts <= event.ts]
    items: list[EvidenceItem] = []
    for i, ev in enumerate(event.evidence, 1):
        thr = f" threshold={ev.threshold:g}" if ev.threshold is not None else ""
        lbl = ",".join(f"{k}={v}" for k, v in sorted(ev.labels.items()))
        items.append(
            EvidenceItem(
                f"A{i}", "alert", f"{ev.metric}{{{lbl}}} value={ev.value:g}{thr} - {ev.note}"
            )
        )
    metric_items: list[str] = []
    node = event.node

    def gpu_rows(gpu: int) -> list[GPUSample]:
        return [g for s in snaps for g in s.gpus if g.node == node and g.gpu == gpu]

    def describe_gpu(gpu: int) -> None:
        rows = gpu_rows(gpu)
        if not rows:
            return
        lbl = f"{{node={node},gpu={gpu}}}"
        avail = [g for g in rows if g.available]
        if len(avail) < len(rows):
            first_lost = next(
                s.ts
                for s in snaps
                for g in s.gpus
                if g.node == node and g.gpu == gpu and not g.available
            )
            metric_items.append(
                f"{m.GPU_UTIL}{lbl} unavailable (not readable) from "
                f"t={first_lost:.0f}s; {len(rows) - len(avail)} of "
                f"{len(rows)} scrapes missing"
            )
        if avail:
            metric_items.append(f"{m.GPU_TEMP}{lbl} {_series([g.temp_c for g in avail])}")
            metric_items.append(f"{m.SM_CLOCK}{lbl} {_series([g.sm_clock_mhz for g in avail])}")
            metric_items.append(f"{m.GPU_UTIL}{lbl} {_series([g.util_pct for g in avail])}")
            bits = sorted({g.throttle_reasons for g in avail})
            metric_items.append(
                f"{m.THROTTLE_REASONS}{lbl} distinct values in window: "
                + ", ".join(hex(b) for b in bits)
            )
        metric_items.append(
            f"{m.ECC_SBE_TOTAL}{lbl} {_series([float(g.ecc_sbe_total) for g in rows])}"
        )
        metric_items.append(
            f"{m.ECC_DBE_TOTAL}{lbl} {_series([float(g.ecc_dbe_total) for g in rows])}"
        )
        metric_items.append(f"{m.XID_ERRORS}{lbl} {_series([float(g.xid) for g in rows])}")
        if snaps:
            last = snaps[-1]
            peers = [g for g in last.gpus if g.node == node and g.gpu != gpu and g.available]
            if peers:
                metric_items.append(
                    f"peer GPUs on {node} at t={last.ts:.0f}s: median "
                    f"{m.GPU_TEMP}={statistics.median(g.temp_c for g in peers):.5g} "
                    f"median {m.SM_CLOCK}={statistics.median(g.sm_clock_mhz for g in peers):.5g}"
                )

    if node is not None and event.gpu is not None:
        describe_gpu(event.gpu)
    elif node is not None:
        steps = [
            n.step_time_s
            for s in snaps
            for n in s.nodes
            if n.node == node and n.step_time_s is not None
        ]
        med = [
            statistics.median([n.step_time_s for n in s.nodes if n.step_time_s is not None])
            for s in snaps
            if any(n.step_time_s is not None for n in s.nodes)
        ]
        metric_items.append(f"{m.STEP_TIME}{{node={node}}} {_series(steps)}")
        metric_items.append(f"{m.STEP_TIME} cluster median per scrape {_series(med)}")
        waits = [n.net_xmit_wait for s in snaps for n in s.nodes if n.node == node]
        wmed = [statistics.median(n.net_xmit_wait for n in s.nodes) for s in snaps]
        metric_items.append(f"{m.NET_XMIT_WAIT}{{node={node}}} {_series(waits)}")
        metric_items.append(f"{m.NET_XMIT_WAIT} cluster median per scrape {_series(wmed)}")
        bus = [
            n.nccl_busbw_gb_s
            for s in snaps
            for n in s.nodes
            if n.node == node and n.nccl_busbw_gb_s is not None
        ]
        metric_items.append(f"{m.NCCL_BUSBW}{{node={node}}} {_series(bus)}")
        util = [
            statistics.mean(g.util_pct for g in s.gpus if g.node == node and g.available)
            for s in snaps
            if any(g.node == node and g.available for g in s.gpus)
        ]
        cutil = [
            statistics.median(g.util_pct for g in s.gpus if g.available)
            for s in snaps
            if any(g.available for g in s.gpus)
        ]
        metric_items.append(f"{m.GPU_UTIL} mean over {node} GPUs {_series(util)}")
        metric_items.append(f"{m.GPU_UTIL} cluster median {_series(cutil)}")
        suspicious = sorted(
            {
                g.gpu
                for s in snaps
                for g in s.gpus
                if g.node == node
                and (g.throttle_reasons & 0x60 or g.ecc_dbe_total or g.xid or not g.available)
            }
        )
        for gpu in suspicious:
            describe_gpu(gpu)
    items += [EvidenceItem(f"M{i}", "metric", t) for i, t in enumerate(metric_items, 1)]
    logs = [
        log
        for s in snaps
        for log in s.logs
        if (node is None or log.node == node) and log.source != "train"
    ]
    items += [
        EvidenceItem(f"L{i}", "log", f"t={log.ts:.0f}s {log.node} {log.source}: {log.message}")
        for i, log in enumerate(logs[-max_logs:], 1)
    ]
    inc_id = f"{event.kind.value}:{node}:{event.gpu}:{event.ts:.0f}"
    return Incident(inc_id, event, items)
