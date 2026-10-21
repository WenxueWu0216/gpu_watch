"""Builders for hand-crafted snapshots used by detector tests."""

from __future__ import annotations

from gpu_sim.model import GPUSample, LogLine, NodeSample, Snapshot

NODES = [f"node-{i:02d}" for i in range(8)]


def gpu(node: str, idx: int, **kw: object) -> GPUSample:
    base: dict[str, object] = dict(
        node=node,
        gpu=idx,
        util_pct=90.0,
        mem_used_mib=70000.0,
        mem_total_mib=81559.0,
        temp_c=65.0,
        power_w=600.0,
        sm_clock_mhz=1980.0,
        throttle_reasons=0,
        ecc_sbe_total=0,
        ecc_dbe_total=0,
        xid=0,
        nvlink_gbps=270.0,
    )
    base.update(kw)
    return GPUSample(**base)  # type: ignore[arg-type]


def snap(
    ts: float,
    nodes: list[str] | None = None,
    gpus_per_node: int = 2,
    step: dict[str, float | None] | None = None,
    wait: dict[str, float] | None = None,
    busbw: float | None = 45.0,
    gpu_over: dict[tuple[str, int], dict[str, object]] | None = None,
    logs: tuple[LogLine, ...] = (),
    drop_gpus: set[tuple[str, int]] | None = None,
) -> Snapshot:
    nodes = nodes or NODES
    step = step or {}
    wait = wait or {}
    gpu_over = gpu_over or {}
    drop_gpus = drop_gpus or set()
    gs = tuple(
        gpu(n, i, **gpu_over.get((n, i), {}))
        for n in nodes
        for i in range(gpus_per_node)
        if (n, i) not in drop_gpus
    )
    ns = tuple(NodeSample(n, step.get(n, 0.5), 160.0, wait.get(n, 2000.0), busbw) for n in nodes)
    return Snapshot(ts, gs, ns, logs)
