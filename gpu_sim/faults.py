"""Fault definitions and seeded random fault generation."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import StrEnum


class FaultKind(StrEnum):
    """The five fault classes GPUWatch injects and detects."""

    THERMAL = "thermal"
    XID = "xid"
    ECC = "ecc"
    STRAGGLER = "straggler"
    NETWORK = "network"


GPU_SCOPED = frozenset({FaultKind.THERMAL, FaultKind.XID, FaultKind.ECC})


@dataclass(frozen=True)
class Fault:
    """A fault injected into the simulator.

    Attributes:
        kind: fault class.
        node: affected node name (e.g. ``node-03``).
        start_s: simulated time at which the fault begins.
        gpu: affected GPU index for GPU-scoped faults, else ``None``.
        magnitude: kind-specific severity:
            * thermal   - fractional SM-clock reduction at full throttle (0..1)
            * xid       - unused (1.0)
            * ecc       - mean single-bit errors per scrape during the burst
            * straggler - step-time slowdown factor (> 1)
            * network   - remaining NIC bandwidth fraction (0..1)
        variant: kind-specific variant:
            * xid       - XID code as string (default ``"79"``)
            * ecc       - ``"sbe"`` (correctable burst only) or ``"dbe"``
            * straggler - ``"host"`` (input pipeline) or ``"gpu"`` (slow GPU)
    """

    kind: FaultKind
    node: str
    start_s: float
    gpu: int | None = None
    magnitude: float = 1.0
    variant: str = ""

    def describe(self) -> str:
        """Human-readable one-liner."""
        where = self.node if self.gpu is None else f"{self.node}/gpu{self.gpu}"
        extra = f" variant={self.variant}" if self.variant else ""
        return f"{self.kind.value}@{where} t={self.start_s:.0f}s mag={self.magnitude:.3g}{extra}"

    @staticmethod
    def parse(spec: str) -> Fault:
        """Parse ``kind:node[:gpu]@start[,mag=X][,variant=Y]``.

        Example: ``thermal:node-03:2@300,mag=0.2``.
        """
        head, _, opts = spec.partition(",")
        target, _, start = head.partition("@")
        parts = target.split(":")
        if len(parts) < 2 or not start:
            raise ValueError(f"bad fault spec {spec!r}; want kind:node[:gpu]@start")
        kind = FaultKind(parts[0])
        gpu = int(parts[2]) if len(parts) > 2 else None
        if kind in GPU_SCOPED and gpu is None:
            gpu = 0
        magnitude, variant = _DEFAULTS[kind]
        fault = Fault(kind, parts[1], float(start), gpu, magnitude, variant)
        for opt in filter(None, opts.split(",")):
            key, _, val = opt.partition("=")
            if key == "mag":
                fault = replace(fault, magnitude=float(val))
            elif key == "variant":
                fault = replace(fault, variant=val)
            else:
                raise ValueError(f"unknown fault option {key!r}")
        return fault


_DEFAULTS: dict[FaultKind, tuple[float, str]] = {
    FaultKind.THERMAL: (0.2, ""),
    FaultKind.XID: (1.0, "79"),
    FaultKind.ECC: (30.0, "dbe"),
    FaultKind.STRAGGLER: (1.25, "host"),
    FaultKind.NETWORK: (0.5, ""),
}


def random_fault(
    kind: FaultKind,
    rng: random.Random,
    nodes: int,
    gpus_per_node: int,
    start_window: tuple[float, float],
) -> Fault:
    """Draw a fault of ``kind`` with randomized target, onset and severity.

    Severity ranges deliberately include mild cases that sit near or below the
    detector's sensitivity, so the evaluation measures a real detection rate
    rather than a guaranteed 100%.
    """
    node = f"node-{rng.randrange(nodes):02d}"
    start = rng.uniform(*start_window)
    gpu = rng.randrange(gpus_per_node) if kind in GPU_SCOPED else None
    if kind is FaultKind.THERMAL:
        return Fault(kind, node, start, gpu, magnitude=rng.uniform(0.02, 0.35))
    if kind is FaultKind.XID:
        return Fault(kind, node, start, gpu, magnitude=1.0, variant="79")
    if kind is FaultKind.ECC:
        variant = "dbe" if rng.random() < 0.5 else "sbe"
        return Fault(kind, node, start, gpu, magnitude=rng.uniform(0.2, 60.0), variant=variant)
    if kind is FaultKind.STRAGGLER:
        variant = "host" if rng.random() < 0.5 else "gpu"
        return Fault(kind, node, start, None, magnitude=rng.uniform(1.01, 1.40), variant=variant)
    return Fault(kind, node, start, None, magnitude=rng.uniform(0.30, 0.98))
