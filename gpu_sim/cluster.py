"""Deterministic simulator of an N-node x G-GPU synchronous data-parallel job.

Model (deliberately simple, but physically coherent):

* Every scrape interval (``tick_s``) the job runs synchronous DDP steps. Each
  node's *time-to-barrier* is the slowest of its GPUs' kernel time plus host
  (input pipeline) overhead. The step wall time is the slowest node plus the
  ring all-reduce time, which is bounded by the slowest NIC in the ring.
* GPU utilization is kernel-busy time / step wall time, so a straggler lowers
  utilization on *healthy* nodes (they wait at the barrier). This is why the
  detector keys on step time and uses utilization only as corroboration.
* Temperatures follow a first-order thermal model; above the slowdown
  temperature the GPU sets thermal throttle bits and drops SM clock, which in
  turn lengthens that node's step time.
* Benign noise that a real cluster also produces is included on purpose:
  per-node speed bias, sample jitter, short GC-like spikes, rare multi-scrape
  slowdowns, periodic cluster-wide checkpoint pauses, power-cap throttling
  and sporadic correctable (SBE) ECC errors. These drive the false-positive
  rate measured by ``make eval``.

All randomness comes from ``random.Random`` instances seeded from the run seed,
so a given ``(config, seed, faults)`` triple always yields identical output.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from gpu_sim.faults import Fault, FaultKind
from gpu_sim.model import (
    THROTTLE_GPU_IDLE,
    THROTTLE_HW_THERMAL,
    THROTTLE_SW_POWER_CAP,
    THROTTLE_SW_THERMAL,
    GPUSample,
    LogLine,
    NodeSample,
    Snapshot,
)

XID_MESSAGES: dict[int, str] = {
    48: "An uncorrectable double bit error (DBE) has been detected on GPU in the framebuffer",
    79: "GPU has fallen off the bus.",
    94: "Contained ECC error",
}


@dataclass(frozen=True)
class ClusterConfig:
    """Static parameters of the simulated cluster (defaults loosely H100-like)."""

    nodes: int = 8
    gpus_per_node: int = 8
    tick_s: float = 10.0
    base_gpu_step_s: float = 0.45
    base_host_s: float = 0.02
    base_allreduce_s: float = 0.04
    base_busbw_gb_s: float = 45.0
    base_net_tx_gbps: float = 160.0
    base_xmit_wait: float = 2000.0
    mem_total_mib: float = 81559.0
    max_sm_clock_mhz: float = 1980.0
    power_limit_w: float = 700.0
    slowdown_temp_c: float = 84.0
    hw_thermal_temp_c: float = 92.0
    step_jitter: float = 0.02
    node_bias_sigma: float = 0.01
    benign_noise: bool = True
    checkpoint_every_s: float = 600.0

    def node_names(self) -> list[str]:
        """Names of all nodes, ``node-00`` .. ``node-NN``."""
        return [f"node-{i:02d}" for i in range(self.nodes)]


@dataclass
class _GPUState:
    temp_c: float
    temp_bias: float
    power_bias: float
    speed_bias: float
    pci_bus: int
    sbe: int = 0
    dbe: int = 0
    xid: int = 0
    available: bool = True


@dataclass
class _NodeState:
    bias: float
    wait_bias: float
    spike_ticks: int = 0
    spike_factor: float = 1.0
    gpus: list[_GPUState] = field(default_factory=list)


class ClusterSim:
    """Seeded, reproducible cluster simulator.

    Example:
        >>> sim = ClusterSim(ClusterConfig(nodes=4), seed=1)
        >>> snap = sim.step()
        >>> len(snap.gpus)
        32
    """

    def __init__(
        self,
        config: ClusterConfig | None = None,
        seed: int = 0,
        faults: Sequence[Fault] = (),
    ) -> None:
        """Create a simulator; ``faults`` may also be added later via :meth:`inject`."""
        self.config = config or ClusterConfig()
        self.seed = seed
        self.faults: list[Fault] = list(faults)
        self.tick = 0
        self.job_hung = False
        self._fired: set[tuple[int, str]] = set()
        self._noise = random.Random(f"gpuwatch-noise-{seed}")
        self._benign = random.Random(f"gpuwatch-benign-{seed}")
        self._fault_rng = random.Random(f"gpuwatch-fault-{seed}")
        setup = random.Random(f"gpuwatch-setup-{seed}")
        cfg = self.config
        self._nodes: dict[str, _NodeState] = {}
        for name in cfg.node_names():
            node = _NodeState(
                bias=1.0 + setup.gauss(0.0, cfg.node_bias_sigma),
                wait_bias=1.0 + setup.gauss(0.0, 0.05),
            )
            for g in range(cfg.gpus_per_node):
                node.gpus.append(
                    _GPUState(
                        temp_c=35.0,
                        temp_bias=setup.gauss(0.0, 2.5),
                        power_bias=1.0 + setup.gauss(0.0, 0.04),
                        speed_bias=1.0 + setup.gauss(0.0, 0.004),
                        pci_bus=0x18 + 0x10 * g,
                    )
                )
            self._nodes[name] = node

    @property
    def now(self) -> float:
        """Simulated timestamp (seconds) of the most recent snapshot."""
        return self.tick * self.config.tick_s

    def inject(self, fault: Fault) -> None:
        """Add a fault at runtime (used by the HTTP server)."""
        if fault.node not in self._nodes:
            raise ValueError(f"unknown node {fault.node}")
        self.faults.append(fault)

    def run(self, duration_s: float) -> Iterator[Snapshot]:
        """Yield snapshots until ``duration_s`` of simulated time has elapsed."""
        while self.now + self.config.tick_s <= duration_s + 1e-9:
            yield self.step()

    # ------------------------------------------------------------------ core
    def _active(self, t: float, kind: FaultKind) -> list[Fault]:
        return [f for f in self.faults if f.kind is kind and t >= f.start_s]

    def step(self) -> Snapshot:
        """Advance one scrape interval and return the observed snapshot."""
        cfg = self.config
        self.tick += 1
        t = self.now
        dt = cfg.tick_s
        logs: list[LogLine] = []
        n = self._noise

        thermal = {(f.node, f.gpu): f for f in self._active(t, FaultKind.THERMAL)}
        stragglers = {f.node: f for f in self._active(t, FaultKind.STRAGGLER)}
        network = {f.node: f for f in self._active(t, FaultKind.NETWORK)}
        self._apply_discrete_faults(t, logs)

        # Per-GPU clock ratio from current thermal state (computed before the
        # step so a hot GPU slows its node within the same interval).
        clock_ratio: dict[tuple[str, int], float] = {}
        throttle_bits: dict[tuple[str, int], int] = {}
        for name, node in self._nodes.items():
            for g, gs in enumerate(node.gpus):
                ratio, bits = 1.0, 0
                tf = thermal.get((name, g))
                if tf is not None and gs.temp_c > cfg.slowdown_temp_c:
                    target = self._thermal_target(tf)
                    level = min(
                        1.0,
                        (gs.temp_c - cfg.slowdown_temp_c) / max(0.5, target - cfg.slowdown_temp_c),
                    )
                    ratio = 1.0 - tf.magnitude * level
                    bits |= THROTTLE_SW_THERMAL
                    if gs.temp_c >= cfg.hw_thermal_temp_c:
                        bits |= THROTTLE_HW_THERMAL
                clock_ratio[(name, g)] = ratio
                throttle_bits[(name, g)] = bits

        # Benign transient slowdowns (GC pauses, checkpoint uploads).
        if cfg.benign_noise:
            for node in self._nodes.values():
                if node.spike_ticks > 0:
                    node.spike_ticks -= 1
                    if node.spike_ticks == 0:
                        node.spike_factor = 1.0
                else:
                    r = self._benign.random()
                    if r < 0.0004:
                        node.spike_ticks = self._benign.randint(3, 5)
                        node.spike_factor = 1.0 + self._benign.uniform(0.06, 0.15)
                    elif r < 0.004:
                        node.spike_ticks = self._benign.randint(1, 2)
                        node.spike_factor = 1.0 + self._benign.uniform(0.10, 0.40)
        checkpoint = (
            cfg.benign_noise and cfg.checkpoint_every_s > 0 and (t % cfg.checkpoint_every_s) < dt
        )

        gpu_busy: dict[tuple[str, int], float] = {}
        compute: dict[str, float] = {}
        for name, node in self._nodes.items():
            sf = stragglers.get(name)
            jitter = max(0.5, 1.0 + n.gauss(0.0, cfg.step_jitter))
            busiest = 0.0
            for g, gs in enumerate(node.gpus):
                busy = cfg.base_gpu_step_s * gs.speed_bias / clock_ratio[(name, g)]
                if sf is not None and sf.variant == "gpu":
                    busy *= sf.magnitude
                busy *= node.bias * jitter * node.spike_factor
                gpu_busy[(name, g)] = busy
                busiest = max(busiest, busy)
            host = cfg.base_host_s * node.bias * jitter
            if sf is not None and sf.variant != "gpu":
                host += (cfg.base_gpu_step_s + cfg.base_host_s) * (sf.magnitude - 1.0)
            c = busiest + host
            if checkpoint:
                c *= 1.3
            compute[name] = c

        net_factor = min([f.magnitude for f in network.values()], default=1.0)
        allreduce = cfg.base_allreduce_s / net_factor * max(0.5, 1.0 + n.gauss(0.0, 0.03))
        wall = max(compute.values()) + allreduce
        busbw = cfg.base_busbw_gb_s * cfg.base_allreduce_s / allreduce

        gpus: list[GPUSample] = []
        nodes: list[NodeSample] = []
        for name, node in self._nodes.items():
            for g, gs in enumerate(node.gpus):
                gpus.append(
                    self._gpu_sample(
                        name,
                        g,
                        gs,
                        thermal.get((name, g)),
                        gpu_busy[(name, g)] / wall,
                        clock_ratio[(name, g)],
                        throttle_bits[(name, g)],
                        dt,
                    )
                )
            nf = network.get(name)
            wait = cfg.base_xmit_wait * node.wait_bias * max(0.1, 1.0 + n.gauss(0.0, 0.10))
            if nf is not None:
                wait *= 1.0 + 6.0 * (1.0 - nf.magnitude)
            if self.job_hung:
                nodes.append(
                    NodeSample(
                        name, None, max(0.0, n.gauss(0.2, 0.05)), max(0.0, wait * 0.01), None
                    )
                )
                continue
            tx = cfg.base_net_tx_gbps * (cfg.base_allreduce_s + cfg.base_gpu_step_s) / wall
            nodes.append(
                NodeSample(
                    node=name,
                    step_time_s=round(compute[name], 5),
                    net_tx_gbps=round(tx * max(0.5, 1.0 + n.gauss(0.0, 0.02)), 3),
                    net_xmit_wait=round(wait, 2),
                    nccl_busbw_gb_s=round(busbw * max(0.5, 1.0 + n.gauss(0.0, 0.01)), 3),
                )
            )
            if self.tick % 6 == 0:
                logs.append(
                    LogLine(
                        t,
                        name,
                        "train",
                        f"rank{int(name[-2:]) * cfg.gpus_per_node} step "
                        f"{self.tick * 20} step_time={compute[name]:.3f}s",
                    )
                )
        return Snapshot(ts=t, gpus=tuple(gpus), nodes=tuple(nodes), logs=tuple(logs))

    # --------------------------------------------------------------- helpers
    def _thermal_target(self, f: Fault) -> float:
        return self.config.slowdown_temp_c + 2.0 + 30.0 * f.magnitude

    def _apply_discrete_faults(self, t: float, logs: list[LogLine]) -> None:
        """Handle XID / ECC faults that change counters or job state."""
        cfg = self.config
        for idx, f in enumerate(self.faults):
            if t < f.start_s or f.gpu is None:
                continue
            gs = self._nodes[f.node].gpus[f.gpu]
            bus = f"PCI:0000:{gs.pci_bus:02x}:00"
            if f.kind is FaultKind.XID and (idx, "xid") not in self._fired:
                self._fired.add((idx, "xid"))
                code = int(f.variant or "79")
                gs.xid = code
                logs.append(
                    LogLine(
                        t,
                        f.node,
                        "kernel",
                        f"NVRM: Xid ({bus}): {code}, pid='<unknown>', "
                        f"name=<unknown>, {XID_MESSAGES.get(code, 'Unknown error')}",
                    )
                )
                if code == 79:
                    gs.available = False
                    self.job_hung = True
                    logs.append(
                        LogLine(
                            t + 1,
                            f.node,
                            "nccl",
                            "NCCL WARN Cuda failure 'unspecified launch failure'",
                        )
                    )
            elif f.kind is FaultKind.ECC and t < f.start_s + 90.0:
                lam = f.magnitude * cfg.tick_s / 10.0
                gs.sbe += _poisson(self._fault_rng, lam)
                if f.variant == "dbe" and t >= f.start_s + 20.0 and (idx, "dbe") not in self._fired:
                    self._fired.add((idx, "dbe"))
                    gs.dbe += 1 + self._fault_rng.randrange(2)
                    gs.xid = 48
                    logs.append(
                        LogLine(
                            t,
                            f.node,
                            "kernel",
                            f"NVRM: Xid ({bus}): 48, pid=4242, name=python, {XID_MESSAGES[48]}",
                        )
                    )

    def _gpu_sample(
        self,
        name: str,
        g: int,
        gs: _GPUState,
        thermal_fault: Fault | None,
        busy_frac: float,
        ratio: float,
        bits: int,
        dt: float,
    ) -> GPUSample:
        cfg = self.config
        n = self._noise
        if cfg.benign_noise and self._benign.random() < 2e-4:
            gs.sbe += 1
        if not gs.available:
            return GPUSample(
                name,
                g,
                0.0,
                0.0,
                cfg.mem_total_mib,
                0.0,
                0.0,
                0.0,
                0,
                gs.sbe,
                gs.dbe,
                gs.xid,
                0.0,
                available=False,
            )
        util = 0.0 if self.job_hung else min(100.0, max(0.0, 100.0 * busy_frac + n.gauss(0, 1.5)))
        target = 40.0 + 30.0 * util / 100.0 + gs.temp_bias
        if thermal_fault is not None:
            target = max(target, self._thermal_target(thermal_fault))
        alpha = 1.0 - math.exp(-dt / 60.0)
        gs.temp_c += (target - gs.temp_c) * alpha + n.gauss(0.0, 0.4)
        power = 90.0 + 660.0 * (util / 100.0) * ratio * gs.power_bias + n.gauss(0.0, 8.0)
        clock = cfg.max_sm_clock_mhz * ratio
        if power >= cfg.power_limit_w:
            power = cfg.power_limit_w + n.gauss(0.0, 1.0)
            clock *= 0.97
            bits |= THROTTLE_SW_POWER_CAP
        if util < 5.0:
            bits |= THROTTLE_GPU_IDLE
            clock = 345.0
        mem_used = 0.0 if self.job_hung else cfg.mem_total_mib * 0.86 + n.gauss(0.0, 150.0)
        nvlink = 0.0 if self.job_hung else 300.0 * util / 100.0 * max(0.5, 1 + n.gauss(0, 0.05))
        return GPUSample(
            node=name,
            gpu=g,
            util_pct=round(util, 2),
            mem_used_mib=round(mem_used, 1),
            mem_total_mib=cfg.mem_total_mib,
            temp_c=round(gs.temp_c, 2),
            power_w=round(max(0.0, power), 2),
            sm_clock_mhz=round(clock + n.gauss(0.0, 5.0), 1),
            throttle_reasons=bits,
            ecc_sbe_total=gs.sbe,
            ecc_dbe_total=gs.dbe,
            xid=gs.xid,
            nvlink_gbps=round(nvlink, 2),
        )


def _poisson(rng: random.Random, lam: float) -> int:
    """Sample a Poisson variate (Knuth for small lambda, normal approx above 30)."""
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1
