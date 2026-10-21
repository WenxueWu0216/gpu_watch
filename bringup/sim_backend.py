"""Simulated node for the bring-up pipeline.

:class:`SimNodeRunner` answers ``nvidia-smi``, ``dcgmi diag``, ``gpu_burn``
and ``all_reduce_perf`` by running :class:`gpu_sim.ClusterSim` under load with
the node's injected faults and rendering the result in each tool's output
format, so the same parsers used on real hardware are exercised in CI.
The rendered text is synthetic: it follows the tools' documented formats but
was not captured from real GPUs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from bringup.runner import CommandResult
from gpu_sim import ClusterConfig, ClusterSim, Fault, FaultKind
from gpu_sim.model import THERMAL_THROTTLE_MASK, Snapshot

DRIVER_VERSION = "550.54.15"
NCCL_SIZES = [8 << 20, 16 << 20, 32 << 20, 64 << 20, 128 << 20, 256 << 20, 512 << 20, 1 << 30]


@dataclass
class SimNodeRunner:
    """Answers vendor-tool invocations for one simulated node."""

    node: str = "node-00"
    faults: list[Fault] = field(default_factory=list)
    seed: int = 0
    gpus: int = 8
    burn_s: float = 300.0
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Run the simulated burn-in window once and cache the observed state."""
        idx = 0
        if self.node.startswith("node-") and self.node[5:].isdigit():
            idx = int(self.node[5:])
        cfg = ClusterConfig(nodes=max(2, idx + 1), gpus_per_node=self.gpus, benign_noise=False)
        # Faults on this node are present from t=0 of the bring-up.
        faults = [replace(f, node=self.node, start_s=0.0) for f in self.faults]
        sim = ClusterSim(cfg, seed=self.seed, faults=faults)
        self._snaps: list[Snapshot] = list(sim.run(self.burn_s))
        self._net = min(
            [f.magnitude for f in self.faults if f.kind is FaultKind.NETWORK], default=1.0
        )
        self._base_busbw = cfg.base_busbw_gb_s

    def _gpus(self, snap: Snapshot) -> list:  # type: ignore[type-arg]
        return [g for g in snap.gpus if g.node == self.node]

    def run(self, argv: list[str], timeout_s: float = 600.0) -> CommandResult:
        """Dispatch on the tool name."""
        self.calls.append(" ".join(argv))
        tool = next(
            (
                a
                for a in argv
                if a.split("/")[-1] in ("nvidia-smi", "dcgmi", "gpu_burn", "all_reduce_perf")
            ),
            argv[0],
        )
        tool = tool.split("/")[-1]
        if tool == "nvidia-smi":
            return self._nvidia_smi(argv)
        if tool == "dcgmi":
            return self._dcgmi(argv)
        if tool == "gpu_burn":
            return self._gpu_burn(argv)
        if tool == "all_reduce_perf":
            return self._nccl(argv)
        return CommandResult(tuple(argv), 127, "", f"{argv[0]}: command not found (sim)")

    def _nvidia_smi(self, argv: list[str]) -> CommandResult:
        last = self._gpus(self._snaps[-1])
        lost = [g for g in last if not g.available]
        if lost:
            return CommandResult(
                tuple(argv),
                15,
                "",
                f"Unable to determine the device handle for GPU0000:"
                f"{0x18 + 0x10 * lost[0].gpu:02x}:00.0: Unknown Error\n",
            )
        lines = [
            f"{g.gpu}, NVIDIA H100 80GB HBM3, GPU-sim-{self.node}-{g.gpu}, {DRIVER_VERSION}, "
            f"00000000:{0x18 + 0x10 * g.gpu:02X}:00.0, Enabled, {g.ecc_dbe_total}"
            for g in last
        ]
        return CommandResult(tuple(argv), 0, "\n".join(lines) + "\n")

    def _dcgmi(self, argv: list[str]) -> CommandResult:
        last = self._gpus(self._snaps[-1])
        if any(not g.available for g in last):
            return CommandResult(
                tuple(argv),
                226,
                "",
                "Error: Unable to complete diagnostic for group 0. "
                "Return: (-17) Host engine error\n",
            )
        throttled = {
            g.gpu
            for s in self._snaps
            for g in self._gpus(s)
            if g.throttle_reasons & THERMAL_THROTTLE_MASK
        }
        first = {g.gpu: g for g in self._gpus(self._snaps[0])}
        mem, stress = [], []
        for g in last:
            sbe = g.ecc_sbe_total - first[g.gpu].ecc_sbe_total
            if g.ecc_dbe_total:
                mem.append(
                    {
                        "gpu_id": str(g.gpu),
                        "status": "Fail",
                        "info": f"{g.ecc_dbe_total} uncorrectable ECC (DBE) errors detected",
                    }
                )
            elif sbe > 50:
                mem.append(
                    {
                        "gpu_id": str(g.gpu),
                        "status": "Warn",
                        "info": f"{sbe} correctable ECC (SBE) errors during test",
                    }
                )
            else:
                mem.append({"gpu_id": str(g.gpu), "status": "Pass"})
            if g.gpu in throttled:
                stress.append(
                    {
                        "gpu_id": str(g.gpu),
                        "status": "Fail",
                        "info": f"Thermal violations detected; max temp {g.temp_c:.0f} C, "
                        f"SM clock {g.sm_clock_mhz:.0f} MHz",
                    }
                )
            else:
                stress.append({"gpu_id": str(g.gpu), "status": "Pass"})
        doc = {
            "version": "3.3.5 (simulated)",
            "DCGM GPU Diagnostic": {
                "test_categories": [
                    {
                        "category": "Deployment",
                        "tests": [
                            {"name": "Denylist", "results": [{"status": "Pass"}]},
                            {"name": "NVML Library", "results": [{"status": "Pass"}]},
                            {"name": "Persistence Mode", "results": [{"status": "Pass"}]},
                        ],
                    },
                    {"category": "Hardware", "tests": [{"name": "GPU Memory", "results": mem}]},
                    {
                        "category": "Stress",
                        "tests": [{"name": "Targeted Stress", "results": stress}],
                    },
                ]
            },
        }
        rc = 0 if all(r["status"] != "Fail" for r in mem + stress) else 226
        return CommandResult(tuple(argv), rc, json.dumps(doc, indent=1))

    def _gpu_burn(self, argv: list[str]) -> CommandResult:
        last = self._gpus(self._snaps[-1])
        secs = argv[-1] if argv[-1].isdigit() else "?"
        lines = [f"Burning for {secs} seconds."]
        for g in last:
            if not g.available:
                return CommandResult(
                    tuple(argv),
                    1,
                    "\n".join(lines),
                    "Couldn't init a GPU test: CUDA_ERROR_UNKNOWN\n",
                )
        lines.append(f"Tested {len(last)} GPUs:")
        for g in last:
            lines.append(f"\tGPU {g.gpu}: {'FAULTY' if g.ecc_dbe_total else 'OK'}")
        return CommandResult(tuple(argv), 0, "\n".join(lines) + "\n")

    def _nccl(self, argv: list[str]) -> CommandResult:
        last = self._gpus(self._snaps[-1])
        if any(not g.available for g in last):
            return CommandResult(
                tuple(argv),
                1,
                "",
                f"{self.node}: Test CUDA failure common.cu:1032 'unspecified launch failure'\n",
            )
        peak = self._base_busbw * self._net
        out = [
            f"# nThread 1 nGpus {self.gpus} minBytes {NCCL_SIZES[0]} maxBytes {NCCL_SIZES[-1]} "
            "step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0",
            "#",
            "#                                                              out-of-place"
            "                       in-place          ",
            "#       size         count      type   redop    root     time   algbw   busbw "
            "#wrong     time   algbw   busbw #wrong",
            "#        (B)    (elements)                               (us)  (GB/s)  (GB/s)  "
            "          (us)  (GB/s)  (GB/s)       ",
        ]
        vals = []
        n = self.gpus
        for i, size in enumerate(NCCL_SIZES):
            eff = 0.55 + 0.45 * i / (len(NCCL_SIZES) - 1)  # small messages are latency bound
            busbw = peak * eff
            algbw = busbw * n / (2 * (n - 1))
            t_us = size / (algbw * 1e3)
            vals.append(busbw)
            out.append(
                f"{size:12d} {size // 4:13d}     float     sum      -1 {t_us:8.1f} "
                f"{algbw:7.2f} {busbw:7.2f}      0 {t_us:8.1f} {algbw:7.2f} "
                f"{busbw:7.2f}      0"
            )
        out += [
            "# Out of bounds values : 0 OK",
            f"# Avg bus bandwidth    : {sum(vals) / len(vals):.3f} ",
            "#",
            "",
        ]
        return CommandResult(tuple(argv), 0, "\n".join(out))
