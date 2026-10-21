r"""CLI: gate a node before it joins the pool.

Examples::

    # simulated node with an injected ECC fault, Slurm adapter against a fake
    python -m bringup --backend sim --node node-03 --sim-fault ecc:node-03:2@0,variant=dbe \
        --scheduler slurm

    # real node (run ON the node); prints drain commands but does not execute them
    python -m bringup --backend live --node $(hostname) --scheduler slurm --min-busbw 350

    # actually drain on failure
    python -m bringup --backend live --node $(hostname) --scheduler k8s --min-busbw 350 --execute

Exit codes: 0 = admitted, 2 = failed the gate, 1 = usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bringup.checks import BurnInCheck, Check, DriverCheck, NcclCheck
from bringup.fakes import FakeKubectl, FakeRunner, FakeSlurm
from bringup.pipeline import HealthGate
from bringup.runner import CommandRunner, MutationGuard, SubprocessRunner
from bringup.schedulers import K8sAdapter, NoopAdapter, SchedulerAdapter, SlurmAdapter
from bringup.sim_backend import SimNodeRunner
from gpu_sim import Fault
from tracing.otel import setup_telemetry

SIM_MIN_BUSBW = 30.0  # sim healthy avg busbw is ~35 GB/s (45 GB/s peak)


def build(args: argparse.Namespace) -> HealthGate:
    """Wire checks, runners and the scheduler adapter from CLI args."""
    tool_runner: CommandRunner
    sched_runner: CommandRunner
    wanted = {c.strip() for c in args.checks.split(",") if c.strip()}
    unknown = wanted - {"driver", "burn_in", "nccl"}
    if unknown:
        raise SystemExit(f"unknown --checks {sorted(unknown)}")
    if args.backend == "sim":
        tool_runner = SimNodeRunner(
            node=args.node,
            faults=[Fault.parse(f) for f in args.sim_fault],
            seed=args.seed,
            gpus=args.expected_gpus,
        )
        slurm, kube = FakeSlurm(), FakeKubectl()
        slurm.add(args.node)
        kube.add(args.node, pods=["ddp-bench-0"])
        sched_runner = FakeRunner({"scontrol": slurm, "kubectl": kube})
        min_busbw = args.min_busbw if args.min_busbw is not None else SIM_MIN_BUSBW
    else:
        if args.min_busbw is None and "nccl" in wanted:
            raise SystemExit(
                "--min-busbw is required for --backend live (GB/s; it depends on "
                "GPU model and interconnect, see README)"
            )
        tool_runner = sched_runner = SubprocessRunner()
        min_busbw = args.min_busbw or 0.0
    checks: list[Check] = []
    if "driver" in wanted:
        checks.append(DriverCheck(args.expected_gpus, args.min_driver))
    if "burn_in" in wanted and args.burn_in != "none":
        checks.append(
            BurnInCheck(tool=args.burn_in, level=args.dcgm_level, duration_s=args.burn_seconds)
        )
    if "nccl" in wanted:
        checks.append(NcclCheck(min_busbw, command_template=args.nccl_cmd))
    guard = MutationGuard(sched_runner, dry_run=not args.execute)
    scheduler: SchedulerAdapter
    if args.scheduler == "slurm":
        scheduler = SlurmAdapter(guard)
    elif args.scheduler == "k8s":
        scheduler = K8sAdapter(guard)
    else:
        scheduler = NoopAdapter(guard)
    return HealthGate(checks, tool_runner, scheduler, Path(args.report_dir))


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(
        prog="bringup", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--node", required=True)
    p.add_argument("--backend", choices=["sim", "live"], default="sim")
    p.add_argument("--scheduler", choices=["slurm", "k8s", "none"], default="none")
    p.add_argument(
        "--execute", action="store_true", help="really run drain/resume commands (default: dry-run)"
    )
    p.add_argument("--report-dir", default="reports")
    p.add_argument(
        "--checks",
        default="driver,burn_in,nccl",
        help="comma-separated subset of driver,burn_in,nccl",
    )
    p.add_argument("--expected-gpus", type=int, default=8)
    p.add_argument("--min-driver", default="535.0")
    p.add_argument("--burn-in", choices=["dcgm", "gpu_burn", "none"], default="dcgm")
    p.add_argument("--dcgm-level", type=int, default=3)
    p.add_argument("--burn-seconds", type=int, default=120)
    p.add_argument("--min-busbw", type=float, help="all-reduce avg busbw threshold, GB/s")
    p.add_argument("--nccl-cmd", default="all_reduce_perf -b 8M -e 1G -f 2 -g 8")
    p.add_argument(
        "--sim-fault",
        action="append",
        default=[],
        help="sim backend: fault spec on this node (repeatable)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--otlp-endpoint")
    args = p.parse_args(argv)
    setup_telemetry("gpuwatch-bringup", args.otlp_endpoint)
    result = build(args).gate(args.node)
    for c in result.checks:
        sys.stdout.write(f"[{str(c.status).upper():7}] {c.name}: {c.summary}\n")
    for a in result.actions:
        sys.stdout.write(f"  {'EXEC' if a.executed else 'DRY-RUN'}: {a.command}\n")
    sys.stdout.write(f"decision: {result.decision}; report: {args.report_dir}/{args.node}.md\n")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
