"""The health gate: run checks, decide, drain or admit, write a diagnosis report."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bringup.checks import Check, CheckResult, Status
from bringup.runner import CommandRunner, RecordedAction
from bringup.schedulers import SchedulerAdapter

# Failing check -> (likely cause, next step, runbook section).
DIAGNOSIS: dict[str, tuple[str, str, str]] = {
    "driver": (
        "GPU missing from the bus, driver mismatch, or uncorrectable ECC errors",
        "Check `dmesg | grep -i xid`, `nvidia-smi -q -d ECC,ROW_REMAPPER`; reseat/RMA the GPU "
        "or reinstall the driver; reboot to clear volatile ECC state before re-gating.",
        "runbook.md#1-xid-errors",
    ),
    "burn_in": (
        "GPU fails stress: thermal throttling, memory errors, or compute faults under load",
        "Read the failing DCGM test info; for thermal check fans/airflow/inlet temperature "
        "(`nvidia-smi -q -d TEMPERATURE,PERFORMANCE`); for memory errors follow the ECC runbook.",
        "runbook.md#3-thermal-throttle",
    ),
    "nccl_allreduce": (
        "Collective bandwidth below threshold: degraded NVLink/NVSwitch, NIC link speed, "
        "PCIe/ACS settings or a congested fabric port",
        "Run `nvidia-smi nvlink -s`, `ibstat`/`ibv_devinfo` (link rate/width), check "
        "`port_xmit_wait`/symbol errors; rerun nccl-tests with NCCL_DEBUG=INFO.",
        "runbook.md#5-nccl-slow-link",
    ),
}

ECC_DIAGNOSIS = (
    "Uncorrectable (double-bit) or excessive correctable ECC errors in GPU memory",
    "Check `nvidia-smi -q -d ECC,ROW_REMAPPER` for pending row remaps; reset the GPU "
    "(`nvidia-smi -r`) or reboot to apply remapping, then re-gate; RMA if remapping failed.",
    "runbook.md#2-ecc-errors",
)


@dataclass
class GateResult:
    """Everything decided for one node."""

    node: str
    passed: bool
    checks: list[CheckResult]
    decision: str
    scheduler: str
    scheduler_state_before: str
    scheduler_state_after: str
    actions: list[RecordedAction] = field(default_factory=list)
    dry_run: bool = True
    started_at: float = 0.0
    diagnosis: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict."""
        d = asdict(self)
        for c in d["checks"]:
            c["status"] = str(c["status"])
        return d


@dataclass
class HealthGate:
    """Runs checks in order and gates the node.

    A failing *critical* check short-circuits the remaining checks (e.g. no
    point running nccl-tests on a node whose GPU fell off the bus). Any FAIL
    drains the node; all PASS/WARN admits it. Scheduler mutations go through
    the adapter's :class:`~bringup.runner.MutationGuard`, so dry-run (the
    default) records the commands without executing them.
    """

    checks: Sequence[Check]
    runner: CommandRunner
    scheduler: SchedulerAdapter
    report_dir: Path | None = None

    def gate(self, node: str) -> GateResult:
        """Gate ``node`` and (optionally) write ``<report_dir>/<node>.{json,md}``."""
        from tracing.otel import get_meter, get_tracer

        tracer = get_tracer("gpuwatch.bringup")
        started = time.time()
        with tracer.start_as_current_span("bringup.gate") as span:
            span.set_attribute("gpuwatch.node", node)
            span.set_attribute("gpuwatch.scheduler", self.scheduler.name)
            span.set_attribute("gpuwatch.dry_run", self.scheduler.guard.dry_run)
            before = self.scheduler.state(node)
            results: list[CheckResult] = []
            blocked = False
            for check in self.checks:
                if blocked:
                    results.append(
                        CheckResult(
                            check.name, Status.SKIPPED, "skipped after earlier critical failure"
                        )
                    )
                    continue
                with tracer.start_as_current_span(f"bringup.check.{check.name}") as cs:
                    res = check.run(self.runner)
                    cs.set_attribute("gpuwatch.check.status", str(res.status))
                    cs.set_attribute("gpuwatch.check.summary", res.summary[:500])
                    cs.set_attribute("gpuwatch.check.command", res.command)
                results.append(res)
                if res.failed and check.critical:
                    blocked = True
            failed = [r for r in results if r.failed]
            passed = not failed
            if passed:
                ok = self.scheduler.admit(node)
                decision = "admit" if ok else "admit-failed"
            else:
                reason = "; ".join(f"{r.name}: {r.summary}" for r in failed)
                ok = self.scheduler.drain(node, reason)
                decision = "drain" if ok else "drain-failed"
            after = self.scheduler.state(node)
            span.set_attribute("gpuwatch.gate.passed", passed)
            span.set_attribute("gpuwatch.gate.decision", decision)
            get_meter("gpuwatch.bringup").create_counter(
                "gpuwatch.bringup.gate", description="Node gate outcomes"
            ).add(1, {"decision": decision, "scheduler": self.scheduler.name})
        diagnosis = []
        for r in failed:
            cause, step, ref = DIAGNOSIS.get(r.name, ("unknown", "inspect details", ""))
            if "ECC" in r.summary:
                cause, step, ref = ECC_DIAGNOSIS
            diagnosis.append(
                {
                    "check": r.name,
                    "finding": r.summary,
                    "likely_cause": cause,
                    "next_step": step,
                    "runbook": ref,
                }
            )
        result = GateResult(
            node=node,
            passed=passed,
            checks=results,
            decision=decision,
            scheduler=self.scheduler.name,
            scheduler_state_before=before,
            scheduler_state_after=after,
            actions=list(self.scheduler.guard.actions),
            dry_run=self.scheduler.guard.dry_run,
            started_at=started,
            diagnosis=diagnosis,
        )
        if self.report_dir is not None:
            write_report(result, self.report_dir)
        return result


def write_report(result: GateResult, report_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown diagnosis reports; returns their paths."""
    report_dir.mkdir(parents=True, exist_ok=True)
    jpath = report_dir / f"{result.node}.json"
    mpath = report_dir / f"{result.node}.md"
    jpath.write_text(json.dumps(result.to_dict(), indent=2, default=str) + "\n")
    mpath.write_text(render_markdown(result))
    return jpath, mpath


def render_markdown(r: GateResult) -> str:
    """Human-readable diagnosis report."""
    icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skipped": "SKIP"}
    lines = [
        f"# Node health gate: {r.node}",
        "",
        f"- Result: **{'PASSED' if r.passed else 'FAILED'}** -> decision `{r.decision}`"
        f"{' (DRY RUN: no scheduler changes made)' if r.dry_run else ''}",
        f"- Scheduler: {r.scheduler} (state before: `{r.scheduler_state_before}`, "
        f"after: `{r.scheduler_state_after}`)",
        f"- Started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(r.started_at))}",
        "",
        "## Checks",
        "",
        "| Check | Status | Summary | Command |",
        "|---|---|---|---|",
    ]
    for c in r.checks:
        lines.append(f"| {c.name} | {icon[str(c.status)]} | {c.summary} | `{c.command}` |")
    if r.diagnosis:
        lines += ["", "## Diagnosis", ""]
        for d in r.diagnosis:
            lines += [
                f"### {d['check']}",
                f"- Finding: {d['finding']}",
                f"- Likely cause: {d['likely_cause']}",
                f"- Next step: {d['next_step']}",
                f"- Runbook: {d['runbook']}",
                "",
            ]
    if r.actions:
        lines += ["", "## Scheduler actions", ""]
        for a in r.actions:
            status = "executed" if a.executed else "dry-run (not executed)"
            lines.append(
                f"- `{a.command}` - {status}"
                + (f", rc={a.returncode}" if a.returncode is not None else "")
            )
    return "\n".join(lines) + "\n"
