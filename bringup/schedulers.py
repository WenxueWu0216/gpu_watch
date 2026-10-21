"""Scheduler adapters that take a node out of (or admit it into) the pool."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from bringup.runner import MutationGuard

LABEL_HEALTH = "gpuwatch.io/health"
ANNOTATION_REASON = "gpuwatch.io/reason"


class SchedulerAdapter(Protocol):
    """Drain / admit operations for one scheduler."""

    name: str
    guard: MutationGuard

    def state(self, node: str) -> str:
        """Current scheduler state string for ``node``."""
        ...

    def drain(self, node: str, reason: str) -> bool:
        """Remove ``node`` from scheduling; returns True on success (or dry-run)."""
        ...

    def admit(self, node: str) -> bool:
        """Return ``node`` to the schedulable pool."""
        ...


def _clean_reason(reason: str, limit: int = 200) -> str:
    return re.sub(r"[^\w .:/=+%-]", "_", reason)[:limit]


@dataclass
class SlurmAdapter:
    """Slurm via ``scontrol`` (requires SlurmUser / operator privileges)."""

    guard: MutationGuard
    name: str = "slurm"

    def state(self, node: str) -> str:
        """Parse ``State=`` from ``scontrol show node <node> -o``."""
        res = self.guard.read(["scontrol", "show", "node", node, "-o"])
        if not res.ok:
            return f"unknown ({res.stderr.strip()})"
        m = re.search(r"\bState=(\S+)", res.stdout)
        return m.group(1) if m else "unknown"

    def drain(self, node: str, reason: str) -> bool:
        """``scontrol update NodeName=<node> State=DRAIN Reason=<reason>``.

        Running jobs finish; no new jobs are scheduled on the node.
        """
        res = self.guard.mutate(
            [
                "scontrol",
                "update",
                f"NodeName={node}",
                "State=DRAIN",
                f"Reason=gpuwatch: {_clean_reason(reason)}",
            ]
        )
        return res.ok

    def admit(self, node: str) -> bool:
        """``scontrol update NodeName=<node> State=RESUME`` (only if drained/down)."""
        st = self.state(node).upper()
        if not any(s in st for s in ("DRAIN", "DOWN", "FAIL")):
            return True
        return self.guard.mutate(["scontrol", "update", f"NodeName={node}", "State=RESUME"]).ok


@dataclass
class K8sAdapter:
    """Kubernetes via ``kubectl``: cordon + label + drain, or uncordon."""

    guard: MutationGuard
    drain_timeout: str = "600s"
    name: str = "k8s"

    def state(self, node: str) -> str:
        """``schedulable`` / ``cordoned`` plus the GPUWatch health label."""
        res = self.guard.read(["kubectl", "get", "node", node, "-o", "json"])
        if not res.ok:
            return f"unknown ({res.stderr.strip()})"
        obj = json.loads(res.stdout)
        cordoned = bool(obj.get("spec", {}).get("unschedulable", False))
        label = obj.get("metadata", {}).get("labels", {}).get(LABEL_HEALTH, "none")
        return f"{'cordoned' if cordoned else 'schedulable'},health={label}"

    def drain(self, node: str, reason: str) -> bool:
        """Cordon, label/annotate with the reason, then evict pods."""
        ok = self.guard.mutate(["kubectl", "cordon", node]).ok
        ok &= self.guard.mutate(
            ["kubectl", "label", "node", node, f"{LABEL_HEALTH}=failed", "--overwrite"]
        ).ok
        ok &= self.guard.mutate(
            [
                "kubectl",
                "annotate",
                "node",
                node,
                f"{ANNOTATION_REASON}={_clean_reason(reason)}",
                "--overwrite",
            ]
        ).ok
        ok &= self.guard.mutate(
            [
                "kubectl",
                "drain",
                node,
                "--ignore-daemonsets",
                "--delete-emptydir-data",
                f"--timeout={self.drain_timeout}",
            ],
            timeout_s=900,
        ).ok
        return ok

    def admit(self, node: str) -> bool:
        """Label healthy and uncordon."""
        ok = self.guard.mutate(
            ["kubectl", "label", "node", node, f"{LABEL_HEALTH}=passed", "--overwrite"]
        ).ok
        ok &= self.guard.mutate(["kubectl", "uncordon", node]).ok
        return ok


@dataclass
class NoopAdapter:
    """No scheduler integration (report only)."""

    guard: MutationGuard
    name: str = "none"

    def state(self, node: str) -> str:
        """Always ``n/a``."""
        return "n/a"

    def drain(self, node: str, reason: str) -> bool:
        """Do nothing."""
        return True

    def admit(self, node: str) -> bool:
        """Do nothing."""
        return True
