"""In-memory fakes of ``scontrol`` and ``kubectl`` used by tests and the sim backend.

They implement just enough of each CLI's behaviour (argument parsing, state
transitions, error codes) to verify the adapters issue correct commands.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bringup.runner import CommandResult

Handler = Callable[[list[str]], CommandResult]


@dataclass
class FakeRunner:
    """Dispatches argv[0] to a handler and records every call."""

    handlers: dict[str, Handler] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def run(self, argv: list[str], timeout_s: float = 600.0) -> CommandResult:
        """Record and dispatch; unknown binaries behave like 'command not found'."""
        self.calls.append(shlex.join(argv))
        h = self.handlers.get(argv[0])
        if h is None:
            return CommandResult(tuple(argv), 127, "", f"{argv[0]}: command not found")
        return h(argv)


@dataclass
class FakeSlurm:
    """Minimal ``scontrol show node`` / ``scontrol update`` state machine."""

    nodes: dict[str, dict[str, str]] = field(default_factory=dict)

    def add(self, node: str, state: str = "IDLE") -> None:
        """Register a node."""
        self.nodes[node] = {"State": state, "Reason": ""}

    def __call__(self, argv: list[str]) -> CommandResult:
        """Handle an ``scontrol`` invocation."""
        args = argv[1:]
        if args[:2] == ["show", "node"] and len(args) >= 3:
            node = args[2]
            if node not in self.nodes:
                return CommandResult(tuple(argv), 1, "", f"Node {node} not found")
            n = self.nodes[node]
            reason = f" Reason={n['Reason']}" if n["Reason"] else ""
            out = f"NodeName={node} Arch=x86_64 CPUAlloc=0 State={n['State']} Gres=gpu:8{reason}"
            return CommandResult(tuple(argv), 0, out + "\n")
        if args[:1] == ["update"]:
            kv = dict(a.split("=", 1) for a in args[1:] if "=" in a)
            node = kv.get("NodeName", "")
            if node not in self.nodes:
                return CommandResult(tuple(argv), 1, "", f"Invalid node name specified: {node}")
            state = kv.get("State", "").upper()
            n = self.nodes[node]
            if state == "DRAIN":
                if "Reason" not in kv:
                    return CommandResult(tuple(argv), 1, "", "You must specify a reason")
                base = n["State"].split("+")[0]
                n["State"], n["Reason"] = f"{base}+DRAIN", kv["Reason"]
            elif state == "RESUME":
                n["State"], n["Reason"] = n["State"].split("+")[0], ""
            else:
                return CommandResult(tuple(argv), 1, "", f"Invalid node state {state}")
            return CommandResult(tuple(argv), 0)
        return CommandResult(tuple(argv), 1, "", f"unsupported: {' '.join(args)}")


@dataclass
class FakeKubectl:
    """Minimal kubectl for nodes: get -o json, cordon, uncordon, label, annotate, drain."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    pods: dict[str, list[str]] = field(default_factory=dict)
    fail_drain: bool = False

    def add(self, node: str, pods: list[str] | None = None) -> None:
        """Register a node with optional running pods."""
        self.nodes[node] = {"metadata": {"name": node, "labels": {}, "annotations": {}}, "spec": {}}
        self.pods[node] = list(pods or [])

    def __call__(self, argv: list[str]) -> CommandResult:
        """Handle a ``kubectl`` invocation."""
        a = argv[1:]
        verb = a[0] if a else ""
        node = (
            a[2]
            if verb in ("get", "label", "annotate") and len(a) > 2
            else (a[1] if len(a) > 1 else "")
        )
        if node not in self.nodes:
            return CommandResult(
                tuple(argv), 1, "", f'Error from server (NotFound): nodes "{node}" not found'
            )
        n = self.nodes[node]
        if verb == "get":
            return CommandResult(tuple(argv), 0, json.dumps(n))
        if verb == "cordon":
            n["spec"]["unschedulable"] = True
            return CommandResult(tuple(argv), 0, f"node/{node} cordoned\n")
        if verb == "uncordon":
            n["spec"].pop("unschedulable", None)
            return CommandResult(tuple(argv), 0, f"node/{node} uncordoned\n")
        if verb in ("label", "annotate"):
            key = "labels" if verb == "label" else "annotations"
            for kv in a[3:]:
                if "=" in kv and not kv.startswith("--"):
                    k, v = kv.split("=", 1)
                    if k in n["metadata"][key] and "--overwrite" not in a:
                        return CommandResult(tuple(argv), 1, "", f"'{k}' already has a value")
                    n["metadata"][key][k] = v
            return CommandResult(tuple(argv), 0, f"node/{node} {verb}ed\n")
        if verb == "drain":
            if not n["spec"].get("unschedulable"):
                n["spec"]["unschedulable"] = True
            if self.fail_drain:
                return CommandResult(tuple(argv), 1, "", "error: timed out waiting for eviction")
            evicted = self.pods[node]
            self.pods[node] = []
            out = "".join(f"evicting pod default/{p}\n" for p in evicted)
            return CommandResult(tuple(argv), 0, out + f"node/{node} drained\n")
        return CommandResult(tuple(argv), 1, "", f"unsupported verb {verb}")
