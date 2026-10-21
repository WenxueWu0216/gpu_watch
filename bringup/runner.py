"""Command execution abstraction so every check and adapter is testable with fakes."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one command."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        """True if the command exited 0."""
        return self.returncode == 0


class CommandRunner(Protocol):
    """Runs an argv and returns its result."""

    def run(self, argv: list[str], timeout_s: float = 600.0) -> CommandResult:
        """Execute ``argv``."""
        ...


class SubprocessRunner:
    """Executes commands on the local host (live backend)."""

    def run(self, argv: list[str], timeout_s: float = 600.0) -> CommandResult:
        """Run with a timeout; a missing binary or timeout becomes a failed result."""
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except FileNotFoundError as exc:
            return CommandResult(tuple(argv), 127, "", str(exc))
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout if isinstance(exc.stdout, str) else ""
            return CommandResult(tuple(argv), 124, out, f"timed out after {timeout_s}s")
        return CommandResult(tuple(argv), proc.returncode, proc.stdout, proc.stderr)


@dataclass
class RecordedAction:
    """A mutating scheduler command and whether it was actually executed."""

    command: str
    executed: bool
    returncode: int | None = None
    stderr: str = ""


@dataclass
class MutationGuard:
    """Wraps a runner; mutating commands only execute when ``dry_run`` is False.

    Read-only commands (e.g. ``scontrol show node``) always run so the gate can
    report current state even in dry-run mode.
    """

    runner: CommandRunner
    dry_run: bool = True
    actions: list[RecordedAction] = field(default_factory=list)

    def read(self, argv: list[str]) -> CommandResult:
        """Run a read-only command."""
        return self.runner.run(argv, timeout_s=60)

    def mutate(self, argv: list[str], timeout_s: float = 600.0) -> CommandResult:
        """Run (or, in dry-run, only record) a state-changing command."""
        cmd = shlex.join(argv)
        if self.dry_run:
            self.actions.append(RecordedAction(cmd, executed=False))
            return CommandResult(tuple(argv), 0, "", "dry-run: not executed")
        res = self.runner.run(argv, timeout_s=timeout_s)
        self.actions.append(RecordedAction(cmd, True, res.returncode, res.stderr.strip()))
        return res
