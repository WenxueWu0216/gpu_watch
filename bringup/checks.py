"""Node health checks: driver/GPU inventory, burn-in, NCCL all-reduce bandwidth.

Each check runs real vendor tools through a :class:`~bringup.runner.CommandRunner`
and parses their output. The same parsers are exercised in CI through the
simulated runner (``bringup/sim_backend.py``), which emits output in the
tools' documented formats.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from bringup.runner import CommandRunner


class Status(StrEnum):
    """Result of one check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Outcome of a check, with the evidence used to decide it."""

    name: str
    status: Status
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    command: str = ""

    @property
    def failed(self) -> bool:
        """True if the check failed."""
        return self.status is Status.FAIL


class Check(Protocol):
    """A health check."""

    name: str
    critical: bool

    def run(self, runner: CommandRunner) -> CheckResult:
        """Execute the check."""
        ...


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v))


NVIDIA_SMI_FIELDS = (
    "index,name,uuid,driver_version,pci.bus_id,persistence_mode,"
    "ecc.errors.uncorrected.volatile.total"
)


@dataclass
class DriverCheck:
    """GPU inventory and driver sanity via ``nvidia-smi --query-gpu``.

    Fails if nvidia-smi errors (e.g. a GPU fell off the bus), the GPU count
    differs from ``expected_gpus``, the driver is older than ``min_driver``, or
    any GPU reports uncorrectable volatile ECC errors. Persistence mode off is
    a warning.
    """

    expected_gpus: int = 8
    min_driver: str = "535.0"
    name: str = "driver"
    critical: bool = True

    def command(self) -> list[str]:
        """The nvidia-smi invocation."""
        return ["nvidia-smi", f"--query-gpu={NVIDIA_SMI_FIELDS}", "--format=csv,noheader,nounits"]

    def run(self, runner: CommandRunner) -> CheckResult:
        """Run nvidia-smi and evaluate the inventory."""
        argv = self.command()
        t0 = time.monotonic()
        res = runner.run(argv, timeout_s=60)
        cmd = shlex.join(argv)
        if not res.ok:
            return CheckResult(
                self.name,
                Status.FAIL,
                f"nvidia-smi failed (rc={res.returncode})",
                {"stderr": res.stderr.strip()[-2000:], "stdout": res.stdout.strip()[-2000:]},
                time.monotonic() - t0,
                cmd,
            )
        gpus = parse_nvidia_smi_csv(res.stdout)
        problems: list[str] = []
        warnings: list[str] = []
        if len(gpus) != self.expected_gpus:
            problems.append(f"found {len(gpus)} GPUs, expected {self.expected_gpus}")
        for g in gpus:
            if _version_tuple(g["driver_version"]) < _version_tuple(self.min_driver):
                problems.append(
                    f"gpu {g['index']}: driver {g['driver_version']} < {self.min_driver}"
                )
            ecc = g["ecc_uncorrected"]
            if ecc is not None and ecc > 0:
                problems.append(f"gpu {g['index']}: {ecc} uncorrectable ECC errors")
            if g["persistence_mode"].lower() != "enabled":
                warnings.append(f"gpu {g['index']}: persistence mode {g['persistence_mode']}")
        status = Status.FAIL if problems else (Status.WARN if warnings else Status.PASS)
        summary = "; ".join(problems or warnings) or (
            f"{len(gpus)} GPUs, driver {gpus[0]['driver_version'] if gpus else '?'}"
        )
        return CheckResult(
            self.name,
            status,
            summary,
            {"gpus": gpus, "problems": problems, "warnings": warnings},
            time.monotonic() - t0,
            cmd,
        )


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu=<NVIDIA_SMI_FIELDS> --format=csv,noheader,nounits``."""
    out = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        ecc_raw = parts[6]
        ecc = int(ecc_raw) if ecc_raw.isdigit() else None  # "[N/A]" when ECC disabled
        out.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "driver_version": parts[3],
                "pci_bus_id": parts[4],
                "persistence_mode": parts[5],
                "ecc_uncorrected": ecc,
            }
        )
    return out


@dataclass
class BurnInCheck:
    """GPU burn-in.

    ``tool="dcgm"`` runs ``dcgmi diag -r <level> -j`` (DCGM diagnostics; level 3
    includes targeted stress/power/memory tests) and fails on any test result
    with status Fail. ``tool="gpu_burn"`` runs ``gpu_burn <seconds>`` and fails
    on any ``GPU n: FAULTY`` line.
    """

    tool: str = "dcgm"
    level: int = 3
    duration_s: int = 120
    gpu_burn_path: str = "gpu_burn"
    name: str = "burn_in"
    critical: bool = True

    def command(self) -> list[str]:
        """The burn-in invocation."""
        if self.tool == "dcgm":
            return ["dcgmi", "diag", "-r", str(self.level), "-j"]
        if self.tool == "gpu_burn":
            return [self.gpu_burn_path, str(self.duration_s)]
        raise ValueError(f"unknown burn-in tool {self.tool!r}")

    def run(self, runner: CommandRunner) -> CheckResult:
        """Run the burn-in and parse its verdicts."""
        argv = self.command()
        t0 = time.monotonic()
        res = runner.run(argv, timeout_s=max(1800.0, self.duration_s * 3.0))
        cmd = shlex.join(argv)
        if self.tool == "dcgm":
            try:
                results = parse_dcgm_diag_json(res.stdout)
            except ValueError as exc:
                return CheckResult(
                    self.name,
                    Status.FAIL,
                    f"could not parse dcgmi output: {exc}",
                    {"rc": res.returncode, "stderr": res.stderr[-2000:]},
                    time.monotonic() - t0,
                    cmd,
                )
            failed = [r for r in results if r["status"].lower().startswith("fail")]
            warned = [r for r in results if r["status"].lower().startswith("warn")]
            if not results:
                status, summary = Status.FAIL, "dcgmi returned no test results"
            elif failed:
                status = Status.FAIL
                summary = "; ".join(
                    f"{r['test']} gpu {r['gpu_id']}: {r['info']}".strip() for r in failed
                )
            elif warned:
                status, summary = Status.WARN, f"{len(warned)} DCGM warnings"
            else:
                status, summary = Status.PASS, f"{len(results)} DCGM diag results passed"
            return CheckResult(
                self.name,
                status,
                summary,
                {"failed": failed, "warnings": warned, "results": len(results)},
                time.monotonic() - t0,
                cmd,
            )
        verdicts = parse_gpu_burn(res.stdout)
        faulty = sorted(g for g, ok in verdicts.items() if not ok)
        if not verdicts:
            status, summary = Status.FAIL, f"gpu_burn produced no verdicts (rc={res.returncode})"
        elif faulty:
            status, summary = Status.FAIL, f"gpu_burn FAULTY on GPUs {faulty}"
        else:
            status, summary = Status.PASS, f"gpu_burn OK on {len(verdicts)} GPUs"
        return CheckResult(
            self.name, status, summary, {"verdicts": verdicts}, time.monotonic() - t0, cmd
        )


def parse_dcgm_diag_json(text: str) -> list[dict[str, str]]:
    """Extract (category, test, gpu_id, status, info) rows from ``dcgmi diag -j``.

    The JSON layout differs slightly between DCGM releases, so this walks the
    document and collects every object carrying a ``status`` field, inheriting
    the nearest enclosing test name / category.
    """
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    rows: list[dict[str, str]] = []

    def info_of(obj: dict[str, Any]) -> str:
        for key in ("info", "warnings", "warning", "error"):
            val = obj.get(key)
            if isinstance(val, list):
                val = "; ".join(
                    str(v.get("message", v)) if isinstance(v, dict) else str(v) for v in val
                )
            if val:
                return str(val)
        return ""

    def walk(obj: Any, category: str, test: str) -> None:
        if isinstance(obj, dict):
            category = str(obj.get("category", category))
            test = str(obj.get("name", obj.get("test_name", test)))
            if "status" in obj and isinstance(obj["status"], str):
                rows.append(
                    {
                        "category": category,
                        "test": test,
                        "gpu_id": str(obj.get("gpu_id", obj.get("gpu_ids", "all"))),
                        "status": obj["status"],
                        "info": info_of(obj),
                    }
                )
            for v in obj.values():
                if isinstance(v, dict | list):
                    walk(v, category, test)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, category, test)

    walk(doc, "", "")
    return rows


_GPU_BURN_RE = re.compile(r"GPU\s+(\d+):\s+(OK|FAULTY)")


def parse_gpu_burn(text: str) -> dict[int, bool]:
    """Parse final ``GPU n: OK|FAULTY`` verdict lines from gpu-burn."""
    return {int(g): v == "OK" for g, v in _GPU_BURN_RE.findall(text)}


@dataclass
class NcclCheck:
    """NCCL all-reduce bandwidth via nccl-tests ``all_reduce_perf``.

    ``command_template`` may wrap the binary in ``srun``/``mpirun`` for
    multi-node tests. Fails if the reported average bus bandwidth is below
    ``min_busbw_gb_s`` or if any row reports ``#wrong`` > 0 (data corruption).
    """

    min_busbw_gb_s: float
    command_template: str = "all_reduce_perf -b 8M -e 1G -f 2 -g 8"
    name: str = "nccl_allreduce"
    critical: bool = True

    def command(self) -> list[str]:
        """The nccl-tests invocation."""
        return shlex.split(self.command_template)

    def run(self, runner: CommandRunner) -> CheckResult:
        """Run nccl-tests and compare bus bandwidth to the threshold."""
        argv = self.command()
        t0 = time.monotonic()
        res = runner.run(argv, timeout_s=1800)
        cmd = shlex.join(argv)
        parsed = parse_nccl_tests(res.stdout)
        avg = parsed["avg_busbw_gb_s"]
        details: dict[str, Any] = dict(
            parsed, threshold_gb_s=self.min_busbw_gb_s, rc=res.returncode
        )
        if not res.ok or avg is None:
            return CheckResult(
                self.name,
                Status.FAIL,
                f"nccl-tests failed (rc={res.returncode}): "
                f"{res.stderr.strip()[-300:] or 'no bandwidth reported'}",
                details,
                time.monotonic() - t0,
                cmd,
            )
        if parsed["wrong"] > 0:
            return CheckResult(
                self.name,
                Status.FAIL,
                f"nccl-tests reported {parsed['wrong']} wrong results",
                details,
                time.monotonic() - t0,
                cmd,
            )
        if avg < self.min_busbw_gb_s:
            return CheckResult(
                self.name,
                Status.FAIL,
                f"avg busbw {avg:.1f} GB/s < threshold {self.min_busbw_gb_s:.1f}",
                details,
                time.monotonic() - t0,
                cmd,
            )
        return CheckResult(
            self.name,
            Status.PASS,
            f"avg busbw {avg:.1f} GB/s >= {self.min_busbw_gb_s:.1f}",
            details,
            time.monotonic() - t0,
            cmd,
        )


_AVG_RE = re.compile(r"#\s*Avg bus bandwidth\s*:\s*([\d.]+)")


def parse_nccl_tests(text: str) -> dict[str, Any]:
    """Parse nccl-tests stdout: average busbw, per-size rows, wrong-result count."""
    m = _AVG_RE.search(text)
    rows = []
    wrong = 0
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # size count type redop root time algbw busbw #wrong (out-of-place) [x4 in-place]
        if len(parts) >= 9 and parts[0].isdigit():
            try:
                row: dict[str, float] = {
                    "size_bytes": int(parts[0]),
                    "busbw_oop": float(parts[7]),
                    "wrong_oop": int(parts[8]) if parts[8].isdigit() else 0,
                }
            except ValueError:
                continue
            wrong += int(row["wrong_oop"])
            if len(parts) >= 13:
                row["busbw_ip"] = float(parts[11])
                wrong += int(parts[12]) if parts[12].isdigit() else 0
            rows.append(row)
    return {"avg_busbw_gb_s": float(m.group(1)) if m else None, "rows": rows, "wrong": wrong}
