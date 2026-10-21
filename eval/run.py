"""Fault-injection evaluation of the detector on the simulator.

For each fault type, ``runs_per_fault`` simulations are run with distinct
seeds; each injects exactly one randomly drawn fault (target, onset and
severity drawn from the seed). ``healthy_runs`` additional simulations inject
nothing. The detector is scored as follows:

* **Detected**: a *firing* event of an expected kind on the faulted node (and
  GPU, for GPU-scoped faults) with timestamp >= fault onset.
* **Time to detect (TTD)**: first such event timestamp - fault onset, in
  simulated seconds (scrape interval 10 s, so TTD resolution is 10 s).
* **Healthy FP rate (per type)**: fraction of healthy runs with >= 1 firing
  event of that type's expected kinds.
* **Off-target events**: firing events in faulted runs on a different node
  than the fault (cluster-level busbw events are not counted for network
  faults, where they are expected).
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from detector import Detector, DetectorConfig, EventKind
from gpu_sim import ClusterConfig, ClusterSim, Fault, FaultKind, random_fault

HERE = Path(__file__).parent
EXPECTED: dict[FaultKind, frozenset[EventKind]] = {
    FaultKind.THERMAL: frozenset({EventKind.THERMAL}),
    FaultKind.XID: frozenset({EventKind.XID, EventKind.GPU_MISSING}),
    FaultKind.ECC: frozenset({EventKind.ECC}),
    FaultKind.STRAGGLER: frozenset({EventKind.STRAGGLER}),
    FaultKind.NETWORK: frozenset({EventKind.SLOW_LINK}),
}
FAULT_ORDER = [
    FaultKind.THERMAL,
    FaultKind.XID,
    FaultKind.ECC,
    FaultKind.STRAGGLER,
    FaultKind.NETWORK,
]

# Severity buckets used to show how detection depends on fault magnitude.
BUCKETS: dict[FaultKind, list[tuple[str, float, float]]] = {
    FaultKind.THERMAL: [
        ("clock -2..8%", 0.02, 0.08),
        ("clock -8..15%", 0.08, 0.15),
        ("clock -15..35%", 0.15, 0.35),
    ],
    FaultKind.ECC: [
        ("SBE <2/scrape", 0.0, 2.0),
        ("SBE 2..10/scrape", 2.0, 10.0),
        ("SBE >=10/scrape", 10.0, 1e9),
    ],
    FaultKind.STRAGGLER: [
        ("slowdown 1-5%", 1.0, 1.05),
        ("slowdown 5-10%", 1.05, 1.10),
        ("slowdown 10-20%", 1.10, 1.20),
        ("slowdown 20-40%", 1.20, 1.41),
    ],
    FaultKind.NETWORK: [
        ("bw x0.90-0.98", 0.90, 0.99),
        ("bw x0.70-0.90", 0.70, 0.90),
        ("bw x0.30-0.70", 0.30, 0.70),
    ],
}


@dataclass(frozen=True)
class RunResult:
    """Outcome of one simulated run."""

    kind: str
    seed: int
    fault: str
    magnitude: float
    variant: str
    detected: bool
    ttd_s: float | None
    off_target: int
    event_kinds: tuple[str, ...]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the evaluation protocol."""
    cfg: dict[str, Any] = yaml.safe_load((path or HERE / "config.yaml").read_text())
    return cfg


def _run_one(args: tuple[str | None, int, dict[str, Any]]) -> RunResult:
    kind_name, seed, cfg = args
    ccfg = ClusterConfig(nodes=cfg["nodes"], gpus_per_node=cfg["gpus_per_node"])
    fault: Fault | None = None
    if kind_name is not None:
        kind = FaultKind(kind_name)
        fault = random_fault(
            kind,
            random.Random(f"eval-{kind_name}-{seed}"),
            ccfg.nodes,
            ccfg.gpus_per_node,
            tuple(cfg["fault_start_window_s"]),
        )
    sim = ClusterSim(ccfg, seed=seed, faults=[fault] if fault else [])
    events = [
        e
        for e in Detector(DetectorConfig.load()).run(sim.run(cfg["duration_s"]))
        if e.state == "firing"
    ]
    kinds = tuple(sorted({e.kind.value for e in events}))
    if fault is None:
        return RunResult("healthy", seed, "", 0.0, "", False, None, len(events), kinds)
    expected = EXPECTED[fault.kind]
    hits = [
        e
        for e in events
        if e.kind in expected
        and e.node == fault.node
        and e.ts >= fault.start_s
        and (fault.gpu is None or e.gpu == fault.gpu)
    ]
    off = [
        e
        for e in events
        if (e.node is not None and e.node != fault.node)
        or (e.node is None and fault.kind is not FaultKind.NETWORK)
    ]
    return RunResult(
        kind=fault.kind.value,
        seed=seed,
        fault=fault.describe(),
        magnitude=fault.magnitude,
        variant=fault.variant,
        detected=bool(hits),
        ttd_s=round(hits[0].ts - fault.start_s, 3) if hits else None,
        off_target=len(off),
        event_kinds=kinds,
    )


def run_eval(cfg: dict[str, Any], workers: int | None = None) -> list[RunResult]:
    """Run every configured simulation (in parallel; results are deterministic)."""
    base = int(cfg["seed_base"])
    jobs: list[tuple[str | None, int, dict[str, Any]]] = [
        (k.value, base + i, cfg) for k in FAULT_ORDER for i in range(cfg["runs_per_fault"])
    ]
    jobs += [(None, base + 10_000 + i, cfg) for i in range(cfg["healthy_runs"])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, jobs, chunksize=8))


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def summarize(results: Iterable[RunResult], duration_s: float) -> dict[str, Any]:
    """Aggregate per-run results into the metrics reported in docs/results.md."""
    rs = list(results)
    healthy = [r for r in rs if r.kind == "healthy"]
    out: dict[str, Any] = {"per_fault": {}, "buckets": {}, "healthy": {}}
    for kind in FAULT_ORDER:
        runs = [r for r in rs if r.kind == kind.value]
        det = [r for r in runs if r.detected]
        ttd = [r.ttd_s for r in det if r.ttd_s is not None]
        exp = {k.value for k in EXPECTED[kind]}
        fp = [r for r in healthy if exp & set(r.event_kinds)]
        out["per_fault"][kind.value] = {
            "runs": len(runs),
            "detected": len(det),
            "detection_rate": round(len(det) / len(runs), 4) if runs else 0.0,
            "healthy_fp_runs": len(fp),
            "healthy_fp_rate": round(len(fp) / len(healthy), 4) if healthy else 0.0,
            "ttd_median_s": round(statistics.median(ttd), 1) if ttd else None,
            "ttd_p90_s": round(_pct(ttd, 0.9), 1) if ttd else None,
            "ttd_max_s": round(max(ttd), 1) if ttd else None,
            "off_target_events": sum(r.off_target for r in runs),
        }
        rows = []
        for label, lo, hi in BUCKETS.get(kind, []):
            b = [r for r in runs if lo <= r.magnitude < hi]
            if b:
                rows.append(
                    {"bucket": label, "runs": len(b), "detected": sum(r.detected for r in b)}
                )
        if kind in (FaultKind.STRAGGLER, FaultKind.ECC):
            for variant in sorted({r.variant for r in runs}):
                b = [r for r in runs if r.variant == variant]
                rows.append(
                    {
                        "bucket": f"variant={variant}",
                        "runs": len(b),
                        "detected": sum(r.detected for r in b),
                    }
                )
        if rows:
            out["buckets"][kind.value] = rows
    any_fp = [r for r in healthy if r.event_kinds]
    # In healthy runs every firing event is false; RunResult.off_target holds the count.
    total_events = sum(r.off_target for r in healthy)
    hours = len(healthy) * duration_s / 3600
    out["healthy"] = {
        "runs": len(healthy),
        "runs_with_any_event": len(any_fp),
        "any_event_rate": round(len(any_fp) / len(healthy), 4) if healthy else 0.0,
        "false_events": total_events,
        "false_events_per_sim_hour": round(total_events / hours, 4) if hours else 0.0,
    }
    out["missed"] = [asdict(r) for r in rs if r.kind != "healthy" and not r.detected]
    out["healthy_fp_examples"] = [asdict(r) for r in any_fp]
    return out


def check_gate(summary: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Return a list of gate violations (empty = pass)."""
    failures = []
    for kind, floor in thresholds["min_detection_rate"].items():
        rate = summary["per_fault"][kind]["detection_rate"]
        if rate < floor:
            failures.append(f"{kind}: detection rate {rate:.3f} < floor {floor:.3f}")
    fp = summary["healthy"]["any_event_rate"]
    if fp > thresholds["max_healthy_fp_rate"]:
        failures.append(f"healthy FP rate {fp:.3f} > max {thresholds['max_healthy_fp_rate']:.3f}")
    return failures


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:g}"


def results_table(summary: dict[str, Any]) -> str:
    """The headline per-fault markdown table (also embedded in README.md)."""
    lines = [
        "| Fault type (simulated) | Runs | Detection rate | Healthy-run FP rate | "
        "TTD median (sim s) | TTD p90 (sim s) | TTD max (sim s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, s in summary["per_fault"].items():
        lines.append(
            f"| {kind} | {s['runs']} | {_fmt(s['detection_rate'], True)} "
            f"({s['detected']}/{s['runs']}) | {_fmt(s['healthy_fp_rate'], True)} "
            f"({s['healthy_fp_runs']}/{summary['healthy']['runs']}) | "
            f"{_fmt(s['ttd_median_s'])} | {_fmt(s['ttd_p90_s'])} | {_fmt(s['ttd_max_s'])} |"
        )
    h = summary["healthy"]
    lines.append("")
    lines.append(
        f"Healthy runs with >= 1 event of any kind (simulated): "
        f"{h['runs_with_any_event']}/{h['runs']} ({_fmt(h['any_event_rate'], True)}); "
        f"{h['false_events']} false events total = "
        f"{h['false_events_per_sim_hour']:g} per simulated hour."
    )
    return "\n".join(lines)


def render_markdown(
    summary: dict[str, Any], cfg: dict[str, Any], thresholds: dict[str, Any], failures: list[str]
) -> str:
    """Full docs/results.md content."""
    hours = cfg["healthy_runs"] * cfg["duration_s"] / 3600
    out = [
        "# GPUWatch evaluation results (SIMULATED)",
        "",
        "> **All numbers on this page come from the deterministic simulator in `gpu_sim/`,",
        "> not from real GPUs.** They measure the detector against the simulator's fault",
        "> model; they are not a claim about production accuracy. Regenerate with `make eval`.",
        "",
        "## Protocol",
        "",
        f"- Cluster: {cfg['nodes']} nodes x {cfg['gpus_per_node']} GPUs, 10 s scrape interval,"
        f" {cfg['duration_s']} simulated seconds per run.",
        f"- {cfg['runs_per_fault']} runs per fault type, one fault per run, seeds "
        f"{cfg['seed_base']}..{cfg['seed_base'] + cfg['runs_per_fault'] - 1}; fault target, "
        f"onset (uniform in {cfg['fault_start_window_s']} s) and severity drawn per seed.",
        f"- {cfg['healthy_runs']} healthy runs ({hours:.0f} simulated hours) with benign noise "
        "(jitter, GC spikes, multi-scrape slowdowns, checkpoint pauses, power capping, "
        "sporadic SBEs).",
        "- Detector thresholds (`detector/config.yaml`) were tuned on seeds >= 1000, disjoint "
        "from these.",
        "- Matching rules are documented in `eval/run.py`.",
        "",
        "## Results per fault type",
        "",
        results_table(summary),
        "",
        "## Detection by fault severity (simulated)",
        "",
        "Severity ranges intentionally extend below the detector's sensitivity floor "
        "(e.g. 5% step-time slowdown) so the misses are visible rather than hidden.",
        "",
    ]
    for kind, rows in summary["buckets"].items():
        out.append(f"**{kind}**")
        out.append("")
        out.append("| Bucket | Runs | Detected | Rate |")
        out.append("|---|---:|---:|---:|")
        for r in rows:
            out.append(
                f"| {r['bucket']} | {r['runs']} | {r['detected']} | "
                f"{r['detected'] / r['runs'] * 100:.1f}% |"
            )
        out.append("")
    out += [
        "## Off-target events in faulted runs (simulated)",
        "",
        "| Fault type | Off-target firing events |",
        "|---|---:|",
    ]
    for kind, s in summary["per_fault"].items():
        out.append(f"| {kind} | {s['off_target_events']} |")
    out += ["", "## Healthy-run false positives (simulated)", ""]
    if summary["healthy_fp_examples"]:
        out.append("| Seed | Event kinds |")
        out.append("|---:|---|")
        for r in summary["healthy_fp_examples"]:
            out.append(f"| {r['seed']} | {', '.join(r['event_kinds'])} |")
    else:
        out.append("None.")
    out += ["", "## Missed faults (simulated)", ""]
    if summary["missed"]:
        out.append("| Type | Seed | Fault |")
        out.append("|---|---:|---|")
        for r in summary["missed"]:
            out.append(f"| {r['kind']} | {r['seed']} | `{r['fault']}` |")
    else:
        out.append("None.")
    out += [
        "",
        "## CI gate",
        "",
        "Thresholds from `eval/thresholds.yaml`:",
        "",
        "| Fault type | Min detection rate | Measured |",
        "|---|---:|---:|",
    ]
    for kind, floor in thresholds["min_detection_rate"].items():
        out.append(
            f"| {kind} | {floor * 100:.0f}% | "
            f"{summary['per_fault'][kind]['detection_rate'] * 100:.1f}% |"
        )
    out.append("")
    out.append(
        f"Max healthy-run FP rate (any event): "
        f"{thresholds['max_healthy_fp_rate'] * 100:.0f}%; measured "
        f"{summary['healthy']['any_event_rate'] * 100:.1f}%."
    )
    out.append("")
    out.append(
        "Gate status: " + ("**PASS**" if not failures else "**FAIL** - " + "; ".join(failures))
    )
    out.append("")
    return "\n".join(out)


README_BEGIN = "<!-- EVAL:BEGIN (generated by `make eval`; do not edit) -->"
README_END = "<!-- EVAL:END -->"


def update_readme(readme: Path, summary: dict[str, Any]) -> bool:
    """Replace the generated block in README.md; returns True if markers exist."""
    if not readme.exists():
        return False
    text = readme.read_text()
    if README_BEGIN not in text or README_END not in text:
        return False
    head, _, rest = text.partition(README_BEGIN)
    _, _, tail = rest.partition(README_END)
    block = f"{README_BEGIN}\n{results_table(summary)}\n{README_END}"
    readme.write_text(head + block + tail)
    return True


def main_json(summary: dict[str, Any]) -> str:
    """Stable JSON dump of the summary."""
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"
