r"""CLI: diagnose the first alert of a simulated incident.

    python -m ai_diagnosis --fault thermal:node-03:2@300,mag=0.3 --seed 7            # replay
    python -m ai_diagnosis --fault thermal:node-03:2@300,mag=0.3 --seed 7 \
        --backend anthropic --record                                                 # live API

The replay backend needs no API key; it reads ``ai_diagnosis/fixtures``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

from ai_diagnosis.diagnose import (
    AnthropicBackend,
    Backend,
    Diagnoser,
    RecordingBackend,
    ReplayBackend,
)
from ai_diagnosis.incident import Incident, build_incident
from detector import Detector, Event, EventKind
from detector.sources import PrometheusSource
from gpu_sim import ClusterSim, Fault, LogLine, Snapshot

FIXTURES = Path(__file__).with_name("fixtures")


def incident_from_sim(
    faults: list[str],
    seed: int,
    kind: str | None = None,
    duration_s: float = 1200.0,
    window_s: float = 300.0,
) -> Incident | None:
    """Run sim + detector and build an incident for the first matching firing event."""
    sim = ClusterSim(seed=seed, faults=[Fault.parse(f) for f in faults])
    det = Detector()
    history: deque[Snapshot] = deque(maxlen=int(window_s / sim.config.tick_s) + 1)
    for snap in sim.run(duration_s):
        history.append(snap)
        for ev in det.observe(snap):
            if ev.state == "firing" and (kind is None or ev.kind is EventKind(kind)):
                return build_incident(ev, list(history), window_s)
    return None


def read_kernel_log(path: Path, node: str) -> list[LogLine]:
    """Parse ``journalctl -k -o short-unix`` output (``<unix_ts> <host> kernel: <msg>``)."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        ts, _, rest = line.partition(" ")
        try:
            t = float(ts)
        except ValueError:
            continue
        msg = rest.split("kernel: ", 1)[1] if "kernel: " in rest else rest
        out.append(LogLine(t, node, "kernel", msg.strip()))
    return out


def incident_from_live(
    event_file: Path,
    prom_url: str,
    kind: str | None = None,
    kernel_log: Path | None = None,
    window_s: float = 300.0,
    interval_s: float = 10.0,
    session: object = None,
) -> Incident | None:
    """Build an incident for a real detector event using Prometheus history.

    Snapshots for ``[event.ts - window_s, event.ts]`` are re-queried from
    Prometheus; kernel log lines (if given) inside the window are attached to
    the snapshot closest in time.
    """
    event = None
    for line in event_file.read_text().splitlines():
        if line.strip():
            ev = Event.from_dict(json.loads(line))
            if ev.state == "firing" and (kind is None or ev.kind is EventKind(kind)):
                event = ev
                break
    if event is None:
        return None
    src = PrometheusSource(
        prom_url, interval_s=interval_s, start=event.ts - window_s, end=event.ts, session=session
    )
    snaps = list(src)
    if kernel_log is not None and snaps and event.node:
        logs = [
            log
            for log in read_kernel_log(kernel_log, event.node)
            if event.ts - window_s <= log.ts <= event.ts
        ]
        last = snaps[-1]
        snaps[-1] = Snapshot(last.ts, last.gpus, last.nodes, tuple(logs))
    return build_incident(event, snaps, window_s)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(
        prog="ai_diagnosis",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fault", action="append", default=[])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kind", help="diagnose the first event of this kind")
    p.add_argument("--backend", choices=["replay", "anthropic"], default="replay")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--record", action="store_true", help="save anthropic responses as fixtures")
    p.add_argument("--fixtures", type=Path, default=FIXTURES)
    p.add_argument("--json", action="store_true")
    p.add_argument("--show-prompt", action="store_true")
    live = p.add_argument_group("live incident (instead of --fault)")
    live.add_argument(
        "--event", type=Path, help="detector JSONL output; first firing event is used"
    )
    live.add_argument("--prom-url", default="http://localhost:9090")
    live.add_argument("--kernel-log", type=Path, help="journalctl -k -o short-unix output")
    args = p.parse_args(argv)
    if args.event is not None:
        inc = incident_from_live(args.event, args.prom_url, args.kind, args.kernel_log)
    else:
        inc = incident_from_sim(args.fault, args.seed, args.kind)
    if inc is None:
        sys.stderr.write("no alert fired in the simulated window\n")
        return 1
    backend: Backend
    if args.backend == "replay":
        backend = ReplayBackend(args.fixtures)
    else:
        backend = AnthropicBackend(model=args.model)
        if args.record:
            backend = RecordingBackend(backend, args.fixtures)
    if args.show_prompt:
        sys.stdout.write(inc.render() + "\n\n")
    d = Diagnoser(backend).diagnose(inc)
    if isinstance(backend, ReplayBackend) and backend.stale:
        d.notes.append("replay fixture was produced for a different prompt (stale)")
    sys.stdout.write(json.dumps(d.to_dict(), indent=2) if args.json else d.render_markdown(inc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
