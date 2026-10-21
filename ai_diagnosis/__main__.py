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
from detector import Detector, EventKind
from gpu_sim import ClusterSim, Fault, Snapshot

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
    args = p.parse_args(argv)
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
