"""CLI: ``python -m detector --source {sim,jsonl,prometheus}``; prints JSONL events."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from detector.config import DetectorConfig
from detector.engine import Detector
from detector.sources import JsonlSource, PrometheusSource, SimSource, SnapshotSource
from gpu_sim import ClusterConfig, ClusterSim, Fault
from tracing.otel import setup_telemetry


def build_source(args: argparse.Namespace) -> SnapshotSource:
    """Construct the snapshot source selected on the command line."""
    if args.source == "sim":
        sim = ClusterSim(
            ClusterConfig(nodes=args.nodes, gpus_per_node=args.gpus),
            seed=args.seed,
            faults=[Fault.parse(f) for f in args.fault],
        )
        return SimSource(sim, args.duration)
    if args.source == "jsonl":
        return JsonlSource(Path(args.input))
    return PrometheusSource(
        url=args.prom_url,
        interval_s=args.interval,
        start=args.start,
        end=args.end,
        max_polls=args.max_polls,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(prog="detector", description=__doc__)
    p.add_argument("--source", choices=["sim", "jsonl", "prometheus"], default="sim")
    p.add_argument("--config", help="detector YAML (default detector/config.yaml)")
    p.add_argument("--all-states", action="store_true", help="also print resolved events")
    p.add_argument(
        "--otlp-endpoint", help="OTLP/HTTP collector URL (default $OTEL_EXPORTER_OTLP_ENDPOINT)"
    )
    sim = p.add_argument_group("sim")
    sim.add_argument("--nodes", type=int, default=8)
    sim.add_argument("--gpus", type=int, default=8)
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--duration", type=float, default=1800.0)
    sim.add_argument("--fault", action="append", default=[])
    p.add_argument("--input", help="JSONL snapshots (source=jsonl)")
    prom = p.add_argument_group("prometheus")
    prom.add_argument("--prom-url", default="http://localhost:9090")
    prom.add_argument("--interval", type=float, default=10.0)
    prom.add_argument("--start", type=float, help="backfill start (unix s)")
    prom.add_argument("--end", type=float, help="backfill end (unix s)")
    prom.add_argument("--max-polls", type=int)
    args = p.parse_args(argv)

    setup_telemetry("gpuwatch-detector", args.otlp_endpoint)
    detector = Detector(DetectorConfig.load(args.config))
    for snap in build_source(args):
        for ev in detector.observe(snap):
            if ev.state == "firing" or args.all_states:
                sys.stdout.write(ev.to_json() + "\n")
                sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
