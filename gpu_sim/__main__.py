"""CLI: ``python -m gpu_sim {run,serve}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from gpu_sim.cluster import ClusterConfig, ClusterSim
from gpu_sim.faults import Fault


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(prog="gpu_sim", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "serve"):
        s = sub.add_parser(name)
        s.add_argument("--nodes", type=int, default=8)
        s.add_argument("--gpus", type=int, default=8)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument(
            "--fault",
            action="append",
            default=[],
            help="kind:node[:gpu]@start[,mag=X][,variant=Y] (repeatable)",
        )
    sub.choices["run"].add_argument("--duration", type=float, default=600.0)
    sub.choices["serve"].add_argument("--host", default="0.0.0.0")
    sub.choices["serve"].add_argument("--port", type=int, default=9500)
    sub.choices["serve"].add_argument(
        "--speed", type=float, default=1.0, help="simulated seconds per wall second"
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    sim = ClusterSim(
        ClusterConfig(nodes=args.nodes, gpus_per_node=args.gpus),
        seed=args.seed,
        faults=[Fault.parse(f) for f in args.fault],
    )
    if args.cmd == "run":
        for snap in sim.run(args.duration):
            sys.stdout.write(json.dumps(snap.to_dict()) + "\n")
        return 0
    from gpu_sim.server import serve

    serve(sim, args.host, args.port, args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
