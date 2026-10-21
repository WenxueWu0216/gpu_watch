"""``python -m eval``: run the evaluation, write docs/results.{md,json}, enforce the gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from eval.run import (
    HERE,
    check_gate,
    load_config,
    main_json,
    render_markdown,
    run_eval,
    summarize,
    update_readme,
)

ROOT = HERE.parent


def main(argv: list[str] | None = None) -> int:
    """Entry point; exit code 1 if the CI gate fails."""
    p = argparse.ArgumentParser(prog="eval", description=__doc__)
    p.add_argument("--config", type=Path, default=HERE / "config.yaml")
    p.add_argument("--thresholds", type=Path, default=HERE / "thresholds.yaml")
    p.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    p.add_argument("--readme", type=Path, default=ROOT / "README.md")
    p.add_argument("--runs", type=int, help="override runs_per_fault (quick checks)")
    p.add_argument("--healthy", type=int, help="override healthy_runs")
    p.add_argument("--workers", type=int)
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.runs is not None:
        cfg["runs_per_fault"] = args.runs
    if args.healthy is not None:
        cfg["healthy_runs"] = args.healthy
    thresholds = yaml.safe_load(args.thresholds.read_text())
    summary = summarize(run_eval(cfg, args.workers), cfg["duration_s"])
    failures = check_gate(summary, thresholds)
    md = render_markdown(summary, cfg, thresholds, failures)
    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "results.md").write_text(md)
        (args.out_dir / "results.json").write_text(main_json(summary))
        update_readme(args.readme, summary)
    sys.stdout.write(md)
    for f in failures:
        sys.stderr.write(f"EVAL GATE FAILED: {f}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
