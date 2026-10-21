import json
from pathlib import Path

from detector.sources import DEFAULT_GPU_QUERIES, DEFAULT_NODE_QUERIES
from gpu_sim import metrics as m
from gpu_sim.model import Snapshot

ROOT = Path(__file__).resolve().parent.parent


def test_contract_file_matches_python_names():
    names = [x for x in (ROOT / "exporter/metrics_contract.txt").read_text().split() if x]
    assert names == list(m.ALL_METRICS)


def test_detector_queries_cover_every_contract_metric():
    queries = " ".join(list(DEFAULT_GPU_QUERIES.values()) + list(DEFAULT_NODE_QUERIES.values()))
    for name in m.ALL_METRICS:
        assert name in queries, name


def test_go_fixtures_are_valid_snapshots():
    for f in ("snapshot_healthy.json", "snapshot_xid79.json"):
        Snapshot.from_dict(json.loads((ROOT / "tests/fixtures" / f).read_text()))
