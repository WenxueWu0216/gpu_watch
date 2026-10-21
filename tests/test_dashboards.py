import importlib.util
import json
import re
from pathlib import Path

import yaml

from gpu_sim.metrics import ALL_METRICS

ROOT = Path(__file__).resolve().parent.parent
GRAFANA = ROOT / "dashboards" / "grafana"
RULES = ROOT / "dashboards" / "alerts" / "gpuwatch.rules.yaml"

# Metrics produced by components other than the exporter.
EXTRA = {
    "gpuwatch_exporter_up",
    "gpuwatch_detector_active_alerts",
    "gpuwatch_detector_events_total",
    "gpuwatch_detector_observe_duration_milliseconds_bucket",
    "ALERTS",
}


def _gen():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "dashboards" / "generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recorded_names():
    doc = yaml.safe_load(RULES.read_text())
    return {r["record"] for g in doc["groups"] for r in g["rules"] if "record" in r}


def exprs(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "expr":
                yield v
            else:
                yield from exprs(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from exprs(v)


IDENT = re.compile(r"\b([A-Za-z_:][A-Za-z0-9_:]*)\s*(?:\{|\[|\)|$|\s|>|<|=|/|\*)")
PROMQL_WORDS = {
    "avg",
    "sum",
    "max",
    "min",
    "count",
    "by",
    "quantile",
    "rate",
    "increase",
    "or",
    "and",
    "unless",
    "histogram_quantile",
    "le",
    "vector",
    "scalar",
    "bool",
    "offset",
    "avg_over_time",
    "floor",
    "label_values",
    "Hostname",
    "kind",
    "severity",
    "alertname",
    "on",
    "without",
}


def metric_names(expr: str) -> set[str]:
    expr = re.sub(r"\{[^}]*\}", "{}", expr)  # drop label matchers
    return {m for m in IDENT.findall(expr) if m not in PROMQL_WORDS and not m.isdigit()}


def test_committed_dashboards_match_generator():
    rendered = _gen().render()
    assert set(rendered) == {p.name for p in GRAFANA.glob("*.json")}
    for name, text in rendered.items():
        assert (GRAFANA / name).read_text() == text, f"{name} is stale: run dashboards/generate.py"


def test_dashboards_reference_only_known_metrics():
    known = set(ALL_METRICS) | EXTRA | recorded_names()
    for path in GRAFANA.glob("*.json"):
        dash = json.loads(path.read_text())
        assert dash["uid"].startswith("gpuwatch-")
        ids = [p["id"] for p in dash["panels"]]
        assert len(ids) == len(set(ids))
        for e in exprs(dash):
            unknown = metric_names(e) - known
            assert not unknown, f"{path.name}: {e} uses {unknown}"


def test_alert_rules_reference_known_metrics_and_runbook_anchors():
    known = set(ALL_METRICS) | EXTRA | recorded_names()
    doc = yaml.safe_load(RULES.read_text())
    runbook = (ROOT / "runbook.md").read_text()
    anchors = {
        "#" + re.sub(r"[^a-z0-9 -]", "", h.lower()).strip().replace(" ", "-")
        for h in re.findall(r"^##+ (.+)$", runbook, re.M)
    }
    for g in doc["groups"]:
        for r in g["rules"]:
            unknown = metric_names(r["expr"]) - known
            assert not unknown, f"{r.get('alert') or r.get('record')}: {unknown}"
            rb = r.get("labels", {}).get("runbook")
            if rb:
                assert "#" + rb.split("#")[1] in anchors, rb
