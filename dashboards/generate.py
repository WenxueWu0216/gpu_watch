"""Generate the Grafana dashboards in ``dashboards/grafana/`` (``python dashboards/generate.py``).

Dashboards are generated rather than hand-edited so panel layout and PromQL
stay consistent and reviewable; ``tests/test_dashboards.py`` checks that the
committed JSON matches this generator and only references known metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).parent / "grafana"
PROM = {"type": "prometheus", "uid": "prometheus"}
TEMPO = {"type": "tempo", "uid": "tempo"}


class Layout:
    """Places panels on Grafana's 24-column grid, left to right."""

    def __init__(self) -> None:
        """Start at the top-left."""
        self.x = 0
        self.y = 0
        self.row_h = 0
        self.next_id = 1

    def pos(self, w: int, h: int) -> dict[str, int]:
        """Allocate a w x h cell."""
        if self.x + w > 24:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0
        p = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        return p

    def panel(
        self, kind: str, title: str, targets: list[dict[str, Any]], w: int, h: int, **extra: Any
    ) -> dict[str, Any]:
        """Build one panel."""
        p: dict[str, Any] = {
            "id": self.next_id,
            "type": kind,
            "title": title,
            "datasource": PROM,
            "gridPos": self.pos(w, h),
            "targets": targets,
        }
        self.next_id += 1
        p.update(extra)
        return p


def q(expr: str, legend: str = "", ref: str = "A", **kw: Any) -> dict[str, Any]:
    """A Prometheus target."""
    t = {"datasource": PROM, "expr": expr, "legendFormat": legend, "refId": ref}
    t.update(kw)
    return t


def ts_opts(unit: str = "short", **defaults: Any) -> dict[str, Any]:
    """Timeseries field config."""
    return {
        "fieldConfig": {"defaults": {"unit": unit, **defaults}, "overrides": []},
        "options": {
            "legend": {"displayMode": "table", "placement": "right"},
            "tooltip": {"mode": "multi"},
        },
    }


def dashboard(
    uid: str,
    title: str,
    panels: list[dict[str, Any]],
    templating: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Wrap panels into a dashboard."""
    return {
        "uid": uid,
        "title": title,
        "tags": ["gpuwatch"],
        "timezone": "utc",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"},
        "templating": {"list": templating or []},
        "annotations": {"list": []},
        "panels": panels,
    }


def cluster_overview() -> dict[str, Any]:
    """Cluster-wide health at a glance."""
    lay = Layout()
    stat = {"options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background"}}
    panels = [
        lay.panel("stat", "GPUs reporting", [q("count(DCGM_FI_DEV_GPU_UTIL)")], 4, 4, **stat),
        lay.panel(
            "stat",
            "Mean GPU utilization",
            [q("avg(DCGM_FI_DEV_GPU_UTIL)")],
            4,
            4,
            fieldConfig={"defaults": {"unit": "percent"}, "overrides": []},
            **stat,
        ),
        lay.panel(
            "stat",
            "Max GPU temperature",
            [q("max(DCGM_FI_DEV_GPU_TEMP)")],
            4,
            4,
            fieldConfig={
                "defaults": {
                    "unit": "celsius",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "orange", "value": 85},
                            {"color": "red", "value": 92},
                        ],
                    },
                },
                "overrides": [],
            },
            **stat,
        ),
        lay.panel(
            "stat",
            "Thermally throttled GPUs",
            [q("sum(gpuwatch:gpu_thermal_throttled)")],
            4,
            4,
            **stat,
        ),
        lay.panel(
            "stat",
            "Active detector alerts",
            [q("sum(gpuwatch_detector_active_alerts) or vector(0)")],
            4,
            4,
            **stat,
        ),
        lay.panel(
            "stat",
            "All-reduce busbw (GB/s)",
            [q("avg(gpuwatch_nccl_busbw_gb_per_second)")],
            4,
            4,
            **stat,
        ),
        lay.panel(
            "timeseries",
            "GPU utilization by node (mean)",
            [q("avg by (Hostname) (DCGM_FI_DEV_GPU_UTIL)", "{{Hostname}}")],
            12,
            8,
            **ts_opts("percent"),
        ),
        lay.panel(
            "timeseries",
            "Step time by node (time-to-barrier)",
            [
                q("gpuwatch:node_step_time_seconds:max", "{{Hostname}}"),
                q("quantile(0.5, gpuwatch:node_step_time_seconds:max)", "cluster median", "B"),
            ],
            12,
            8,
            **ts_opts("s"),
        ),
        lay.panel(
            "timeseries",
            "Step time ratio to cluster median",
            [q("gpuwatch:node_step_time:ratio_to_cluster_median", "{{Hostname}}")],
            12,
            8,
            **ts_opts("percentunit"),
        ),
        lay.panel(
            "timeseries",
            "NIC transmit-wait ratio to cluster median",
            [q("gpuwatch:node_xmit_wait:ratio_to_cluster_median", "{{Hostname}}")],
            12,
            8,
            **ts_opts("short"),
        ),
        lay.panel(
            "timeseries",
            "NCCL all-reduce bus bandwidth",
            [q("gpuwatch_nccl_busbw_gb_per_second", "{{Hostname}}")],
            12,
            8,
            **ts_opts("GBs"),
        ),
        lay.panel(
            "timeseries",
            "Exporter sources up",
            [q("gpuwatch_exporter_up", "{{instance}} {{source}}")],
            12,
            8,
            **ts_opts("short", min=0, max=1),
        ),
    ]
    return dashboard("gpuwatch-cluster", "GPUWatch / Cluster overview", panels)


def per_node() -> dict[str, Any]:
    """Per-GPU detail for one node (``$node`` variable)."""
    lay = Layout()
    sel = 'Hostname="$node"'
    templ = [
        {
            "name": "node",
            "type": "query",
            "datasource": PROM,
            "refresh": 2,
            "query": {"query": "label_values(DCGM_FI_DEV_GPU_UTIL, Hostname)", "refId": "nodes"},
            "definition": "label_values(DCGM_FI_DEV_GPU_UTIL, Hostname)",
            "current": {},
            "includeAll": False,
            "multi": False,
        }
    ]
    panels = [
        lay.panel(
            "timeseries",
            "GPU utilization",
            [q(f"DCGM_FI_DEV_GPU_UTIL{{{sel}}}", "gpu {{gpu}}")],
            12,
            8,
            **ts_opts("percent"),
        ),
        lay.panel(
            "timeseries",
            "GPU temperature",
            [q(f"DCGM_FI_DEV_GPU_TEMP{{{sel}}}", "gpu {{gpu}}")],
            12,
            8,
            **ts_opts("celsius"),
        ),
        lay.panel(
            "timeseries",
            "SM clock",
            [q(f"DCGM_FI_DEV_SM_CLOCK{{{sel}}}", "gpu {{gpu}}")],
            12,
            8,
            **ts_opts("rotmhz"),
        ),
        lay.panel(
            "timeseries",
            "Power draw",
            [q(f"DCGM_FI_DEV_POWER_USAGE{{{sel}}}", "gpu {{gpu}}")],
            12,
            8,
            **ts_opts("watt"),
        ),
        lay.panel(
            "timeseries",
            "Thermal throttled (SW/HW thermal bit)",
            [q(f"gpuwatch:gpu_thermal_throttled{{{sel}}}", "gpu {{gpu}}")],
            12,
            8,
            **ts_opts("short", min=0, max=1),
        ),
        lay.panel(
            "timeseries",
            "Framebuffer used",
            [q(f"DCGM_FI_DEV_FB_USED{{{sel}}}", "gpu {{gpu}}")],
            12,
            8,
            **ts_opts("decmbytes"),
        ),
        lay.panel(
            "timeseries",
            "ECC errors (increase over 5m)",
            [
                q(f"increase(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL{{{sel}}}[5m])", "SBE gpu {{gpu}}"),
                q(f"increase(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{{{sel}}}[5m])", "DBE gpu {{gpu}}", "B"),
            ],
            12,
            8,
            **ts_opts("short"),
        ),
        lay.panel(
            "timeseries",
            "Last XID",
            [q(f"DCGM_FI_DEV_XID_ERRORS{{{sel}}}", "gpu {{gpu}}")],
            12,
            8,
            **ts_opts("short"),
        ),
        lay.panel(
            "timeseries",
            "NVLink bandwidth",
            [q(f"DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL{{{sel}}}", "gpu {{gpu}}")],
            12,
            8,
            **ts_opts("short"),
        ),
        lay.panel(
            "timeseries",
            "Node step time vs cluster median",
            [
                q(f"gpuwatch:node_step_time_seconds:max{{{sel}}}", "$node"),
                q("quantile(0.5, gpuwatch:node_step_time_seconds:max)", "cluster median", "B"),
            ],
            12,
            8,
            **ts_opts("s"),
        ),
    ]
    return dashboard("gpuwatch-node", "GPUWatch / Node GPUs", panels, templ)


def fault_timeline() -> dict[str, Any]:
    """When did what go wrong: detector alerts, hardware signals, traces."""
    lay = Layout()
    panels = [
        lay.panel(
            "state-timeline",
            "Detector active alerts by kind",
            [q("sum by (kind) (gpuwatch_detector_active_alerts)", "{{kind}}")],
            24,
            7,
            fieldConfig={
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}],
                    },
                },
                "overrides": [],
            },
            options={"showValue": "never", "mergeValues": True},
        ),
        lay.panel(
            "timeseries",
            "Detector events (rate, by kind and severity)",
            [
                q(
                    "sum by (kind, severity) (rate(gpuwatch_detector_events_total"
                    '{state="firing"}[5m]))',
                    "{{kind}} {{severity}}",
                )
            ],
            12,
            8,
            **ts_opts("short"),
        ),
        lay.panel(
            "timeseries",
            "Prometheus-rule alerts firing",
            [q('count by (alertname) (ALERTS{alertstate="firing"})', "{{alertname}}")],
            12,
            8,
            **ts_opts("short"),
        ),
        lay.panel(
            "state-timeline",
            "XID by GPU",
            [q("DCGM_FI_DEV_XID_ERRORS > 0", "{{Hostname}} gpu {{gpu}}")],
            12,
            8,
            options={"showValue": "always", "mergeValues": True},
        ),
        lay.panel(
            "state-timeline",
            "Thermal throttling by GPU",
            [q("gpuwatch:gpu_thermal_throttled == 1", "{{Hostname}} gpu {{gpu}}")],
            12,
            8,
            options={"showValue": "never", "mergeValues": True},
        ),
        lay.panel(
            "timeseries",
            "Detector evaluation latency (p95)",
            [
                q(
                    "histogram_quantile(0.95, sum by (le) "
                    "(rate(gpuwatch_detector_observe_duration_milliseconds_bucket[5m])))",
                    "p95",
                )
            ],
            12,
            8,
            **ts_opts("ms"),
        ),
        lay.panel(
            "table",
            "Recent detector / bring-up traces (Tempo)",
            [
                {
                    "datasource": TEMPO,
                    "queryType": "traceql",
                    "refId": "A",
                    "limit": 20,
                    "query": '{resource.service.namespace="gpuwatch" && span.gpuwatch.events > 0}'
                    ' || {name="bringup.gate"}',
                }
            ],
            12,
            8,
            datasource=TEMPO,
        ),
    ]
    return dashboard("gpuwatch-faults", "GPUWatch / Fault timeline", panels)


DASHBOARDS = {
    "cluster-overview.json": cluster_overview,
    "node-gpu.json": per_node,
    "fault-timeline.json": fault_timeline,
}


def render() -> dict[str, str]:
    """Filename -> JSON text."""
    return {
        name: json.dumps(fn(), indent=2, sort_keys=True) + "\n" for name, fn in DASHBOARDS.items()
    }


def main() -> None:
    """Write all dashboards."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in render().items():
        (OUT / name).write_text(text)


if __name__ == "__main__":
    main()
