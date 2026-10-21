import io
import json
from contextlib import redirect_stdout

import pytest

from detector import Detector, EventKind
from detector.__main__ import main as detector_main
from detector.sources import JsonlSource, PrometheusSource, SimSource
from gpu_sim import ClusterSim, Fault


class FakeResp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class FakePromSession:
    """Answers /api/v1/query with canned vectors keyed by PromQL text."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def get(self, url, params, timeout):
        assert url.endswith("/api/v1/query")
        self.calls.append(params)
        res = [
            {"metric": lbl, "value": [float(params["time"]), str(v)]}
            for lbl, v in self.vectors.get(params["query"], [])
        ]
        return FakeResp({"status": "success", "data": {"resultType": "vector", "result": res}})


def gpu_lbl(node, gpu):
    return {"Hostname": node, "gpu": str(gpu), "UUID": f"GPU-{node}-{gpu}"}


def vectors():
    v = {
        "DCGM_FI_DEV_GPU_UTIL": [
            (gpu_lbl("a", 0), 91),
            (gpu_lbl("a", 1), 92),
            (gpu_lbl("b", 0), 90),
        ],
        "DCGM_FI_DEV_FB_USED": [(gpu_lbl("a", 0), 70000), (gpu_lbl("a", 1), 70000)],
        "DCGM_FI_DEV_FB_FREE": [(gpu_lbl("a", 0), 11559), (gpu_lbl("a", 1), 11559)],
        "DCGM_FI_DEV_GPU_TEMP": [(gpu_lbl("a", 0), 66)],
        "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": [(gpu_lbl("a", 1), 32)],
        "DCGM_FI_DEV_XID_ERRORS": [(gpu_lbl("c", 3), 79)],
        "max by (Hostname) (gpuwatch_train_step_time_seconds)": [
            ({"Hostname": "a"}, 0.51),
            ({"Hostname": "b"}, 0.49),
        ],
        "sum by (Hostname) (gpuwatch_net_xmit_wait_rate)": [({"Hostname": "a"}, 2000)],
        "min by (Hostname) (gpuwatch_nccl_busbw_gb_per_second)": [({"Hostname": "a"}, 44.5)],
        # Series without the required labels are ignored.
        "DCGM_FI_DEV_POWER_USAGE": [({"instance": "x"}, 1.0)],
    }
    return v


def test_prometheus_snapshot_assembly():
    src = PrometheusSource("http://prom:9090/", session=FakePromSession(vectors()))
    s = src.snapshot_at(1000.0)
    by = {(g.node, g.gpu): g for g in s.gpus}
    assert by[("a", 0)].util_pct == 91 and by[("a", 0)].mem_total_mib == 81559
    assert by[("a", 1)].throttle_reasons == 32
    assert by[("c", 3)].xid == 79 and not by[("c", 3)].available
    nodes = {n.node: n for n in s.nodes}
    assert nodes["a"].step_time_s == 0.51 and nodes["a"].nccl_busbw_gb_s == 44.5
    assert nodes["b"].nccl_busbw_gb_s is None


def test_prometheus_xid_flows_into_detector():
    src = PrometheusSource("http://prom", session=FakePromSession(vectors()))
    ev = Detector().observe(src.snapshot_at(1.0))
    assert {e.kind for e in ev} == {EventKind.XID, EventKind.GPU_MISSING}


def test_prometheus_backfill_and_follow_modes():
    sess = FakePromSession(vectors())
    src = PrometheusSource("http://p", interval_s=15, start=100, end=160, session=sess)
    assert [s.ts for s in src] == [100, 115, 130, 145, 160]
    sleeps = []
    clock = iter(range(1000, 2000, 1))
    src = PrometheusSource(
        "http://p",
        interval_s=10,
        max_polls=3,
        session=sess,
        sleep=sleeps.append,
        clock=lambda: float(next(clock)),
    )
    assert len(list(src)) == 3
    assert sleeps == [9.0, 9.0, 9.0]


def test_prometheus_error_status_raises():
    class Bad(FakePromSession):
        def get(self, url, params, timeout):
            return FakeResp({"status": "error", "error": "boom"})

    with pytest.raises(RuntimeError):
        PrometheusSource("http://p", session=Bad({})).snapshot_at(0)


def test_jsonl_source_roundtrip(tmp_path):
    snaps = list(ClusterSim(seed=1).run(100))
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(s.to_dict()) for s in snaps) + "\n")
    assert list(JsonlSource(path)) == snaps


@pytest.mark.parametrize(
    ("spec", "kind"),
    [
        ("thermal:node-03:2@300,mag=0.2", EventKind.THERMAL),
        ("xid:node-02:5@300", EventKind.XID),
        ("ecc:node-04:3@300,mag=10,variant=sbe", EventKind.ECC),
        ("straggler:node-05@300,mag=1.15", EventKind.STRAGGLER),
        ("network:node-01@300,mag=0.6", EventKind.SLOW_LINK),
    ],
)
def test_sim_fault_detected_on_right_target(spec, kind):
    f = Fault.parse(spec)
    ev = [
        e
        for e in Detector().run(SimSource(ClusterSim(seed=42, faults=[f]), 1200))
        if e.state == "firing" and e.kind is kind
    ]
    assert ev, f"{kind} not detected"
    assert ev[0].node == f.node and ev[0].ts >= f.start_s
    if f.gpu is not None:
        assert ev[0].gpu == f.gpu


def test_sim_healthy_has_no_hardware_events_and_at_most_warnings():
    # Benign multi-scrape slowdowns can (rarely) produce a warning-level
    # straggler false positive; ``make eval`` measures that rate. Hardware,
    # thermal and network rules must stay silent on healthy runs.
    for seed in (42, 43, 44):
        events = Detector().run(SimSource(ClusterSim(seed=seed), 1800))
        assert all(e.kind is EventKind.STRAGGLER for e in events)
        assert all(e.severity.value in ("warning", "info") for e in events)


def test_cli_sim_outputs_jsonl():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = detector_main(
            ["--source", "sim", "--seed", "1", "--duration", "400", "--fault", "xid:node-01:0@100"]
        )
    assert rc == 0
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert {x["kind"] for x in lines} == {"xid_error", "gpu_missing"}
    assert all(x["severity"] == "critical" for x in lines)
