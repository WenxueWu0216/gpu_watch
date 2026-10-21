import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from detector import Detector
from gpu_sim import ClusterSim, Fault


def _metric_points(reader, name):
    data = reader.get_metrics_data()
    out = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == name:
                    out.extend(m.data.data_points)
    return out


def test_detector_emits_spans_and_metrics(otel):
    spans, reader = otel
    sim = ClusterSim(seed=1, faults=[Fault.parse("xid:node-01:0@20")])
    Detector().run(sim.run(60))
    observe = [s for s in spans.get_finished_spans() if s.name == "detector.observe"]
    assert len(observe) == 6
    with_events = [s for s in observe if s.attributes["gpuwatch.events"] > 0]
    assert with_events
    names = {e.name for s in with_events for e in s.events}
    assert "firing:xid_error" in names
    pts = _metric_points(reader, "gpuwatch.detector.events")
    kinds = {p.attributes["kind"] for p in pts}
    assert {"xid_error", "gpu_missing"} <= kinds
    assert _metric_points(reader, "gpuwatch.detector.observe.duration")


class _Collector(BaseHTTPRequestHandler):
    paths: ClassVar[list[str]] = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Collector.paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, *a):
        pass


def test_setup_telemetry_exports_otlp_http():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    code = (
        "from tracing.otel import setup_telemetry\n"
        f"assert setup_telemetry('gpuwatch-test', {url!r}, export_interval_ms=200)\n"
        "from detector import Detector\n"
        "from gpu_sim import ClusterSim\n"
        "Detector().run(ClusterSim(seed=0).run(30))\n"
    )
    try:
        subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
    finally:
        srv.shutdown()
    assert "/v1/traces" in _Collector.paths
    assert "/v1/metrics" in _Collector.paths


def test_setup_telemetry_noop_without_endpoint(monkeypatch):
    from tracing import otel

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr(otel, "_configured", False)
    assert otel.setup_telemetry("x") is False
