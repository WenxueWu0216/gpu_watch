"""Real-time HTTP front-end for the simulator.

Endpoints:
    GET  /snapshot  latest :class:`~gpu_sim.model.Snapshot` as JSON
    GET  /healthz   liveness
    POST /inject    body ``{"spec": "thermal:node-03:2@+30,mag=0.2"}``; a start of
                    ``+N`` means N simulated seconds from now.

The Go exporter (``exporter/``, ``--mode=sim``) polls ``/snapshot`` and
re-publishes it in Prometheus exposition format.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gpu_sim.cluster import ClusterSim
from gpu_sim.faults import Fault
from gpu_sim.model import Snapshot

log = logging.getLogger("gpu_sim.server")


class SimRunner:
    """Advances a :class:`ClusterSim` in (scaled) real time on a thread."""

    def __init__(self, sim: ClusterSim, speed: float = 1.0) -> None:
        """``speed`` = simulated seconds per wall-clock second."""
        self.sim = sim
        self.speed = speed
        self._lock = threading.Lock()
        self._latest: Snapshot = sim.step()
        self._stop = threading.Event()

    def latest(self) -> Snapshot:
        """Most recent snapshot."""
        with self._lock:
            return self._latest

    def inject(self, spec: str) -> Fault:
        """Parse and inject a fault; ``@+N`` is relative to the current sim time."""
        relative = "@+" in spec
        fault = Fault.parse(spec.replace("@+", "@"))
        with self._lock:
            if relative:
                fault = replace(fault, start_s=self.sim.now + fault.start_s)
            self.sim.inject(fault)
        log.info("injected %s", fault.describe())
        return fault

    def loop(self) -> None:
        """Step forever until :meth:`stop`."""
        interval = self.sim.config.tick_s / self.speed
        while not self._stop.wait(interval):
            with self._lock:
                self._latest = self.sim.step()

    def stop(self) -> None:
        """Stop the stepping loop."""
        self._stop.set()


def make_handler(runner: SimRunner) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to ``runner``."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: object) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/snapshot":
                self._send(200, runner.latest().to_dict())
            elif self.path == "/healthz":
                self._send(200, {"ok": True, "sim_time_s": runner.latest().ts})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/inject":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                spec = json.loads(self.rfile.read(length) or b"{}")["spec"]
                fault = runner.inject(str(spec))
            except (KeyError, ValueError) as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(200, {"injected": fault.describe()})

        def log_message(self, format: str, *args: object) -> None:
            log.debug(format, *args)

    return Handler


def serve(sim: ClusterSim, host: str, port: int, speed: float) -> None:
    """Run the HTTP server until interrupted."""
    runner = SimRunner(sim, speed)
    threading.Thread(target=runner.loop, daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), make_handler(runner))
    log.info("gpu_sim serving on %s:%d (speed x%.1f)", host, port, speed)
    try:
        httpd.serve_forever()
    finally:
        runner.stop()
        time.sleep(0)
