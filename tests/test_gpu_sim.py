import json
import random
import statistics
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from gpu_sim import ClusterConfig, ClusterSim, Fault, FaultKind, Snapshot, random_fault
from gpu_sim.model import THERMAL_THROTTLE_MASK, THROTTLE_SW_POWER_CAP
from gpu_sim.server import SimRunner, make_handler


def _gpu(snap, node, gpu):
    return next(g for g in snap.gpus if g.node == node and g.gpu == gpu)


def _node(snap, node):
    return next(n for n in snap.nodes if n.node == node)


def test_same_seed_is_reproducible():
    faults = [Fault.parse("thermal:node-02:1@100")]
    a = [s.to_dict() for s in ClusterSim(seed=7, faults=faults).run(400)]
    b = [s.to_dict() for s in ClusterSim(seed=7, faults=faults).run(400)]
    assert a == b


def test_different_seeds_differ():
    a = list(ClusterSim(seed=1).run(100))[-1]
    b = list(ClusterSim(seed=2).run(100))[-1]
    assert a.to_dict() != b.to_dict()


def test_shape_and_ticks():
    cfg = ClusterConfig(nodes=3, gpus_per_node=4, tick_s=5)
    snaps = list(ClusterSim(cfg, seed=0).run(50))
    assert len(snaps) == 10
    assert snaps[-1].ts == 50
    assert len(snaps[0].gpus) == 12
    assert snaps[0].node_names() == ["node-00", "node-01", "node-02"]


def test_snapshot_roundtrip():
    snap = list(ClusterSim(seed=3).run(60))[-1]
    assert Snapshot.from_dict(json.loads(json.dumps(snap.to_dict()))) == snap


def test_healthy_run_has_no_thermal_throttle_but_has_power_cap():
    snaps = list(ClusterSim(seed=11).run(1800))
    assert not any(g.throttle_reasons & THERMAL_THROTTLE_MASK for s in snaps for g in s.gpus)
    # Benign power capping is present as a confounder for the thermal detector.
    assert any(g.throttle_reasons & THROTTLE_SW_POWER_CAP for s in snaps for g in s.gpus)
    assert max(g.temp_c for s in snaps for g in s.gpus) < 80


def test_thermal_fault_heats_and_throttles():
    f = Fault(FaultKind.THERMAL, "node-01", 100.0, gpu=3, magnitude=0.3)
    snaps = list(ClusterSim(seed=1, faults=[f]).run(900))
    before, after = snaps[8], snaps[-1]
    assert _gpu(before, "node-01", 3).throttle_reasons & THERMAL_THROTTLE_MASK == 0
    hot = _gpu(after, "node-01", 3)
    assert hot.temp_c > 85
    assert hot.throttle_reasons & THERMAL_THROTTLE_MASK
    assert hot.sm_clock_mhz < 0.85 * _gpu(after, "node-01", 2).sm_clock_mhz
    # A throttled GPU makes its node slower than its peers.
    med = statistics.median(n.step_time_s for n in after.nodes)
    assert _node(after, "node-01").step_time_s > 1.1 * med


def test_xid79_takes_gpu_off_bus_and_hangs_job():
    f = Fault.parse("xid:node-04:6@200")
    snaps = list(ClusterSim(seed=1, faults=[f]).run(400))
    at = next(s for s in snaps if s.ts >= 200)
    g = _gpu(at, "node-04", 6)
    assert g.xid == 79 and not g.available
    assert any("Xid" in log.message and ": 79," in log.message for log in at.logs)
    assert all(n.step_time_s is None for n in snaps[-1].nodes)


def test_ecc_dbe_burst_increments_counters():
    f = Fault(FaultKind.ECC, "node-00", 100.0, gpu=2, magnitude=20.0, variant="dbe")
    snaps = list(ClusterSim(seed=5, faults=[f]).run(400))
    g = _gpu(snaps[-1], "node-00", 2)
    assert g.ecc_dbe_total >= 1
    assert g.xid == 48
    assert g.ecc_sbe_total > 50
    # Burst is bounded: counters stop growing after the burst window.
    assert (
        _gpu(snaps[-1], "node-00", 2).ecc_sbe_total == _gpu(snaps[-5], "node-00", 2).ecc_sbe_total
    )


def test_ecc_sbe_variant_has_no_dbe():
    f = Fault(FaultKind.ECC, "node-00", 100.0, gpu=2, magnitude=20.0, variant="sbe")
    g = _gpu(list(ClusterSim(seed=5, faults=[f]).run(400))[-1], "node-00", 2)
    assert g.ecc_dbe_total == 0 and g.xid == 0 and g.ecc_sbe_total > 50


@pytest.mark.parametrize("variant", ["host", "gpu"])
def test_straggler_slows_only_target_node(variant):
    f = Fault(FaultKind.STRAGGLER, "node-06", 0.0, magnitude=1.3, variant=variant)
    snap = list(ClusterSim(seed=2, faults=[f]).run(300))[-1]
    others = [n.step_time_s for n in snap.nodes if n.node != "node-06"]
    assert _node(snap, "node-06").step_time_s > 1.2 * statistics.median(others)


def test_straggler_gpu_variant_raises_relative_util():
    f = Fault(FaultKind.STRAGGLER, "node-06", 0.0, magnitude=1.3, variant="gpu")
    snap = list(ClusterSim(seed=2, faults=[f]).run(300))[-1]
    slow = statistics.mean(g.util_pct for g in snap.gpus if g.node == "node-06")
    rest = statistics.mean(g.util_pct for g in snap.gpus if g.node != "node-06")
    assert slow > rest + 10


def test_network_fault_raises_xmit_wait_and_drops_busbw():
    f = Fault(FaultKind.NETWORK, "node-03", 100.0, magnitude=0.5)
    snaps = list(ClusterSim(seed=4, faults=[f]).run(600))
    pre, post = snaps[5], snaps[-1]
    others = [n.net_xmit_wait for n in post.nodes if n.node != "node-03"]
    assert _node(post, "node-03").net_xmit_wait > 2.5 * statistics.median(others)
    assert post.nodes[0].nccl_busbw_gb_s < 0.7 * pre.nodes[0].nccl_busbw_gb_s
    # Per-node time-to-barrier is unaffected by the slow link.
    med = statistics.median(n.step_time_s for n in post.nodes)
    assert _node(post, "node-03").step_time_s < 1.06 * med


def test_fault_parse():
    f = Fault.parse("ecc:node-01:3@120,mag=5,variant=sbe")
    assert (f.kind, f.node, f.gpu, f.start_s, f.magnitude, f.variant) == (
        FaultKind.ECC,
        "node-01",
        3,
        120.0,
        5.0,
        "sbe",
    )
    assert Fault.parse("straggler:node-02@0").gpu is None
    with pytest.raises(ValueError):
        Fault.parse("thermal-node")
    with pytest.raises(ValueError):
        Fault.parse("thermal:node-01:0@1,bogus=1")


def test_random_fault_is_seeded_and_in_range():
    for kind in FaultKind:
        a = random_fault(kind, random.Random(3), 8, 8, (300, 900))
        b = random_fault(kind, random.Random(3), 8, 8, (300, 900))
        assert a == b
        assert 300 <= a.start_s <= 900
        assert a.node.startswith("node-")


def test_inject_unknown_node_rejected():
    with pytest.raises(ValueError):
        ClusterSim(seed=0).inject(Fault.parse("thermal:node-99:0@0"))


def test_http_server_snapshot_and_inject():
    runner = SimRunner(ClusterSim(ClusterConfig(nodes=2), seed=0), speed=1.0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runner))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        snap = json.load(urllib.request.urlopen(base + "/snapshot"))
        assert len(snap["gpus"]) == 16
        req = urllib.request.Request(
            base + "/inject",
            method="POST",
            data=json.dumps({"spec": "thermal:node-01:0@+10"}).encode(),
        )
        assert "thermal" in json.load(urllib.request.urlopen(req))["injected"]
        assert runner.sim.faults[0].start_s == pytest.approx(runner.sim.now + 10, abs=1)
        bad = urllib.request.Request(base + "/inject", method="POST", data=b'{"spec": "x"}')
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
    finally:
        httpd.shutdown()
