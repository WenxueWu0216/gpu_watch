import json
from dataclasses import fields

import pytest
import yaml

from detector import Detector, DetectorConfig, Event, EventKind, Severity
from detector.config import DEFAULT_CONFIG_PATH
from detector.stats import mad, median, robust_z
from gpu_sim.model import (
    THROTTLE_HW_THERMAL,
    THROTTLE_SW_POWER_CAP,
    THROTTLE_SW_THERMAL,
    LogLine,
)
from tests.helpers import NODES, snap

CFG = DetectorConfig()  # dataclass defaults (asserted equal to config.yaml below)


def firing(events, kind=None):
    return [e for e in events if e.state == "firing" and (kind is None or e.kind is kind)]


def feed(det, snaps):
    out = []
    for s in snaps:
        out.extend((s.ts, e) for e in det.observe(s))
    return out


# ------------------------------------------------------------------ stats/config
def test_median_and_mad():
    assert median([1, 2, 3, 100]) == 2.5
    assert mad([1, 2, 3, 4, 100]) == 1


def test_robust_z_basic_and_floor():
    vals = {f"n{i}": 1.0 for i in range(7)} | {"slow": 1.2}
    z = robust_z(vals, mad_floor_frac=0.01)
    # MAD is 0 so the floor (0.01 * median) is used: 0.6745 * 0.2 / 0.01 = 13.49
    assert z["slow"].z == pytest.approx(13.49, rel=1e-3)
    assert z["n0"].z == 0
    assert z["slow"].rel == pytest.approx(0.2)


def test_robust_z_needs_three_points():
    assert robust_z({"a": 1.0, "b": 2.0}, 0.01) == {}


def test_yaml_defaults_match_dataclass():
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    assert {f.name for f in fields(DetectorConfig)} == set(data)
    assert DetectorConfig.load() == DetectorConfig()


def test_config_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown"):
        DetectorConfig.load(bogus=1)


def test_event_json_roundtrip():
    det = Detector(CFG)
    ev = firing(det.observe(snap(10, gpu_over={("node-01", 0): {"xid": 79}})))[0]
    again = Event.from_dict(json.loads(ev.to_json()))
    assert again == ev


# --------------------------------------------------------------------- straggler
def test_straggler_fires_after_window_and_persistence():
    det = Detector(CFG)
    events = feed(det, [snap(t * 10, step={"node-03": 0.6}) for t in range(1, 30)])
    s = [(ts, e) for ts, e in events if e.kind is EventKind.STRAGGLER]
    assert len(s) == 1
    ts, ev = s[0]
    # window fills on scrape 12, then persistence=3 consecutive evaluations
    assert ts == 10 * (CFG.step_window + CFG.persistence - 1)
    assert ev.node == "node-03" and ev.severity is Severity.WARNING
    assert ev.evidence[0].metric == "gpuwatch_train_step_time_seconds"
    assert "z=" in ev.evidence[0].note


def test_straggler_critical_when_much_slower():
    det = Detector(CFG)
    ev = firing(det.run([snap(t, step={"node-03": 0.8}) for t in range(20)]), EventKind.STRAGGLER)
    assert ev[0].severity is Severity.CRITICAL


def test_small_deviation_is_not_a_straggler():
    det = Detector(CFG)
    # 3% slower: below step_min_rel (5%) even though z exceeds the threshold
    assert not firing(det.run([snap(t, step={"node-03": 0.515}) for t in range(40)]))


def test_short_spike_absorbed_by_window_median():
    det = Detector(CFG)
    snaps = [snap(t, step={"node-02": 0.9 if 15 <= t < 20 else 0.5}) for t in range(60)]
    assert not firing(det.run(snaps))


def test_straggler_resolves_after_recovery():
    det = Detector(CFG)
    snaps = [snap(t, step={"node-03": 0.6 if t < 30 else 0.5}) for t in range(80)]
    events = det.run(snaps)
    assert [e.state for e in events if e.kind is EventKind.STRAGGLER] == ["firing", "resolved"]


def test_step_gap_resets_window():
    det = Detector(CFG)
    snaps = [
        snap(
            t,
            step={"node-03": None if t == 8 else 0.6}
            | {n: (None if t == 8 else 0.5) for n in NODES if n != "node-03"},
        )
        for t in range(1, 30)
    ]
    ev = [e for e in det.run(snaps) if e.kind is EventKind.STRAGGLER]
    # gap at scrape 8 => window refills from scrape 9: fires at 9 + 12 + 2 - 1 = 22
    assert ev[0].ts == 22


def test_straggler_gpu_bound_pattern_and_correlation():
    det = Detector(CFG)
    over = {("node-03", g): {"util_pct": 100.0} for g in range(2)}
    over[("node-03", 1)] |= {"throttle_reasons": THROTTLE_SW_THERMAL, "temp_c": 88.0}
    ev = firing(
        det.run([snap(t, step={"node-03": 0.6}, gpu_over=over) for t in range(20)]),
        EventKind.STRAGGLER,
    )[0]
    notes = " ".join(e.note for e in ev.evidence)
    assert "GPU-bound" in ev.summary
    assert "thermal_throttle firing on gpu 1" in notes


def test_two_node_pairwise_mode():
    det = Detector(CFG)
    nodes = ["a", "b"]
    ev = firing(
        det.run([snap(t, nodes=nodes, step={"b": 0.6}) for t in range(20)]), EventKind.STRAGGLER
    )
    assert len(ev) == 1 and ev[0].node == "b" and "pairwise" in ev[0].summary
    det = Detector(CFG)
    assert not firing(det.run([snap(t, nodes=nodes, step={"b": 0.53}) for t in range(20)]))


# ---------------------------------------------------------------------- hardware
@pytest.mark.parametrize(
    ("code", "sev"), [(79, Severity.CRITICAL), (48, Severity.CRITICAL), (13, Severity.WARNING)]
)
def test_xid_severity_and_immediate(code, sev):
    det = Detector(CFG)
    log = LogLine(5, "node-01", "kernel", f"NVRM: Xid (PCI:0000:18:00): {code}, ...")
    ev = firing(det.observe(snap(10, gpu_over={("node-01", 1): {"xid": code}}, logs=(log,))))
    assert len(ev) == 1
    assert (ev[0].kind, ev[0].severity, ev[0].node, ev[0].gpu) == (EventKind.XID, sev, "node-01", 1)
    assert any(e.metric == "log:kernel" for e in ev[0].evidence)
    # Latched: no duplicate on the next scrape.
    assert not firing(det.observe(snap(20, gpu_over={("node-01", 1): {"xid": code}})))


def test_new_xid_code_is_new_event():
    det = Detector(CFG)
    det.observe(snap(10, gpu_over={("node-01", 1): {"xid": 13}}))
    ev = firing(det.observe(snap(20, gpu_over={("node-01", 1): {"xid": 79}})))
    assert [e.severity for e in ev] == [Severity.CRITICAL]


def test_gpu_unavailable_and_missing():
    det = Detector(CFG)
    ev = firing(
        det.observe(snap(10, gpu_over={("node-02", 0): {"available": False}})),
        EventKind.GPU_MISSING,
    )
    assert ev and ev[0].severity is Severity.CRITICAL
    det = Detector(CFG)
    det.observe(snap(10))
    events = det.run([snap(t, drop_gpus={("node-05", 1)}) for t in range(20, 60, 10)])
    ev = firing(events, EventKind.GPU_MISSING)
    assert len(ev) == 1 and ev[0].node == "node-05" and ev[0].ts == 40  # persistence 3


def test_ecc_dbe_latched_critical():
    det = Detector(CFG)
    det.observe(snap(10, gpu_over={("node-04", 1): {"ecc_dbe_total": 2}}))  # baseline
    ev = firing(det.observe(snap(20, gpu_over={("node-04", 1): {"ecc_dbe_total": 3}})))
    assert ev[0].kind is EventKind.ECC and ev[0].severity is Severity.CRITICAL
    later = det.run(
        [snap(t, gpu_over={("node-04", 1): {"ecc_dbe_total": 3}}) for t in range(30, 300, 10)]
    )
    assert not [e for e in later if e.kind is EventKind.ECC]  # stays firing, no resolve


def test_ecc_sbe_window_threshold_and_reset():
    det = Detector(CFG)
    # 1 SBE per scrape: 6 in a window < threshold 10
    assert not det.run(
        [snap(t, gpu_over={("node-04", 1): {"ecc_sbe_total": t}}) for t in range(10)]
    )
    det = Detector(CFG)
    ev = firing(
        det.run([snap(t, gpu_over={("node-04", 1): {"ecc_sbe_total": 5 * t}}) for t in range(10)]),
        EventKind.ECC,
    )
    assert ev and ev[0].severity is Severity.WARNING
    # Counter reset (driver reload) must not produce a negative delta / event.
    det = Detector(CFG)
    totals = [100, 100, 0, 0, 1]
    assert not det.run(
        [snap(t, gpu_over={("node-04", 1): {"ecc_sbe_total": v}}) for t, v in enumerate(totals)]
    )


def test_ecc_escalates_from_sbe_to_dbe():
    det = Detector(CFG)
    snaps = [
        snap(
            t,
            gpu_over={
                ("node-04", 1): {"ecc_sbe_total": 5 * t, "ecc_dbe_total": 1 if t >= 6 else 0}
            },
        )
        for t in range(10)
    ]
    ev = firing(det.run(snaps), EventKind.ECC)
    assert [e.severity for e in ev] == [Severity.WARNING, Severity.CRITICAL]


# ----------------------------------------------------------------------- thermal
def test_thermal_bits_fire_after_persistence():
    det = Detector(CFG)
    over = {
        ("node-06", 1): {
            "throttle_reasons": THROTTLE_SW_THERMAL,
            "temp_c": 88.0,
            "sm_clock_mhz": 1600.0,
        }
    }
    events = feed(det, [snap(t, gpu_over=over) for t in range(1, 6)])
    th = [(ts, e) for ts, e in events if e.kind is EventKind.THERMAL]
    assert th[0][0] == CFG.thermal_persistence
    assert th[0][1].severity is Severity.WARNING
    assert "SW_THERMAL" in " ".join(e.note for e in th[0][1].evidence)


def test_thermal_hw_bit_is_critical():
    det = Detector(CFG)
    over = {("node-06", 1): {"throttle_reasons": THROTTLE_HW_THERMAL, "temp_c": 90.0}}
    ev = firing(det.run([snap(t, gpu_over=over) for t in range(5)]), EventKind.THERMAL)
    assert ev[0].severity is Severity.CRITICAL


def test_power_cap_is_not_thermal():
    det = Detector(CFG)
    over = {
        ("node-06", 1): {
            "throttle_reasons": THROTTLE_SW_POWER_CAP,
            "sm_clock_mhz": 1900.0,
            "temp_c": 88.0,
        }
    }
    assert not det.run([snap(t, gpu_over=over) for t in range(10)])


def test_thermal_fallback_without_bits():
    det = Detector(CFG)
    over = {("node-06", 1): {"temp_c": 89.0, "sm_clock_mhz": 1500.0}}
    ev = firing(
        det.run([snap(t, gpus_per_node=4, gpu_over=over) for t in range(5)]), EventKind.THERMAL
    )
    assert ev and ev[0].gpu == 1
    # Hot but not slower than peers: not throttling.
    det = Detector(CFG)
    over = {("node-06", 1): {"temp_c": 89.0}}
    assert not det.run([snap(t, gpus_per_node=4, gpu_over=over) for t in range(5)])


# ----------------------------------------------------------------------- network
def test_slow_link_fires_on_outlier_node():
    det = Detector(CFG)
    ev = firing(
        det.run([snap(t, wait={"node-02": 5000.0}) for t in range(12)]), EventKind.SLOW_LINK
    )
    assert len(ev) == 1 and ev[0].node == "node-02"
    assert ev[0].ts == CFG.window + CFG.persistence - 2  # ts starts at 0


def test_slow_link_ignores_moderate_variation():
    det = Detector(CFG)
    assert not det.run([snap(t, wait={"node-02": 2400.0}) for t in range(20)])


def test_slow_link_two_node_pairwise():
    det = Detector(CFG)
    ev = firing(det.run([snap(t, nodes=["a", "b"], wait={"a": 9000.0}) for t in range(12)]))
    assert ev and ev[0].node == "a" and ev[0].kind is EventKind.SLOW_LINK


def test_busbw_degraded_against_frozen_baseline():
    det = Detector(CFG)
    snaps = [snap(t, busbw=45.0 if t < 20 else 20.0) for t in range(200)]
    ev = [e for e in det.run(snaps) if e.kind is EventKind.BUSBW_DEGRADED]
    # Baseline is frozen while degraded, so the alert stays firing (no resolve).
    assert [e.state for e in ev] == ["firing"]
    assert ev[0].node is None


def test_healthy_synthetic_cluster_is_quiet():
    det = Detector(CFG)
    assert det.run([snap(t) for t in range(100)]) == []
    assert det.active() == []
