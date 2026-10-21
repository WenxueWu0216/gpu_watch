"""Stateful fault detector: snapshots in, structured events out."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from detector.config import DetectorConfig
from detector.events import Event, EventKind, Evidence, Severity
from detector.stats import PeerStats, median, robust_z
from gpu_sim import metrics as m
from gpu_sim.model import (
    THERMAL_THROTTLE_MASK,
    THROTTLE_GPU_IDLE,
    THROTTLE_HW_THERMAL,
    THROTTLE_SW_POWER_CAP,
    THROTTLE_SW_THERMAL,
    Snapshot,
)

AlertKey = tuple[EventKind, str | None, int | None, str]


@dataclass(frozen=True)
class _Candidate:
    severity: Severity
    summary: str
    evidence: tuple[Evidence, ...]
    persistence: int


@dataclass
class _AlertState:
    pending: int = 0
    clean: int = 0
    firing: bool = False
    severity: Severity = Severity.INFO
    fired_at: float | None = None


def _decode_throttle(bits: int) -> str:
    names = {
        THROTTLE_GPU_IDLE: "GPU_IDLE",
        THROTTLE_SW_POWER_CAP: "SW_POWER_CAP",
        THROTTLE_SW_THERMAL: "SW_THERMAL",
        THROTTLE_HW_THERMAL: "HW_THERMAL",
    }
    return "|".join(n for b, n in names.items() if bits & b) or "NONE"


class Detector:
    """Detects stragglers, XID/ECC hardware faults, thermal throttling and slow links.

    Feed one :class:`~gpu_sim.model.Snapshot` per scrape interval to
    :meth:`observe`; it returns events whose state changed in that interval
    (newly firing, escalated, or resolved).
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        """Create a detector with ``config`` (defaults from ``config.yaml``)."""
        self.cfg = config or DetectorConfig.load()
        w = self.cfg.window
        self._step: dict[str, deque[float]] = {}
        self._util: dict[str, deque[float]] = {}
        self._wait: dict[str, deque[float]] = {}
        self._sbe: dict[tuple[str, int], deque[int]] = {}
        self._dbe_base: dict[tuple[str, int], int] = {}
        self._seen_gpus: set[tuple[str, int]] = set()
        self._busbw_win: deque[float] = deque(maxlen=w)
        self._busbw_hist: deque[float] = deque(maxlen=self.cfg.busbw_baseline_scrapes)
        self._alerts: dict[AlertKey, _AlertState] = {}

    # ----------------------------------------------------------------- public
    def observe(self, snap: Snapshot) -> list[Event]:
        """Ingest one snapshot and return state-change events."""
        cands: dict[AlertKey, _Candidate] = {}
        self._update_windows(snap)
        self._check_hardware(snap, cands)
        self._check_thermal(snap, cands)
        self._check_straggler(snap, cands)
        self._check_network(snap, cands)
        self._check_busbw(snap, cands)
        return self._advance(snap.ts, cands)

    def run(self, snapshots: Iterable[Snapshot]) -> list[Event]:
        """Convenience: observe every snapshot and return all events."""
        out: list[Event] = []
        for s in snapshots:
            out.extend(self.observe(s))
        return out

    def active(self) -> list[tuple[AlertKey, Severity]]:
        """Currently firing alerts."""
        return [(k, s.severity) for k, s in self._alerts.items() if s.firing]

    # ---------------------------------------------------------------- windows
    def _update_windows(self, snap: Snapshot) -> None:
        w = self.cfg.window
        sw = self.cfg.step_window
        util_by_node: dict[str, list[float]] = {}
        for g in snap.gpus:
            if g.available:
                util_by_node.setdefault(g.node, []).append(g.util_pct)
        for n in snap.nodes:
            if n.step_time_s is None:
                # A gap in step data invalidates the window (job hung/restarted).
                self._step.pop(n.node, None)
                self._util.pop(n.node, None)
                # NIC back-pressure is only comparable across nodes under load.
                self._wait.pop(n.node, None)
                continue
            self._step.setdefault(n.node, deque(maxlen=sw)).append(n.step_time_s)
            if n.node in util_by_node:
                u = sum(util_by_node[n.node]) / len(util_by_node[n.node])
                self._util.setdefault(n.node, deque(maxlen=sw)).append(u)
            self._wait.setdefault(n.node, deque(maxlen=w)).append(n.net_xmit_wait)

    @staticmethod
    def _full(d: dict[str, deque[float]]) -> dict[str, float]:
        """Window medians for keys whose window is full."""
        return {k: median(list(v)) for k, v in d.items() if len(v) == v.maxlen}

    # --------------------------------------------------------------- hardware
    def _check_hardware(self, snap: Snapshot, cands: dict[AlertKey, _Candidate]) -> None:
        cfg = self.cfg
        present: set[tuple[str, int]] = set()
        kernel_logs: dict[str, list[Evidence]] = {}
        for log in snap.logs:
            if log.source == "kernel" and "Xid" in log.message:
                kernel_logs.setdefault(log.node, []).append(
                    Evidence(f"log:{log.source}", log.ts, None, {"node": log.node}, log.message)
                )
        for g in snap.gpus:
            key = (g.node, g.gpu)
            present.add(key)
            self._seen_gpus.add(key)
            labels = {"node": g.node, "gpu": str(g.gpu)}
            if g.xid:
                sev = Severity.CRITICAL if g.xid in cfg.xid_critical else Severity.WARNING
                ev = (Evidence(m.XID_ERRORS, g.xid, None, labels, f"last XID = {g.xid}"),)
                cands[(EventKind.XID, g.node, g.gpu, str(g.xid))] = _Candidate(
                    sev,
                    f"XID {g.xid} on {g.node} gpu {g.gpu}",
                    ev + tuple(kernel_logs.get(g.node, [])),
                    1,
                )
            if not g.available:
                cands[(EventKind.GPU_MISSING, g.node, g.gpu, "")] = _Candidate(
                    Severity.CRITICAL,
                    f"{g.node} gpu {g.gpu} stopped reporting telemetry",
                    (Evidence(m.GPU_UTIL, 0.0, None, labels, "GPU not readable by DCGM"),),
                    1,
                )
            # ECC: DBE is latched until the volatile counter resets; SBE is windowed.
            base = self._dbe_base.setdefault(key, g.ecc_dbe_total)
            if g.ecc_dbe_total < base:
                self._dbe_base[key] = base = g.ecc_dbe_total
            sbe = self._sbe.setdefault(key, deque(maxlen=cfg.window + 1))
            if sbe and g.ecc_sbe_total < sbe[-1]:
                sbe.clear()
            sbe.append(g.ecc_sbe_total)
            sbe_delta = sbe[-1] - sbe[0]
            ecc_ev: list[Evidence] = []
            sev = Severity.INFO
            if g.ecc_dbe_total > base:
                sev = Severity.CRITICAL
                ecc_ev.append(
                    Evidence(
                        m.ECC_DBE_TOTAL,
                        g.ecc_dbe_total - base,
                        0,
                        labels,
                        "uncorrectable (double-bit) ECC errors since baseline",
                    )
                )
            if sbe_delta >= cfg.sbe_window_delta:
                sev = sev if sev is Severity.CRITICAL else Severity.WARNING
                ecc_ev.append(
                    Evidence(
                        m.ECC_SBE_TOTAL,
                        sbe_delta,
                        cfg.sbe_window_delta,
                        labels,
                        f"correctable ECC errors in last {len(sbe) - 1} scrapes",
                    )
                )
            if ecc_ev:
                cands[(EventKind.ECC, g.node, g.gpu, "")] = _Candidate(
                    sev,
                    f"ECC errors on {g.node} gpu {g.gpu}",
                    tuple(ecc_ev) + tuple(kernel_logs.get(g.node, [])),
                    1,
                )
        for node, gpu in self._seen_gpus - present:
            cands[(EventKind.GPU_MISSING, node, gpu, "")] = _Candidate(
                Severity.CRITICAL,
                f"{node} gpu {gpu} missing from scrape",
                (
                    Evidence(
                        m.GPU_UTIL, 0.0, None, {"node": node, "gpu": str(gpu)}, "series absent"
                    ),
                ),
                cfg.persistence,
            )

    # ---------------------------------------------------------------- thermal
    def _check_thermal(self, snap: Snapshot, cands: dict[AlertKey, _Candidate]) -> None:
        cfg = self.cfg
        clocks: dict[str, list[float]] = {}
        for g in snap.gpus:
            if g.available and not g.throttle_reasons & THROTTLE_GPU_IDLE:
                clocks.setdefault(g.node, []).append(g.sm_clock_mhz)
        for g in snap.gpus:
            if not g.available:
                continue
            labels = {"node": g.node, "gpu": str(g.gpu)}
            peers = clocks.get(g.node, [])
            peer_clock = median(peers) if peers else g.sm_clock_mhz
            bits_hit = bool(g.throttle_reasons & THERMAL_THROTTLE_MASK)
            fallback = (
                g.temp_c >= cfg.thermal_temp_c
                and g.sm_clock_mhz < (1 - cfg.thermal_clock_drop) * peer_clock
                and not g.throttle_reasons & (THROTTLE_SW_POWER_CAP | THROTTLE_GPU_IDLE)
            )
            if not (bits_hit or fallback):
                continue
            critical = (
                bool(g.throttle_reasons & THROTTLE_HW_THERMAL)
                or g.temp_c >= cfg.thermal_critical_temp_c
            )
            ev = (
                Evidence(m.GPU_TEMP, g.temp_c, cfg.thermal_temp_c, labels, "GPU temperature (C)"),
                Evidence(
                    m.SM_CLOCK,
                    g.sm_clock_mhz,
                    round(peer_clock, 1),
                    labels,
                    "SM clock vs. median of same-node peers",
                ),
                Evidence(
                    m.THROTTLE_REASONS,
                    g.throttle_reasons,
                    None,
                    labels,
                    _decode_throttle(g.throttle_reasons),
                ),
            )
            cands[(EventKind.THERMAL, g.node, g.gpu, "")] = _Candidate(
                Severity.CRITICAL if critical else Severity.WARNING,
                f"thermal throttling on {g.node} gpu {g.gpu} "
                f"({g.temp_c:.0f}C, clock {g.sm_clock_mhz:.0f} vs {peer_clock:.0f} MHz)",
                ev,
                cfg.thermal_persistence,
            )

    # -------------------------------------------------------------- straggler
    def _check_straggler(self, snap: Snapshot, cands: dict[AlertKey, _Candidate]) -> None:
        cfg = self.cfg
        steps = self._full(self._step)
        utils = self._full(self._util)
        if len(steps) < 2:
            return
        if len(steps) == 2:
            (a, va), (b, vb) = sorted(steps.items(), key=lambda kv: kv[1])
            if vb > va * (1 + cfg.pairwise_min_rel):
                rel = vb / va - 1
                cands[(EventKind.STRAGGLER, b, None, "")] = _Candidate(
                    Severity.WARNING,
                    f"{b} step time {rel:.0%} slower than {a} (2-node pairwise mode)",
                    (
                        Evidence(
                            m.STEP_TIME,
                            vb,
                            va * (1 + cfg.pairwise_min_rel),
                            {"node": b},
                            f"windowed median vs. peer {a} = {va:.4f}s",
                        ),
                    ),
                    cfg.persistence,
                )
            return
        zs = robust_z(steps, cfg.step_mad_floor)
        uz = robust_z(utils, cfg.util_mad_floor)
        for node, st in zs.items():
            if st.z <= cfg.z_threshold or st.rel <= cfg.step_min_rel:
                continue
            ev = [
                Evidence(
                    m.STEP_TIME,
                    round(st.value, 5),
                    round(st.median, 5),
                    {"node": node},
                    f"windowed median step time; robust z={st.z:.1f}, "
                    f"+{st.rel:.1%} vs cluster median",
                )
            ]
            pattern = "host/input-bound (GPU busy time unchanged)"
            u: PeerStats | None = uz.get(node)
            if u is not None:
                ev.append(
                    Evidence(
                        m.GPU_UTIL,
                        round(u.value, 2),
                        round(u.median, 2),
                        {"node": node},
                        f"node mean GPU util vs cluster median; robust z={u.z:.1f}",
                    )
                )
                if u.z > cfg.z_threshold:
                    pattern = "GPU-bound (slow GPU work: clocks, HBM, PCIe)"
            ev.append(Evidence("derived:straggler_pattern", 0.0, None, {"node": node}, pattern))
            for (kind, n, gpu, _), state in self._alerts.items():
                if state.firing and n == node and kind in (EventKind.THERMAL, EventKind.ECC):
                    ev.append(
                        Evidence(
                            f"correlated:{kind.value}",
                            1.0,
                            None,
                            {"node": node, "gpu": str(gpu)},
                            f"{kind.value} firing on gpu {gpu}; likely cause",
                        )
                    )
            sev = Severity.CRITICAL if st.rel >= cfg.step_critical_rel else Severity.WARNING
            cands[(EventKind.STRAGGLER, node, None, "")] = _Candidate(
                sev,
                f"{node} is a straggler: step time +{st.rel:.1%} vs cluster median "
                f"(z={st.z:.1f}), {pattern}",
                tuple(ev),
                cfg.persistence,
            )

    # ---------------------------------------------------------------- network
    def _check_network(self, snap: Snapshot, cands: dict[AlertKey, _Candidate]) -> None:
        cfg = self.cfg
        waits = self._full(self._wait)
        busbw = [n.nccl_busbw_gb_s for n in snap.nodes if n.nccl_busbw_gb_s is not None]
        bus_ev = (
            (
                Evidence(
                    m.NCCL_BUSBW,
                    round(median(busbw), 3),
                    None,
                    {},
                    "cluster all-reduce bus bandwidth (GB/s), current scrape",
                ),
            )
            if busbw
            else ()
        )
        flagged: dict[str, tuple[float, float, str]] = {}
        if len(waits) == 2:
            (a, va), (b, vb) = sorted(waits.items(), key=lambda kv: kv[1])
            if vb > va * (1 + 2 * cfg.net_min_rel):
                flagged[b] = (vb, va, f"vs peer {a} (2-node pairwise mode)")
        else:
            for node, st in robust_z(waits, cfg.net_mad_floor).items():
                if st.z > cfg.z_threshold and st.rel > cfg.net_min_rel:
                    flagged[node] = (st.value, st.median, f"robust z={st.z:.1f}")
        for node, (val, ref, note) in flagged.items():
            rel = val / ref - 1 if ref else 0.0
            sev = Severity.CRITICAL if rel >= cfg.net_critical_rel else Severity.WARNING
            cands[(EventKind.SLOW_LINK, node, None, "")] = _Candidate(
                sev,
                f"{node} NIC transmit-wait {rel:+.0%} vs cluster median: slow/congested link",
                (
                    Evidence(
                        m.NET_XMIT_WAIT,
                        round(val, 2),
                        round(ref, 2),
                        {"node": node},
                        f"windowed median port_xmit_wait rate {note}",
                    ),
                    *bus_ev,
                ),
                cfg.persistence,
            )

    def _check_busbw(self, snap: Snapshot, cands: dict[AlertKey, _Candidate]) -> None:
        cfg = self.cfg
        vals = [n.nccl_busbw_gb_s for n in snap.nodes if n.nccl_busbw_gb_s is not None]
        if not vals:
            return
        self._busbw_win.append(median(vals))
        hist = list(self._busbw_hist)
        anomalous = False
        if len(self._busbw_win) == cfg.window and len(hist) >= 2 * cfg.window:
            cur, base = median(list(self._busbw_win)), median(hist)
            if cur < (1 - cfg.busbw_drop) * base:
                anomalous = True
                cands[(EventKind.BUSBW_DEGRADED, None, None, "")] = _Candidate(
                    Severity.WARNING,
                    f"cluster all-reduce busbw {cur:.1f} GB/s is {1 - cur / base:.0%} "
                    f"below baseline {base:.1f} GB/s",
                    (
                        Evidence(
                            m.NCCL_BUSBW,
                            round(cur, 3),
                            round((1 - cfg.busbw_drop) * base, 3),
                            {},
                            "windowed median vs. pre-incident baseline",
                        ),
                    ),
                    cfg.persistence,
                )
        if not anomalous:
            # Baseline is frozen while degraded so a long incident does not
            # become the new normal.
            self._busbw_hist.append(self._busbw_win[-1])

    # ---------------------------------------------------------- state machine
    def _advance(self, ts: float, cands: dict[AlertKey, _Candidate]) -> list[Event]:
        out: list[Event] = []
        for key, c in cands.items():
            st = self._alerts.setdefault(key, _AlertState())
            st.pending += 1
            st.clean = 0
            escalated = st.firing and c.severity.rank > st.severity.rank
            if (not st.firing and st.pending >= c.persistence) or escalated:
                st.firing = True
                st.severity = c.severity
                st.fired_at = ts if st.fired_at is None else st.fired_at
                out.append(Event(ts, key[0], c.severity, key[1], key[2], c.summary, c.evidence))
        for key, st in self._alerts.items():
            if key in cands:
                continue
            st.pending = 0
            if st.firing:
                st.clean += 1
                if st.clean >= self.cfg.resolve_after:
                    st.firing = False
                    st.fired_at = None
                    st.severity = Severity.INFO
                    out.append(
                        Event(
                            ts,
                            key[0],
                            Severity.INFO,
                            key[1],
                            key[2],
                            f"resolved: {key[0].value}",
                            (),
                            state="resolved",
                        )
                    )
        return out
