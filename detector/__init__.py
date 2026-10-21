"""Fault detector: stragglers, XID/ECC, thermal throttling, slow NCCL links."""

from detector.config import DetectorConfig
from detector.engine import Detector
from detector.events import Event, EventKind, Evidence, Severity

__all__ = ["Detector", "DetectorConfig", "Event", "EventKind", "Evidence", "Severity"]
