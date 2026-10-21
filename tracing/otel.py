"""OpenTelemetry setup shared by the detector and the bring-up pipeline.

Instrumented code only depends on ``opentelemetry-api`` (no-op unless a
provider is installed). :func:`setup_telemetry` installs SDK providers that
export traces and metrics over OTLP/HTTP to a collector, e.g. the
``otel-collector`` service in ``docker-compose.yml``.
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import Any

from opentelemetry import metrics, trace

log = logging.getLogger("gpuwatch.tracing")
_configured = False


def setup_telemetry(
    service_name: str,
    endpoint: str | None = None,
    export_interval_ms: int = 10_000,
) -> bool:
    """Install OTLP/HTTP trace + metric exporters.

    Args:
        service_name: ``service.name`` resource attribute.
        endpoint: collector base URL (e.g. ``http://otel-collector:4318``);
            defaults to ``$OTEL_EXPORTER_OTLP_ENDPOINT``. If unset, telemetry
            stays a no-op.
        export_interval_ms: metric export period.

    Returns:
        True if exporters were installed.
    """
    global _configured
    endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint or _configured:
        return _configured
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # pragma: no cover - depends on optional extras
        log.warning("opentelemetry-sdk / OTLP exporter not installed; telemetry disabled")
        return False
    base = endpoint.rstrip("/")
    resource = Resource.create({"service.name": service_name, "service.namespace": "gpuwatch"})
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces")))
    trace.set_tracer_provider(tp)
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{base}/v1/metrics"),
        export_interval_millis=export_interval_ms,
    )
    mp = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(mp)
    atexit.register(tp.shutdown)
    atexit.register(mp.shutdown)
    _configured = True
    log.info("OTLP telemetry -> %s (service %s)", base, service_name)
    return True


def setup_in_memory(service_name: str = "gpuwatch-test") -> tuple[Any, Any]:
    """Install in-memory exporters (tests). Returns ``(span_exporter, metric_reader)``.

    OpenTelemetry providers can be set only once per process, so call this once
    (e.g. from a session-scoped pytest fixture).
    """
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    resource = Resource.create({"service.name": service_name})
    spans = InMemorySpanExporter()
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(SimpleSpanProcessor(spans))
    trace.set_tracer_provider(tp)
    reader = InMemoryMetricReader()
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))
    return spans, reader


def get_tracer(name: str) -> trace.Tracer:
    """Tracer from the current global provider."""
    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    """Meter from the current global provider."""
    return metrics.get_meter(name)
