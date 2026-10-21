import pytest

from tracing.otel import setup_in_memory

_SPANS, _READER = setup_in_memory()


@pytest.fixture
def otel():
    """In-memory OTel exporters (installed once per test process)."""
    _SPANS.clear()
    return _SPANS, _READER
