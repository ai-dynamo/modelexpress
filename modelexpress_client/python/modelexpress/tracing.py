try:
    from opentelemetry import trace
    tracer = trace.get_tracer("modelexpress")
except ImportError:
    from contextlib import contextmanager

    class _NoOpSpan:
        def set_attribute(self, *_a, **_kw):
            pass

    class _NoOpTracer:
        @contextmanager
        def start_as_current_span(self, *_a, **_kw):
            yield _NoOpSpan()

    tracer = _NoOpTracer()
