try:
    from opentelemetry import trace
    tracer = trace.get_tracer("modelexpress")
except ImportError:
    from contextlib import contextmanager

    class _NoOpTracer:
        @contextmanager
        def start_as_current_span(self, *_a, **_kw):
            yield None

    tracer = _NoOpTracer()