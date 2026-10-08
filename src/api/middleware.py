"""HTTP-level metrics: request counts and latency per route."""
import time
from typing import Optional

from fastapi import Request
from prometheus_client import REGISTRY, CollectorRegistry, Counter, Histogram


def build_metrics_middleware(registry: Optional[CollectorRegistry] = None):
    """Create a middleware that records every request in `registry`.

    Requests are labeled by route template (for example `/predict`), and
    anything that matches no route is grouped under `unmatched`, so unknown
    URLs cannot create unbounded label values.
    """
    registry = registry or REGISTRY
    request_count = Counter(
        "api_requests",
        "Total API requests",
        ["method", "endpoint", "status_code"],
        registry=registry,
    )
    request_latency = Histogram(
        "api_request_latency_seconds",
        "API request latency in seconds",
        ["method", "endpoint"],
        registry=registry,
    )

    async def metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            endpoint = getattr(route, "path", "unmatched")
            request_count.labels(method=request.method, endpoint=endpoint, status_code=str(status_code)).inc()
            request_latency.labels(method=request.method, endpoint=endpoint).observe(time.perf_counter() - start)

    return metrics_middleware
