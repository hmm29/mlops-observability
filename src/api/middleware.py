# src/api/middleware.py
from fastapi import Request
import time
from prometheus_client import Counter, Histogram

# API metrics
REQUEST_COUNT = Counter(
    'api_requests_total', 
    'Total API requests', 
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds', 
    'API request latency in seconds',
    ['method', 'endpoint']
)

async def metrics_middleware(request: Request, call_next):
    """Middleware to collect API metrics"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    latency = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method, 
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(latency)
    
    return response
