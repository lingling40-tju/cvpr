"""Process-local workaround for Instrumentator 7.1 / Starlette 0.52 routing.

The profiler only needs a route label. Starlette now exposes an included router
without a ``path`` attribute, which crashes the installed metrics middleware.
Using the request path as its metrics label leaves model inference untouched.
"""

try:
    from prometheus_fastapi_instrumentator import routing

    routing.get_route_name = lambda request: request.scope.get("path")
except ImportError:
    pass
