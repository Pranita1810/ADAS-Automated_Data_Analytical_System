import time
from fastapi import FastAPI, Request
from starlette.middleware.gzip import GZipMiddleware
from services.api.V1.endpoints.getdata import router as sales_router

try:
    from mahgaurd import get_logger, metrics
    logger = get_logger("ADAS.API")
except ImportError:
    import logging
    logger = logging.getLogger("ADAS.API")
    metrics = None

app = FastAPI(
    title="ADA",
    swagger_ui_parameters={
        "syntaxHighlight": False,  # Prevents browser freeze in /docs when displaying 12k+ rows
        "displayRequestDuration": True
    }
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration*1000.0:.1f}ms)")
    if metrics:
        metrics.record_duration(f"API {request.method} {request.url.path}", duration)
    return response

# Enable GZip compression so 3.5MB JSON compresses to ~250KB over the network
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(sales_router)