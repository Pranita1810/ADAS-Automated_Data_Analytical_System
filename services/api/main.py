from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware
from services.api.V1.endpoints.getdata import router as sales_router

app = FastAPI(
    title="ADA",
    swagger_ui_parameters={
        "syntaxHighlight": False,  # Prevents browser freeze in /docs when displaying 12k+ rows
        "displayRequestDuration": True
    }
)

# Enable GZip compression so 3.5MB JSON compresses to ~250KB over the network
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(sales_router)