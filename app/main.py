"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.middleware import setup_middleware
from app.routers import health_router, prediction_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    This factory pattern enables easy testing and configuration.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Plant Disease Classification API",
        description="API for predicting plant diseases from leaf images using EfficientNet-B2.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup exception handlers
    setup_middleware(app)

    # Register routers
    app.include_router(health_router)
    app.include_router(prediction_router)

    # Serve static files
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # Root route serves the HTML UI
    @app.get("/")
    async def root():
        return FileResponse("app/static/index.html")

    return app


# Create the app instance for uvicorn to load
app = create_app()
