"""
Main FastAPI application for ML Risk Pipeline.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.config import settings, BASE_DIR
from app.api import routes
from app.utils.logging import app_logger

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production ML Training & Inference Pipeline for Risk Scoring"
)

# Include API routes
app.include_router(routes.router, prefix="/api", tags=["api"])

# Mount static files for frontend
frontend_dir = BASE_DIR / "app" / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def root():
    """Serve the frontend application."""
    frontend_path = frontend_dir / "index.html"
    if not frontend_path.exists():
        return {"message": "Frontend not found. Please ensure index.html exists."}
    return FileResponse(frontend_path)


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    app_logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    app_logger.info(f"Debug mode: {settings.debug}")
    app_logger.info(f"Base directory: {BASE_DIR}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    app_logger.info(f"Shutting down {settings.app_name}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
