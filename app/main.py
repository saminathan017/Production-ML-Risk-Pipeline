"""
NeuralNexus — Production ML Training & Inference Platform.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from pathlib import Path

from app.config import settings, BASE_DIR
from app.api import routes
from app.utils.logging import app_logger

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuralNexus — ML Risk Intelligence Platform",
    version=settings.app_version,
    description=(
        "End-to-end production ML pipeline: training, versioning, "
        "real-time inference, monitoring, and live WebSocket feed."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────

app.include_router(routes.router, prefix="/api", tags=["NeuralNexus API"])

# ─── Static files + SPA root ─────────────────────────────────────────────────

frontend_dir = BASE_DIR / "app" / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    index = frontend_dir / "index.html"
    if not index.exists():
        return {"message": "Frontend not found — ensure index.html exists in app/frontend/"}
    return FileResponse(index)


# ─── Lifecycle ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    # Record start time for uptime tracking
    Path("/tmp/neuralnexus_start.txt").write_text(str(time.time()))

    app_logger.info("=" * 60)
    app_logger.info(f"  NeuralNexus v{settings.app_version} starting up")
    app_logger.info(f"  Debug mode : {settings.debug}")
    app_logger.info(f"  Base dir   : {BASE_DIR}")
    app_logger.info("=" * 60)

    # Eagerly initialise inference engine so first request isn't slow
    try:
        from app.ml.inference import get_inference_engine
        engine = get_inference_engine()
        if engine.model:
            app_logger.info(f"  Model loaded: {engine.model_version} ({engine.model.model_type})")
        else:
            app_logger.warning("  No model loaded — run train_model.py first")
    except Exception as exc:
        app_logger.error(f"  Inference engine init failed: {exc}")


@app.on_event("shutdown")
async def shutdown_event():
    app_logger.info("NeuralNexus shutting down gracefully.")


# ─── Entry-point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
