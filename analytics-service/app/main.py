from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import analytics, models, predictions, data
from app.ml.model_manager import manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-train if dataset exists at startup
    dataset_path = settings.dataset_path
    if Path(dataset_path).exists():
        print(f"[Startup] Dataset found at {dataset_path}. Training models...")
        try:
            result = manager.train(dataset_path)
            print(f"[Startup] Training complete: {result}")
        except Exception as e:
            print(f"[Startup] Training failed: {e}")
    else:
        print(f"[Startup] No dataset found at {dataset_path}. Upload via POST /api/data/upload")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analytics.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(data.router, prefix="/api")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "trained": manager.is_trained,
        "version": settings.app_version,
    }
