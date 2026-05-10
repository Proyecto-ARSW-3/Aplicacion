import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from app.ml.model_manager import manager
from app.core.config import settings

router = APIRouter(prefix="/data", tags=["data"])

_training_status = {"status": "idle", "message": "No training started yet."}


def _run_training(path: str):
    global _training_status
    _training_status = {"status": "training", "message": "Training models..."}
    try:
        result = manager.train(path)
        _training_status = {
            "status": "complete",
            "message": f"Trained {len(result['models_trained'])} models on {result['total_samples']} samples.",
            "models_trained": result["models_trained"],
            "total_samples": result["total_samples"],
            "dropout_samples": result["dropout_samples"],
        }
    except Exception as e:
        _training_status = {"status": "error", "message": str(e)}


@router.post("/train")
def train_models(background_tasks: BackgroundTasks):
    dataset_path = settings.dataset_path
    if not Path(dataset_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found at {dataset_path}. "
                   "Upload it via POST /data/upload or place it manually.",
        )
    if _training_status.get("status") == "training":
        raise HTTPException(status_code=409, detail="Training already in progress.")
    background_tasks.add_task(_run_training, dataset_path)
    return {"message": "Training started in background. Poll /data/status for progress."}


@router.get("/status")
def training_status():
    return _training_status


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".tsv")):
        raise HTTPException(status_code=422, detail="Only CSV files are accepted.")
    dest = Path(settings.dataset_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": f"Dataset uploaded to {dest}. Call POST /data/train to start training."}


@router.get("/info")
def dataset_info():
    if not manager.is_trained:
        return {"trained": False}
    df = manager.df_raw
    target_map = {0: "Desertor", 1: "Graduado", 2: "Activo"}
    dist = df["Target"].map(target_map).value_counts().to_dict()
    return {
        "trained": True,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "class_distribution": dist,
        "threshold": manager.threshold,
    }
