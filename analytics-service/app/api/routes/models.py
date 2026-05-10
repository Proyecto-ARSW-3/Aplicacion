from fastapi import APIRouter, HTTPException
from app.ml.model_manager import manager
from app.services.prediction_service import get_model_comparison, get_feature_importance

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/comparison")
def comparison():
    if not manager.is_trained:
        raise HTTPException(status_code=503, detail="Models not trained.")
    return get_model_comparison()


@router.get("/feature-importance")
def feature_importance(model_name: str | None = None):
    if not manager.is_trained:
        raise HTTPException(status_code=503, detail="Models not trained.")
    return get_feature_importance(model_name)


@router.get("/list")
def list_models():
    if not manager.is_trained:
        raise HTTPException(status_code=503, detail="Models not trained.")
    return [
        {
            "name": m.name,
            "auc_roc": round(m.auc_roc, 4),
            "accuracy": round(m.accuracy, 4),
            "is_best": m.name == manager.best_model.name if manager.best_model else False,
        }
        for m in manager.models
    ]
