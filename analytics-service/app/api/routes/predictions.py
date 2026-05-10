from fastapi import APIRouter, HTTPException, Query
from app.ml.model_manager import manager
from app.schemas.schemas import ThresholdRequest, PredictionRequest
from app.services.prediction_service import get_students_list

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/students")
def students(
    threshold: float | None = Query(None, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    risk_filter: str | None = Query(None, pattern="^(alto|medio|bajo|all)?$"),
):
    if not manager.is_trained:
        raise HTTPException(status_code=503, detail="Models not trained.")
    return get_students_list(threshold, page, page_size, risk_filter)


@router.post("/threshold")
def update_threshold(body: ThresholdRequest):
    if not manager.is_trained:
        raise HTTPException(status_code=503, detail="Models not trained.")
    if not 0.0 <= body.threshold <= 1.0:
        raise HTTPException(status_code=422, detail="Threshold must be between 0 and 1.")
    return manager.update_threshold(body.threshold)


@router.get("/threshold")
def current_threshold():
    return {
        "threshold": manager.threshold,
        "optimal_threshold": manager.best_model.optimal_threshold if manager.best_model else 0.5,
    }


@router.post("/predict")
def predict_student(body: PredictionRequest):
    if not manager.is_trained:
        raise HTTPException(status_code=503, detail="Models not trained.")
    features = body.model_dump()
    return manager.predict_single(features)
