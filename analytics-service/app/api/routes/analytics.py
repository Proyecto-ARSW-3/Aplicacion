from fastapi import APIRouter, HTTPException
from app.ml.model_manager import manager
from app.services.analytics_service import get_overview, get_survival_data, get_risk_distribution

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview")
def overview():
    if not manager.is_trained:
        raise HTTPException(status_code=503, detail="Models not trained. Call /data/train first.")
    return get_overview()


@router.get("/survival")
def survival():
    if not manager.is_trained:
        raise HTTPException(status_code=503, detail="Models not trained.")
    return get_survival_data()


@router.get("/risk-distribution")
def risk_distribution():
    if not manager.is_trained:
        raise HTTPException(status_code=503, detail="Models not trained.")
    return get_risk_distribution()


@router.get("/roc")
def roc_curve(model_name: str | None = None):
    if not manager.is_trained:
        raise HTTPException(status_code=503, detail="Models not trained.")
    return manager.get_roc_analysis(model_name)
