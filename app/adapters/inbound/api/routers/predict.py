from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.adapters.inbound.api import deps
from app.adapters.inbound.api.schemas import (
    ErrorResponse,
    PredictionItem,
    PredictionPage,
    PredictRequest,
    PredictResponse,
)
from app.adapters.inbound.api.security import AnyUser
from app.adapters.inbound.api.use_case_ports import ListPredictionsUseCase, PredictRoundUseCase
from app.domain.entities import FundingRoundInput

router = APIRouter(tags=["predictions"])
AUTH_ERRORS = {401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}}


@router.post("/predict", response_model=PredictResponse,
             responses={**AUTH_ERRORS, 422: {"model": ErrorResponse},
                        503: {"model": ErrorResponse}})
def predict(body: PredictRequest, claims: AnyUser,
            predict_round: Annotated[PredictRoundUseCase, Depends(deps.predict_round)]):
    """Estimate the round amount in USD and store it in the caller's history."""
    outcome = predict_round.execute(claims.user_id, FundingRoundInput(**body.model_dump()))
    estimate = outcome.estimate
    return PredictResponse(
        prediction_id=outcome.prediction.id,
        estimated_amount_usd=estimate.estimated_amount_usd,
        estimated_log10=estimate.estimated_log10,
        range_usd=estimate.range_usd,
        model_version=outcome.model_version,
        warning=estimate.warning,
    )


@router.get("/predictions", response_model=PredictionPage,
            responses={**AUTH_ERRORS, 422: {"model": ErrorResponse}})
def list_predictions(
    claims: AnyUser,
    list_use_case: Annotated[ListPredictionsUseCase, Depends(deps.list_predictions)],
    page: Annotated[int, Query(description="1-based page number")] = 1,
    size: Annotated[int, Query(description="items per page, 1-100")] = 20,
):
    """History, newest first: a `user` sees only their own predictions, an `admin` all."""
    result = list_use_case.execute(claims, page=page, size=size)
    return PredictionPage(
        items=[
            PredictionItem(
                id=p.id, user_id=p.user_id, model_version_id=p.model_version_id,
                input=PredictRequest(**asdict(p.input)),
                estimated_amount_usd=p.estimated_amount_usd,
                estimated_log10=p.estimated_log10, created_at=p.created_at,
            )
            for p in result.items
        ],
        total=result.total, page=result.page, size=result.size,
    )
