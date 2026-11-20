from typing import Annotated

from fastapi import APIRouter, Depends

from app.adapters.inbound.api import deps
from app.adapters.inbound.api.schemas import (
    ErrorResponse,
    MetricsResponse,
    MetricsSchema,
    ModelVersionResponse,
)
from app.adapters.inbound.api.security import AnyUser
from app.adapters.inbound.api.use_case_ports import GetMetricsUseCase
from app.domain.entities import ModelMetrics, ModelVersion

router = APIRouter(tags=["metrics"])


def metrics_schema(metrics: ModelMetrics) -> MetricsSchema:
    return MetricsSchema(mae_log=metrics.mae_log, r2=metrics.r2, rmse_log=metrics.rmse_log,
                         mdape=metrics.mdape, coverage=metrics.coverage)


def version_response(version: ModelVersion) -> ModelVersionResponse:
    # artifact_path is internal (server filesystem) and is not exposed.
    return ModelVersionResponse(
        version=version.version, is_active=version.is_active,
        metrics=metrics_schema(version.metrics), n_train=version.n_train,
        n_test=version.n_test, params=version.params,
        training_file_id=version.training_file_id, created_at=version.created_at,
    )


@router.get("/metrics", response_model=MetricsResponse,
            responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
def get_metrics(_: AnyUser,
                metrics_use_case: Annotated[GetMetricsUseCase, Depends(deps.get_metrics)]):
    """Metrics of the active model version and the history of every version."""
    report = metrics_use_case.execute()
    return MetricsResponse(
        active=version_response(report.active) if report.active else None,
        versions=[version_response(v) for v in report.versions],
    )
