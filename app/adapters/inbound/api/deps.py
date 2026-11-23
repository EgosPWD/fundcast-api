"""FastAPI dependencies that hand the already-built use cases to the routers."""

from typing import TYPE_CHECKING

from fastapi import Request

from app.adapters.inbound.api.use_case_ports import (
    GetMetricsUseCase,
    GetOptionsUseCase,
    ListPredictionsUseCase,
    LoginUseCase,
    PredictRoundUseCase,
    RegisterUserUseCase,
    RetrainModelUseCase,
    UploadTrainingFileUseCase,
)
from app.application.ports.token_service import TokenService

if TYPE_CHECKING:
    from app.adapters.inbound.api.app_factory import ApiUseCases


def use_cases(request: Request) -> "ApiUseCases":
    return request.app.state.use_cases


def token_service(request: Request) -> TokenService:
    return use_cases(request).token_service


def register_user(request: Request) -> RegisterUserUseCase:
    return use_cases(request).register_user


def login(request: Request) -> LoginUseCase:
    return use_cases(request).login


def predict_round(request: Request) -> PredictRoundUseCase:
    return use_cases(request).predict_round


def list_predictions(request: Request) -> ListPredictionsUseCase:
    return use_cases(request).list_predictions


def get_metrics(request: Request) -> GetMetricsUseCase:
    return use_cases(request).get_metrics


def get_options(request: Request) -> GetOptionsUseCase:
    return use_cases(request).get_options


def upload_training_file(request: Request) -> UploadTrainingFileUseCase:
    return use_cases(request).upload_training_file


def retrain_model(request: Request) -> RetrainModelUseCase:
    return use_cases(request).retrain_model
