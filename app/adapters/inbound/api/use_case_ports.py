"""Structural types of the use cases the API calls.

Routers depend on these shapes, not on concrete classes, so the HTTP edge stays stable
while use case internals change (seam contract S4 for upload and retrain).
"""

from typing import Protocol

from app.application.ports.token_service import TokenClaims
from app.application.use_cases.get_metrics import MetricsReport
from app.application.use_cases.predict_round import PredictionOutcome
from app.domain.entities import (
    CategoryOptions,
    FundingRoundInput,
    ModelVersion,
    Page,
    Prediction,
    TrainingFile,
    User,
)


class RegisterUserUseCase(Protocol):
    def execute(self, email: str, password: str) -> User: ...


class LoginUseCase(Protocol):
    def execute(self, email: str, password: str) -> str: ...


class PredictRoundUseCase(Protocol):
    def execute(self, user_id: int, round_input: FundingRoundInput) -> PredictionOutcome: ...


class ListPredictionsUseCase(Protocol):
    def execute(self, requester: TokenClaims, page: int, size: int) -> Page[Prediction]: ...


class GetMetricsUseCase(Protocol):
    def execute(self) -> MetricsReport: ...


class GetOptionsUseCase(Protocol):
    def execute(self) -> CategoryOptions: ...


class UploadTrainingFileUseCase(Protocol):
    def execute(self, uploader_id: int, filename: str, content: bytes) -> TrainingFile: ...


class RetrainResult(Protocol):
    """RetrainOutcome(model_version, activated) from S4."""

    @property
    def model_version(self) -> ModelVersion: ...

    @property
    def activated(self) -> bool: ...


class RetrainModelUseCase(Protocol):
    def execute(self) -> RetrainResult: ...
