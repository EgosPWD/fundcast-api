"""HTTP request and response models (AGENTS.md sections 3 and 6)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    created_at: datetime | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictRequest(BaseModel):
    """What the client sends; the API derives the 14 model columns from it.

    Unknown fields (for example `news_excerpt`) are ignored and never stored.
    """

    year: int = Field(examples=[2025])
    founded_year: int = Field(examples=[2021])
    stage: str = Field(examples=["Seed"])
    sector: str = Field(examples=["Fintech"])
    country: str = Field(examples=["United States"])
    city: str | None = Field(default=None, examples=["New York"])
    investor_count: int | None = Field(default=None, examples=[3])
    lead_investor: str | None = Field(default=None, examples=["Y Combinator"])
    news_mentions: int | None = Field(default=None, examples=[2])
    news_source: str | None = Field(default=None, examples=["TechCrunch"])
    company_domain: str | None = Field(default=None, examples=["acme.io"])
    prev_rounds: int | None = Field(default=None, examples=[1])
    prev_amount_usd: float | None = Field(default=None, examples=[1500000])


class PredictResponse(BaseModel):
    prediction_id: int
    estimated_amount_usd: float
    estimated_log10: float
    range_usd: tuple[float, float]
    model_version: int
    warning: str | None = None


class PredictionItem(BaseModel):
    id: int
    user_id: int
    model_version_id: int
    input: PredictRequest
    estimated_amount_usd: float
    estimated_log10: float
    created_at: datetime | None = None


class PredictionPage(BaseModel):
    items: list[PredictionItem]
    total: int
    page: int
    size: int


class MetricsSchema(BaseModel):
    mae_log: float
    r2: float
    rmse_log: float
    mdape: float
    coverage: float


class ModelVersionResponse(BaseModel):
    version: int
    is_active: bool
    metrics: MetricsSchema
    n_train: int
    n_test: int
    params: dict[str, Any]
    training_file_id: int | None = None
    created_at: datetime | None = None


class MetricsResponse(BaseModel):
    active: ModelVersionResponse | None
    versions: list[ModelVersionResponse]


class TrainingFileResponse(BaseModel):
    id: int
    filename: str
    rows: int
    rows_inserted: int
    rows_rejected: int
    status: str


class RetrainResponse(BaseModel):
    version: int
    activated: bool
    metrics: MetricsSchema
    n_train: int
    n_test: int
    message: str | None = None


class OptionsResponse(BaseModel):
    """Values stored in the training data; send them as they are (case-sensitive). A value
    from an upload the active model was not retrained with is treated as unknown."""

    sectors: list[str]
    stages: list[str]
    countries: list[str]


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str
