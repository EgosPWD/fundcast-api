"""Domain entities and business rules. Standard library only (AGENTS.md section 5)."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.domain.errors import InvalidInputError

# Ordinal rank of each funding stage (AGENTS.md section 3).
# Stages without a rank (Undisclosed, Debt, Bridge, unknown) map to null.
STAGE_RANK: dict[str, int] = {
    "Grant": 0,
    "Angel": 0,
    "Crowdfunding": 0,
    "Pre-seed": 1,
    "Seed": 2,
    "Pre-Series A": 3,
    "Series A": 4,
    "Pre-Series B": 5,
    "Series B": 6,
    "Pre-Series C": 7,
    "Series C": 8,
    "Series D": 10,
    "Series E": 11,
    "Series F": 12,
    "Series G": 13,
    "Series H": 14,
}

# Exact input contract of the sklearn Pipeline, in order (AGENTS.md section 3).
MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "year",
    "company_age",
    "investor_count",
    "news_mentions",
    "has_lead_investor",
    "prev_log_amount",
    "stage_rank",
    "stage",
    "sector",
    "country",
    "city",
    "lead_investor",
    "news_source",
)

# Minimum columns of a CSV accepted by /retrain (AGENTS.md section 13).
# `news_excerpt` is never required nor used: it leaks the target.
REQUIRED_TRAINING_COLUMNS: frozenset[str] = frozenset(
    {
        "announced_date",
        "year",
        "company_domain",
        "founded_year",
        "stage",
        "sector",
        "country",
        "city",
        "lead_investor",
        "investor_count",
        "news_mentions",
        "news_source",
        "amount_usd",
        "amount_status",
    }
)

RANGE_FACTOR = 1.5
ACTIVATION_MAE_TOLERANCE = 0.01
MAX_TRAINING_FILE_BYTES = 20 * 1024 * 1024
UNDISCLOSED_STAGE = "Undisclosed"
UNDISCLOSED_WARNING = (
    "Stage 'Undisclosed' is less reliable: median absolute percentage error is about 67%."
)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _clean_required(name: str, value: str) -> str:
    stripped = (value or "").strip()
    if not stripped:
        raise InvalidInputError(f"{name} must not be empty")
    return stripped


class Role(StrEnum):
    USER = "user"
    ADMIN = "admin"


class TrainingFileStatus(StrEnum):
    UPLOADED = "uploaded"
    USED = "used"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FundingRoundInput:
    """What a client sends to POST /predict."""

    year: int
    founded_year: int
    stage: str
    sector: str
    country: str
    city: str | None = None
    investor_count: int | None = None
    lead_investor: str | None = None
    news_mentions: int | None = None
    news_source: str | None = None
    prev_amount_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("stage", "sector", "country"):
            object.__setattr__(self, name, _clean_required(name, getattr(self, name)))
        for name in ("city", "lead_investor", "news_source"):
            object.__setattr__(self, name, _clean_optional(getattr(self, name)))
        if self.prev_amount_usd is not None and self.prev_amount_usd <= 0:
            raise InvalidInputError("prev_amount_usd must be greater than 0")

    def to_features(self) -> dict[str, Any]:
        """Derive the 13 model columns, in contract order."""
        rank = STAGE_RANK.get(self.stage)
        return {
            "year": self.year,
            "company_age": self.year - self.founded_year,
            "investor_count": self.investor_count,
            "news_mentions": self.news_mentions,
            "has_lead_investor": int(self.lead_investor is not None),
            "prev_log_amount": (
                math.log10(self.prev_amount_usd) if self.prev_amount_usd is not None else None
            ),
            "stage_rank": float(rank) if rank is not None else None,
            "stage": self.stage,
            "sector": self.sector,
            "country": self.country,
            "city": self.city,
            "lead_investor": self.lead_investor,
            "news_source": self.news_source,
        }


@dataclass(frozen=True, slots=True)
class Estimate:
    """Model output converted back from log10 to USD."""

    estimated_log10: float
    estimated_amount_usd: float
    range_usd: tuple[float, float]
    warning: str | None = None

    @classmethod
    def from_log10(cls, log10_value: float, stage: str) -> "Estimate":
        amount = 10**log10_value
        return cls(
            estimated_log10=log10_value,
            estimated_amount_usd=amount,
            range_usd=(amount / RANGE_FACTOR, amount * RANGE_FACTOR),
            warning=UNDISCLOSED_WARNING if stage == UNDISCLOSED_STAGE else None,
        )


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """Test-set metrics of a trained model."""

    mae_log: float
    r2: float
    rmse_log: float
    mdape: float
    coverage: float


def should_activate(new: ModelMetrics, active: ModelMetrics | None) -> bool:
    """A retrained model goes live only if MAE_log_new <= MAE_log_active + 0.01."""
    if active is None:
        return True
    # Rounding absorbs float noise right at the tolerance boundary.
    return round(new.mae_log - active.mae_log, 9) <= ACTIVATION_MAE_TOLERANCE


@dataclass(slots=True)
class User:
    email: str
    password_hash: str
    role: Role = Role.USER
    id: int | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class TrainingFile:
    filename: str
    stored_path: str
    rows: int
    uploaded_by: int
    status: TrainingFileStatus = TrainingFileStatus.UPLOADED
    id: int | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class ModelVersion:
    version: int
    artifact_path: str
    metrics: ModelMetrics
    n_train: int
    n_test: int
    params: dict[str, Any] = field(default_factory=dict)
    is_active: bool = False
    training_file_id: int | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class Prediction:
    user_id: int
    model_version_id: int
    input: FundingRoundInput
    estimated_amount_usd: float
    estimated_log10: float
    id: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Page[T]:
    """One page of a paginated listing."""

    items: list[T]
    total: int
    page: int
    size: int
