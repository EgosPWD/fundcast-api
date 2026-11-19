"""In-memory fakes of the application ports, for use-case unit tests.

Repositories return copies, like a real database: a fake that hands out its stored
objects hides bugs where a use case relies on in-place mutation.
"""

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from app.application.ports.token_service import TokenClaims
from app.application.ports.trainer import TrainingResult
from app.domain.entities import (
    ModelMetrics,
    ModelVersion,
    Page,
    Prediction,
    TrainingFile,
    TrainingFileStatus,
    User,
)
from app.domain.errors import AuthenticationError, InvalidTrainingFileError

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class InMemoryUserRepository:
    def __init__(self) -> None:
        self.users: dict[int, User] = {}

    def add(self, user: User) -> User:
        stored = replace(user, id=len(self.users) + 1, created_at=NOW)
        self.users[stored.id] = stored
        return stored

    def get_by_email(self, email: str) -> User | None:
        return next((u for u in self.users.values() if u.email == email), None)

    def get_by_id(self, user_id: int) -> User | None:
        return self.users.get(user_id)


class InMemoryPredictionRepository:
    def __init__(self) -> None:
        self.items: list[Prediction] = []

    def add(self, prediction: Prediction) -> Prediction:
        stored = replace(prediction, id=len(self.items) + 1, created_at=NOW)
        self.items.append(stored)
        return stored

    def list(self, user_id: int | None, page: int, size: int) -> Page[Prediction]:
        rows = [p for p in reversed(self.items) if user_id is None or p.user_id == user_id]
        start = (page - 1) * size
        return Page(items=rows[start : start + size], total=len(rows), page=page, size=size)


class InMemoryModelVersionRepository:
    def __init__(self) -> None:
        self.versions: dict[int, ModelVersion] = {}

    def add(self, model_version: ModelVersion) -> ModelVersion:
        stored = replace(model_version, id=len(self.versions) + 1, created_at=NOW)
        self.versions[stored.id] = stored
        return replace(stored)

    def get_active(self) -> ModelVersion | None:
        active = next((v for v in self.versions.values() if v.is_active), None)
        return replace(active) if active else None

    def list_all(self) -> list[ModelVersion]:
        rows = sorted(self.versions.values(), key=lambda v: v.version, reverse=True)
        return [replace(v) for v in rows]

    def activate(self, model_version_id: int) -> None:
        for v in self.versions.values():
            v.is_active = v.id == model_version_id

    def next_version_number(self) -> int:
        return max((v.version for v in self.versions.values()), default=0) + 1


class InMemoryTrainingFileRepository:
    def __init__(self) -> None:
        self.files: dict[int, TrainingFile] = {}

    def add(self, training_file: TrainingFile) -> TrainingFile:
        stored = replace(training_file, id=len(self.files) + 1, created_at=NOW)
        self.files[stored.id] = stored
        return stored

    def get_by_id(self, training_file_id: int) -> TrainingFile | None:
        return self.files.get(training_file_id)

    def get_latest(self) -> TrainingFile | None:
        return max(self.files.values(), key=lambda f: f.id or 0, default=None)

    def update_status(self, training_file_id: int, status: TrainingFileStatus) -> None:
        self.files[training_file_id].status = status


class InMemoryFundingRoundRepository:
    def __init__(self, rounds=None) -> None:
        self.rounds = list(rounds or [])

    def add_many(self, rounds) -> int:
        self.rounds.extend(rounds)
        return len(rounds)

    def list_all(self):
        return list(self.rounds)

    def count(self) -> int:
        return len(self.rounds)

    def history(self, company_domain: str, up_to_year: int):
        rows = [r for r in self.rounds if r.company_domain == company_domain and r.year <= up_to_year]
        return sorted(rows, key=lambda r: r.announced_date)


class FakePasswordHasher:
    def hash(self, plain_password: str) -> str:
        return f"hashed:{plain_password}"

    def verify(self, plain_password: str, password_hash: str) -> bool:
        return password_hash == f"hashed:{plain_password}"


class FakeTokenService:
    def issue(self, claims: TokenClaims) -> str:
        return f"token:{claims.user_id}:{claims.role}"

    def decode(self, token: str) -> TokenClaims:
        raise AuthenticationError("not used in these tests")


class FakePredictor:
    def __init__(self, log10_value: float = 6.0, lead_freq_map=None) -> None:
        self.log10_value = log10_value
        self.lead_freq_map = lead_freq_map or {"Y Combinator": 41}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def lead_frequency_map(self, artifact_path: str) -> dict[str, int]:
        return self.lead_freq_map

    def predict_log10(self, artifact_path: str, features: dict[str, Any]) -> float:
        self.calls.append((artifact_path, features))
        return self.log10_value


class FakeTrainer:
    def __init__(self, mae_log: float = 0.35, rows: int = 100, fail: Exception | None = None):
        self.mae_log, self.rows, self.fail = mae_log, rows, fail
        self.train_calls: list[tuple[Sequence[str], dict[str, Any], str]] = []
        self.invalid = False

    def validate_dataset(self, path: str) -> int:
        if self.invalid:
            raise InvalidTrainingFileError("missing required columns: amount_usd")
        return self.rows

    def train(
        self, source_paths: Sequence[str], params: dict[str, Any], artifact_path: str
    ) -> TrainingResult:
        self.train_calls.append((list(source_paths), params, artifact_path))
        if self.fail:
            raise self.fail
        metrics = ModelMetrics(mae_log=self.mae_log, r2=0.62, rmse_log=0.49, mdape=0.52,
                               coverage=0.48)
        return TrainingResult(artifact_path=artifact_path, metrics=metrics, n_train=800,
                              n_test=200, params=params)


class FakeFileStorage:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def save(self, filename: str, content: bytes) -> str:
        path = f"uploads/{len(self.files) + 1}_{filename}"
        self.files[path] = content
        return path

    def delete(self, stored_path: str) -> None:
        self.files.pop(stored_path, None)


class FakeRetrainLock:
    def __init__(self, taken: bool = False) -> None:
        self.taken = taken
        self.released = 0

    def acquire(self) -> bool:
        if self.taken:
            return False
        self.taken = True
        return True

    def release(self) -> None:
        self.taken = False
        self.released += 1
