"""TestClient wiring: real use cases, real bcrypt/JWT/predictor, in-memory repositories.

Upload and retrain use stubs with the S4 signatures (the real classes are rewritten in
a parallel unit), so the HTTP edge is tested independently of their internals.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapters.inbound.api.app_factory import ApiUseCases, create_app
from app.adapters.outbound.ml.sklearn_predictor import SklearnPredictor
from app.adapters.outbound.security.bcrypt_hasher import BcryptPasswordHasher
from app.adapters.outbound.security.jwt_token_service import JwtTokenService
from app.application.use_cases.get_metrics import GetMetrics
from app.application.use_cases.get_options import GetOptions
from app.application.use_cases.list_predictions import ListPredictions
from app.application.use_cases.login import Login
from app.application.use_cases.predict_round import PredictRound
from app.application.use_cases.register_user import RegisterUser
from datetime import date

from app.domain.entities import (
    FundingRound,
    ModelMetrics,
    ModelVersion,
    Role,
    TrainingFile,
    TrainingFileStatus,
    User,
)
from tests.unit.fakes import (
    InMemoryFundingRoundRepository,
    InMemoryModelVersionRepository,
    InMemoryPredictionRepository,
    InMemoryUserRepository,
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"
BASELINE = ARTIFACTS_DIR / "modelo_rondas.joblib"
JWT_SECRET = "integration-test-secret-long-enough-for-hs256"
ALLOWED_ORIGIN = "http://localhost:5173"
ADMIN_EMAIL, ADMIN_PASSWORD = "admin@example.com", "admin-password"
BASE_METRICS = ModelMetrics(mae_log=0.350, r2=0.615, rmse_log=0.495, mdape=0.529,
                            coverage=0.473)


@dataclass(frozen=True, slots=True)
class StubRetrainOutcome:
    model_version: ModelVersion
    activated: bool


@dataclass
class StubUploadTrainingFile:
    """S4 signature: execute(uploader_id, filename, content) -> TrainingFile."""

    error: Exception | None = None
    calls: list[tuple[int, str, bytes]] = field(default_factory=list)

    def execute(self, uploader_id: int, filename: str, content: bytes) -> TrainingFile:
        self.calls.append((uploader_id, filename, content))
        if self.error:
            raise self.error
        return TrainingFile(filename=filename, stored_path="/srv/uploads/abc.csv", rows=10,
                            uploaded_by=uploader_id, status=TrainingFileStatus.LOADED,
                            rows_inserted=8, rows_rejected=2, id=1)


@dataclass
class StubRetrainModel:
    """S4 signature: execute() -> RetrainOutcome(model_version, activated)."""

    activated: bool = True
    error: Exception | None = None
    calls: int = 0

    def execute(self) -> StubRetrainOutcome:
        self.calls += 1
        if self.error:
            raise self.error
        version = ModelVersion(version=2, artifact_path="/srv/artifacts/v2.joblib",
                               metrics=BASE_METRICS, n_train=12_758, n_test=3_190,
                               is_active=self.activated, id=2)
        return StubRetrainOutcome(model_version=version, activated=self.activated)


@dataclass
class ApiContext:
    client: TestClient
    users: InMemoryUserRepository
    predictions: InMemoryPredictionRepository
    models: InMemoryModelVersionRepository
    upload: StubUploadTrainingFile
    retrain: StubRetrainModel
    startup_calls: list[str]


@pytest.fixture(scope="session")
def predictor() -> SklearnPredictor:
    return SklearnPredictor(artifacts_dir=str(ARTIFACTS_DIR))


@pytest.fixture
def api(predictor) -> ApiContext:
    hasher = BcryptPasswordHasher(rounds=4)
    tokens = JwtTokenService(secret=JWT_SECRET, expire_minutes=5)
    users, predictions = InMemoryUserRepository(), InMemoryPredictionRepository()
    models = InMemoryModelVersionRepository()
    models.add(ModelVersion(version=1, artifact_path=str(BASELINE), metrics=BASE_METRICS,
                            n_train=12_758, n_test=3_190, is_active=True))
    users.add(User(email=ADMIN_EMAIL, password_hash=hasher.hash(ADMIN_PASSWORD),
                   role=Role.ADMIN))
    upload, retrain = StubUploadTrainingFile(), StubRetrainModel()
    rounds = InMemoryFundingRoundRepository([
        FundingRound(announced_date=date(2024, 1, 1), year=2024, company_domain=f"c{i}.io",
                     founded_year=2020, stage=stage, amount_status="verified",
                     round_id=f"r{i}", sector=sector, country=country, city=city)
        for i, (stage, sector, country, city) in enumerate([
            ("Seed", "Fintech", "Chile", "Santiago"),
            ("Series A", "AI-native", "United States", "New York"),
            ("Seed", "Fintech", None, "Atlantis City"),
            ("Seed", "Fintech", "United States", "Gotham"),
        ])
    ])
    startup_calls: list[str] = []
    app = create_app(
        ApiUseCases(
            register_user=RegisterUser(users, hasher),
            login=Login(users, hasher, tokens),
            predict_round=PredictRound(models, predictor, predictions, rounds),
            list_predictions=ListPredictions(predictions),
            get_metrics=GetMetrics(models),
            get_options=GetOptions(models, predictor, rounds),
            upload_training_file=upload,
            retrain_model=retrain,
            token_service=tokens,
        ),
        cors_origins=[ALLOWED_ORIGIN],
        on_startup=lambda: startup_calls.append("started"),
    )
    with TestClient(app) as client:
        yield ApiContext(client, users, predictions, models, upload, retrain, startup_calls)


def register_and_login(client: TestClient, email: str, password: str = "user-password") -> dict:
    response = client.post("/auth/register", json={"email": email, "password": password})
    assert response.status_code == 201, response.text
    return login_headers(client, email, password)


def login_headers(client: TestClient, email: str, password: str) -> dict:
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def user_headers(api) -> dict:
    return register_and_login(api.client, "user@example.com")


@pytest.fixture
def admin_headers(api) -> dict:
    return login_headers(api.client, ADMIN_EMAIL, ADMIN_PASSWORD)
