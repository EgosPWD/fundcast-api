"""Composition root: wires ports to adapters and builds the FastAPI app (AGENTS.md section 5).

Run with `uvicorn app.main:app`. Settings come from the environment or `.env`.
"""

from pathlib import Path

from fastapi import FastAPI

from app.adapters.inbound.api.app_factory import ApiUseCases, create_app
from app.adapters.outbound.ml.csv_rounds_parser import CsvRoundsParser
from app.adapters.outbound.ml.joblib_model_inspector import JoblibModelInspector
from app.adapters.outbound.ml.sklearn_predictor import SklearnPredictor
from app.adapters.outbound.ml.sklearn_trainer import SklearnTrainer
from app.adapters.outbound.postgres.funding_round_repo import SqlAlchemyFundingRoundRepository
from app.adapters.outbound.postgres.model_repo import SqlAlchemyModelVersionRepository
from app.adapters.outbound.postgres.prediction_repo import SqlAlchemyPredictionRepository
from app.adapters.outbound.postgres.session import make_engine, make_session_factory
from app.adapters.outbound.postgres.training_file_repo import SqlAlchemyTrainingFileRepository
from app.adapters.outbound.postgres.user_repo import SqlAlchemyUserRepository
from app.adapters.outbound.security.bcrypt_hasher import BcryptPasswordHasher
from app.adapters.outbound.security.jwt_token_service import JwtTokenService
from app.adapters.outbound.storage.local_file_storage import LocalFileStorage
from app.adapters.outbound.storage.thread_retrain_lock import ThreadRetrainLock
from app.application.use_cases.ensure_admin import EnsureAdmin
from app.application.use_cases.ensure_baseline_model import EnsureBaselineModel
from app.application.use_cases.get_metrics import GetMetrics
from app.application.use_cases.list_predictions import ListPredictions
from app.application.use_cases.login import Login
from app.application.use_cases.predict_round import PredictRound
from app.application.use_cases.register_user import RegisterUser
from app.application.use_cases.retrain_model import RetrainModel
from app.application.use_cases.seed_funding_rounds import SeedFundingRounds
from app.application.use_cases.upload_training_file import UploadTrainingFile
from app.config import Settings

BASELINE_ARTIFACT = "modelo_rondas.joblib"
# Split sizes of the notebook baseline (AGENTS.md section 1); not stored in the artifact.
BASELINE_N_TRAIN, BASELINE_N_TEST = 12_758, 3_190


def build_app(settings: Settings) -> FastAPI:
    sessions = make_session_factory(make_engine(settings.database_url))
    users = SqlAlchemyUserRepository(sessions)
    models = SqlAlchemyModelVersionRepository(sessions)
    predictions = SqlAlchemyPredictionRepository(sessions)
    files = SqlAlchemyTrainingFileRepository(sessions)
    rounds = SqlAlchemyFundingRoundRepository(sessions)

    hasher = BcryptPasswordHasher()
    tokens = JwtTokenService(
        settings.jwt_secret, settings.jwt_algorithm, settings.jwt_expire_minutes
    )
    parser = CsvRoundsParser()

    use_cases = ApiUseCases(
        register_user=RegisterUser(users, hasher),
        login=Login(users, hasher, tokens),
        predict_round=PredictRound(
            models, SklearnPredictor(settings.artifacts_dir), predictions, rounds
        ),
        list_predictions=ListPredictions(predictions),
        get_metrics=GetMetrics(models),
        upload_training_file=UploadTrainingFile(
            LocalFileStorage(settings.uploads_dir), parser, files, rounds
        ),
        retrain_model=RetrainModel(
            rounds, files, models, SklearnTrainer(), ThreadRetrainLock(),
            settings.artifacts_dir, SklearnTrainer.WINNING_PARAMS,
        ),
        token_service=tokens,
    )

    def on_startup() -> None:
        # AGENTS.md section 6 "Arranque": seed the original CSV, register the baseline
        # model as active version 1 and create the admin; each step is idempotent.
        SeedFundingRounds(rounds, parser).execute(settings.seed_dataset_path)
        EnsureBaselineModel(models, JoblibModelInspector(settings.artifacts_dir)).execute(
            str(Path(settings.artifacts_dir) / BASELINE_ARTIFACT),
            n_train=BASELINE_N_TRAIN, n_test=BASELINE_N_TEST,
        )
        EnsureAdmin(users, hasher).execute(settings.admin_email, settings.admin_password)

    return create_app(use_cases, settings.cors_origins, on_startup=on_startup)


def __getattr__(name: str) -> FastAPI:
    # `uvicorn app.main:app` builds the app on first access, so importing this module
    # (tests, tooling) does not require the environment variables.
    if name == "app":
        return build_app(Settings())  # pyright: ignore[reportCallIssue]
    raise AttributeError(name)
