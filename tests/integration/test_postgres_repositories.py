"""Repository tests against a real PostgreSQL migrated with Alembic.

Run with TEST_DATABASE_URL pointing to a disposable database, for example:
TEST_DATABASE_URL=postgresql+psycopg://test:test@localhost:55432/startups_test
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.adapters.outbound.postgres.model_repo import SqlAlchemyModelVersionRepository
from app.adapters.outbound.postgres.prediction_repo import SqlAlchemyPredictionRepository
from app.adapters.outbound.postgres.session import make_engine, make_session_factory
from app.adapters.outbound.postgres.training_file_repo import SqlAlchemyTrainingFileRepository
from app.adapters.outbound.postgres.user_repo import SqlAlchemyUserRepository
from app.domain.entities import (
    FundingRoundInput,
    ModelMetrics,
    ModelVersion,
    Prediction,
    Role,
    TrainingFile,
    TrainingFileStatus,
    User,
)

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def session_factory():
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, check=True,
                   env={**os.environ, "DATABASE_URL": DATABASE_URL}, capture_output=True)
    engine = make_engine(DATABASE_URL)
    yield make_session_factory(engine)
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_tables(session_factory):
    with session_factory.begin() as session:
        session.execute(text("TRUNCATE predictions, model_versions, training_files, users "
                             "RESTART IDENTITY CASCADE"))


@pytest.fixture
def users(session_factory):
    return SqlAlchemyUserRepository(session_factory)


@pytest.fixture
def models(session_factory):
    return SqlAlchemyModelVersionRepository(session_factory)


def metrics(mae_log=0.352) -> ModelMetrics:
    return ModelMetrics(mae_log=mae_log, r2=0.62, rmse_log=0.494, mdape=0.525, coverage=0.479)


def new_version(number: int, active: bool = False) -> ModelVersion:
    return ModelVersion(version=number, artifact_path=f"artifacts/model_v{number}.joblib",
                        metrics=metrics(), n_train=12_758, n_test=3_190,
                        params={"model__max_iter": 400}, is_active=active)


class TestUserRepository:
    def test_add_and_fetch_by_email_and_id(self, users):
        saved = users.add(User(email="ana@mail.com", password_hash="h", role=Role.ADMIN))

        assert saved.id is not None and saved.created_at is not None
        assert users.get_by_email("ana@mail.com") == saved
        assert users.get_by_id(saved.id) == saved
        assert users.get_by_email("ghost@mail.com") is None

    def test_email_is_unique(self, users):
        users.add(User(email="ana@mail.com", password_hash="h"))

        with pytest.raises(IntegrityError):
            users.add(User(email="ana@mail.com", password_hash="h2"))


class TestModelVersionRepository:
    def test_activate_keeps_a_single_active_version(self, models):
        first = models.add(new_version(1, active=True))
        second = models.add(new_version(2))

        models.activate(second.id)

        assert models.get_active().id == second.id
        assert [v.version for v in models.list_all()] == [2, 1]
        assert [v.is_active for v in models.list_all()] == [True, False]
        assert models.next_version_number() == 3
        assert first.params == {"model__max_iter": 400}

    def test_database_rejects_two_active_versions(self, models):
        models.add(new_version(1, active=True))

        with pytest.raises(IntegrityError):
            models.add(new_version(2, active=True))

    def test_empty_repository(self, models):
        assert models.get_active() is None
        assert models.list_all() == []
        assert models.next_version_number() == 1


class TestTrainingFileRepository:
    def test_add_latest_and_status_update(self, session_factory, users):
        files = SqlAlchemyTrainingFileRepository(session_factory)
        admin = users.add(User(email="admin@mail.com", password_hash="h", role=Role.ADMIN))
        files.add(TrainingFile(filename="a.csv", stored_path="up/a.csv", rows=1,
                               uploaded_by=admin.id))
        latest = files.add(TrainingFile(filename="b.csv", stored_path="up/b.csv", rows=2,
                                        uploaded_by=admin.id))

        files.update_status(latest.id, TrainingFileStatus.USED)

        assert files.get_latest().id == latest.id
        assert files.get_by_id(latest.id).status is TrainingFileStatus.USED
        assert files.get_by_id(999) is None


class TestPredictionRepository:
    def test_round_trip_and_filtered_pagination(self, session_factory, users, models):
        predictions = SqlAlchemyPredictionRepository(session_factory)
        ana = users.add(User(email="ana@mail.com", password_hash="h"))
        bob = users.add(User(email="bob@mail.com", password_hash="h"))
        version = models.add(new_version(1, active=True))
        round_input = FundingRoundInput(year=2025, founded_year=2021, stage="Seed",
                                        sector="Fintech", country="Chile", prev_amount_usd=1e6)
        for user in (ana, bob, ana, ana):
            predictions.add(Prediction(user_id=user.id, model_version_id=version.id,
                                       input=round_input, estimated_amount_usd=1e6,
                                       estimated_log10=6.0))

        first_page = predictions.list(user_id=ana.id, page=1, size=2)
        second_page = predictions.list(user_id=ana.id, page=2, size=2)

        assert first_page.total == 3
        assert len(first_page.items) == 2 and len(second_page.items) == 1
        assert {p.user_id for p in first_page.items + second_page.items} == {ana.id}
        assert first_page.items[0].id > first_page.items[1].id
        assert first_page.items[0].input == round_input
        assert predictions.list(user_id=None, page=1, size=10).total == 4
