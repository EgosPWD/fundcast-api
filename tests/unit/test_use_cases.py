import pytest

from app.application.ports.token_service import TokenClaims
from app.application.use_cases.ensure_admin import EnsureAdmin
from app.application.use_cases.get_metrics import GetMetrics
from app.application.use_cases.list_predictions import ListPredictions
from app.application.use_cases.login import Login
from app.application.use_cases.predict_round import PredictRound
from app.application.use_cases.register_user import RegisterUser
from app.application.use_cases.retrain_model import RetrainModel
from app.application.use_cases.upload_training_file import UploadTrainingFile
from app.domain.entities import (
    MAX_TRAINING_FILE_BYTES,
    FundingRoundInput,
    ModelMetrics,
    ModelVersion,
    Prediction,
    Role,
    TrainingFile,
    TrainingFileStatus,
    User,
)
from app.domain.errors import (
    AdminProvisioningError,
    AuthenticationError,
    EmailAlreadyRegisteredError,
    InvalidInputError,
    InvalidTrainingFileError,
    ModelNotAvailableError,
    NotFoundError,
    RetrainInProgressError,
)
from tests.unit.fakes import (
    FakeFileStorage,
    FakePasswordHasher,
    FakePredictor,
    FakeRetrainLock,
    FakeTokenService,
    FakeTrainer,
    InMemoryModelVersionRepository,
    InMemoryPredictionRepository,
    InMemoryTrainingFileRepository,
    InMemoryUserRepository,
)

ADMIN = TokenClaims(user_id=99, role=Role.ADMIN)


def metrics(mae_log: float = 0.352) -> ModelMetrics:
    return ModelMetrics(mae_log=mae_log, r2=0.62, rmse_log=0.494, mdape=0.525, coverage=0.479)


def add_active_model(models: InMemoryModelVersionRepository, mae_log: float = 0.352):
    version = models.add(ModelVersion(version=models.next_version_number(),
                                      artifact_path="artifacts/modelo_rondas.joblib",
                                      metrics=metrics(mae_log), n_train=12_757, n_test=3_190))
    models.activate(version.id)
    return version


def round_input(**overrides) -> FundingRoundInput:
    data = {"year": 2025, "founded_year": 2021, "stage": "Seed", "sector": "Fintech",
            "country": "Chile"}
    data.update(overrides)
    return FundingRoundInput(**data)


class TestRegisterUser:
    def test_creates_user_with_role_user_and_hashed_password(self):
        users = InMemoryUserRepository()

        user = RegisterUser(users, FakePasswordHasher()).execute("  Ana@Mail.com ", "secret123")

        assert user.id == 1
        assert user.email == "ana@mail.com"
        assert user.role is Role.USER
        assert user.password_hash == "hashed:secret123"
        assert users.get_by_email("ana@mail.com") == user

    def test_duplicate_email_is_rejected(self):
        users = InMemoryUserRepository()
        RegisterUser(users, FakePasswordHasher()).execute("ana@mail.com", "secret123")

        with pytest.raises(EmailAlreadyRegisteredError):
            RegisterUser(users, FakePasswordHasher()).execute("ANA@mail.com", "other1234")

    @pytest.mark.parametrize(("email", "password"), [("no-at-sign", "secret123"),
                                                     ("ana@mail.com", "short")])
    def test_invalid_email_or_short_password_is_rejected(self, email, password):
        with pytest.raises(InvalidInputError):
            RegisterUser(InMemoryUserRepository(), FakePasswordHasher()).execute(email, password)


class TestEnsureAdmin:
    def test_creates_admin_once(self):
        users = InMemoryUserRepository()
        ensure = EnsureAdmin(users, FakePasswordHasher())

        ensure.execute("admin@example.com", "admin-pass")
        ensure.execute("admin@example.com", "admin-pass")

        assert len(users.users) == 1
        assert users.get_by_email("admin@example.com").role is Role.ADMIN


class TestLogin:
    def setup_method(self):
        self.users = InMemoryUserRepository()
        RegisterUser(self.users, FakePasswordHasher()).execute("ana@mail.com", "secret123")
        self.login = Login(self.users, FakePasswordHasher(), FakeTokenService())

    def test_valid_credentials_return_a_token_with_id_and_role(self):
        assert self.login.execute("ANA@mail.com", "secret123") == "token:1:user"

    @pytest.mark.parametrize(("email", "password"), [("ana@mail.com", "wrong-pass"),
                                                     ("ghost@mail.com", "secret123")])
    def test_wrong_credentials_raise_authentication_error(self, email, password):
        with pytest.raises(AuthenticationError, match="Invalid email or password"):
            self.login.execute(email, password)


class TestPredictRound:
    def test_predicts_with_active_model_and_saves_history(self):
        models, predictions, predictor = (InMemoryModelVersionRepository(),
                                          InMemoryPredictionRepository(), FakePredictor(6.0))
        active = add_active_model(models)

        outcome = PredictRound(models, predictor, predictions).execute(7, round_input())

        assert predictor.calls == [(active.artifact_path, round_input().to_features())]
        assert outcome.estimate.estimated_amount_usd == pytest.approx(1_000_000)
        assert outcome.model_version == active.version
        assert outcome.prediction.id == 1
        assert predictions.items == [outcome.prediction]
        saved = predictions.items[0]
        assert (saved.user_id, saved.model_version_id) == (7, active.id)
        assert saved.estimated_log10 == 6.0
        assert saved.input == round_input()

    def test_without_active_model_raises_and_saves_nothing(self):
        predictions = InMemoryPredictionRepository()

        with pytest.raises(ModelNotAvailableError):
            PredictRound(InMemoryModelVersionRepository(), FakePredictor(),
                         predictions).execute(7, round_input())
        assert predictions.items == []


class TestListPredictions:
    def setup_method(self):
        self.repo = InMemoryPredictionRepository()
        for user_id in (1, 2, 1):
            self.repo.add(Prediction(user_id=user_id, model_version_id=1, input=round_input(),
                                     estimated_amount_usd=1.0, estimated_log10=0.0))

    def test_user_only_sees_own_predictions(self):
        page = ListPredictions(self.repo).execute(TokenClaims(1, Role.USER), page=1, size=10)

        assert page.total == 2
        assert {p.user_id for p in page.items} == {1}

    def test_admin_sees_every_prediction(self):
        assert ListPredictions(self.repo).execute(ADMIN, page=1, size=10).total == 3

    @pytest.mark.parametrize(("page", "size"), [(0, 10), (1, 0), (1, 101)])
    def test_invalid_pagination_is_rejected(self, page, size):
        with pytest.raises(InvalidInputError):
            ListPredictions(self.repo).execute(ADMIN, page=page, size=size)


class TestGetMetrics:
    def test_returns_active_version_and_full_history(self):
        models = InMemoryModelVersionRepository()
        first = add_active_model(models)
        second = add_active_model(models, mae_log=0.34)

        report = GetMetrics(models).execute()

        assert report.active.id == second.id
        assert [v.version for v in report.versions] == [second.version, first.version]

    def test_works_without_models(self):
        report = GetMetrics(InMemoryModelVersionRepository()).execute()

        assert report.active is None
        assert report.versions == []


class TestUploadTrainingFile:
    def setup_method(self):
        self.storage, self.trainer = FakeFileStorage(), FakeTrainer(rows=250)
        self.files = InMemoryTrainingFileRepository()
        self.upload = UploadTrainingFile(self.storage, self.trainer, self.files)

    def test_valid_csv_is_stored_and_registered(self):
        training_file = self.upload.execute(99, "rounds.CSV", b"a,b\n1,2\n")

        assert training_file.id == 1
        assert training_file.rows == 250
        assert training_file.status is TrainingFileStatus.UPLOADED
        assert training_file.uploaded_by == 99
        assert self.storage.files[training_file.stored_path] == b"a,b\n1,2\n"

    @pytest.mark.parametrize("filename", ["rounds.txt", "rounds.csv.exe", "rounds"])
    def test_non_csv_extension_is_rejected_before_storing(self, filename):
        with pytest.raises(InvalidTrainingFileError, match=".csv"):
            self.upload.execute(99, filename, b"a,b\n")
        assert self.storage.files == {}

    def test_file_over_20_mb_is_rejected_before_storing(self):
        with pytest.raises(InvalidTrainingFileError, match="20 MB"):
            self.upload.execute(99, "big.csv", b"x" * (MAX_TRAINING_FILE_BYTES + 1))
        assert self.storage.files == {}

    def test_invalid_columns_delete_the_stored_file(self):
        self.trainer.invalid = True

        with pytest.raises(InvalidTrainingFileError, match="amount_usd"):
            self.upload.execute(99, "rounds.csv", b"a,b\n")
        assert self.storage.files == {}
        assert self.files.files == {}


class TestRetrainModel:
    PARAMS = {"model__learning_rate": 0.05}

    def build(self, trainer=None, lock=None):
        self.models = InMemoryModelVersionRepository()
        self.files = InMemoryTrainingFileRepository()
        self.trainer = trainer or FakeTrainer()
        self.lock = lock or FakeRetrainLock()
        self.active = add_active_model(self.models)
        self.upload = self.files.add(TrainingFile(filename="new.csv", stored_path="up/new.csv",
                                                  rows=10, uploaded_by=99))
        return RetrainModel(self.files, self.models, self.trainer, self.lock,
                            baseline_dataset_path="data/base.csv", artifacts_dir="artifacts",
                            params=self.PARAMS)

    def test_trains_on_baseline_plus_file_and_activates_if_not_worse(self):
        outcome = self.build(FakeTrainer(mae_log=0.360)).execute(file_id=None)

        assert self.trainer.train_calls == [
            (["data/base.csv", "up/new.csv"], self.PARAMS, "artifacts/model_v2.joblib")]
        assert outcome.activated is True
        assert outcome.model_version.version == 2
        assert outcome.model_version.training_file_id == self.upload.id
        assert self.models.get_active().version == 2
        assert self.files.get_by_id(self.upload.id).status is TrainingFileStatus.USED
        assert self.lock.taken is False

    def test_worse_model_is_saved_inactive(self):
        outcome = self.build(FakeTrainer(mae_log=0.40)).execute(file_id=None)

        assert outcome.activated is False
        assert outcome.model_version.is_active is False
        assert self.models.get_active().id == self.active.id
        assert len(self.models.list_all()) == 2

    def test_uses_the_requested_file_id(self):
        retrain = self.build()
        other = self.files.add(TrainingFile(filename="b.csv", stored_path="up/b.csv", rows=5,
                                            uploaded_by=99))
        retrain.execute(file_id=self.upload.id)

        assert self.trainer.train_calls[0][0] == ["data/base.csv", "up/new.csv"]
        assert other.status is TrainingFileStatus.UPLOADED

    def test_unknown_file_id_raises_not_found(self):
        with pytest.raises(NotFoundError):
            self.build().execute(file_id=404)
        assert self.lock.taken is False

    def test_no_uploaded_files_raises_not_found(self):
        retrain = self.build()
        self.files.files.clear()

        with pytest.raises(NotFoundError):
            retrain.execute(file_id=None)

    def test_running_retrain_raises_conflict_without_training(self):
        retrain = self.build(lock=FakeRetrainLock(taken=True))

        with pytest.raises(RetrainInProgressError):
            retrain.execute(file_id=None)
        assert self.trainer.train_calls == []

    def test_training_failure_marks_file_failed_and_releases_lock(self):
        retrain = self.build(FakeTrainer(fail=RuntimeError("boom")))

        with pytest.raises(RuntimeError):
            retrain.execute(file_id=None)
        assert self.files.get_by_id(self.upload.id).status is TrainingFileStatus.FAILED
        assert self.lock.taken is False
        assert len(self.models.list_all()) == 1


def test_register_user_rejects_admin_escalation_by_design():
    """Registration has no role parameter: only EnsureAdmin creates admins."""
    users = InMemoryUserRepository()
    user = RegisterUser(users, FakePasswordHasher()).execute("x@mail.com", "secret123")

    assert isinstance(user, User) and user.role is Role.USER


class TestReviewFindings:
    """Regression tests for the CodeRabbit review of PR #3."""

    def test_ensure_admin_refuses_an_existing_user_account(self):
        users = InMemoryUserRepository()
        RegisterUser(users, FakePasswordHasher()).execute("admin@example.com", "public-pass")

        with pytest.raises(AdminProvisioningError, match="admin@example.com"):
            EnsureAdmin(users, FakePasswordHasher()).execute("admin@example.com", "admin-pass")
        account = users.get_by_email("admin@example.com")
        assert account.role is Role.USER
        assert account.password_hash == "hashed:public-pass"

    def test_activated_outcome_reports_is_active(self):
        retrain = TestRetrainModel().build(FakeTrainer(mae_log=0.30))

        outcome = retrain.execute(file_id=None)

        assert outcome.activated is True
        assert outcome.model_version.is_active is True

    def test_failure_while_saving_the_model_marks_the_file_failed(self):
        case = TestRetrainModel()
        retrain = case.build()

        def broken_add(model_version):
            raise RuntimeError("database down")

        case.models.add = broken_add
        with pytest.raises(RuntimeError, match="database down"):
            retrain.execute(file_id=None)
        assert case.files.get_by_id(case.upload.id).status is TrainingFileStatus.FAILED
        assert case.lock.taken is False

    def test_upload_deletes_the_stored_file_if_registration_fails(self):
        storage, files = FakeFileStorage(), InMemoryTrainingFileRepository()

        def broken_add(training_file):
            raise RuntimeError("database down")

        files.add = broken_add
        with pytest.raises(RuntimeError, match="database down"):
            UploadTrainingFile(storage, FakeTrainer(), files).execute(99, "rounds.csv", b"a\n")
        assert storage.files == {}
