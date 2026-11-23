from datetime import date

import pytest

from app.application.ports.token_service import TokenClaims
from app.application.ports.training_data_parser import ParsedRounds
from app.application.use_cases.ensure_admin import EnsureAdmin
from app.application.use_cases.ensure_baseline_model import EnsureBaselineModel
from app.application.use_cases.get_metrics import GetMetrics
from app.application.use_cases.get_options import GetOptions
from app.application.use_cases.list_predictions import ListPredictions
from app.application.use_cases.login import Login
from app.application.use_cases.predict_round import PredictRound
from app.application.use_cases.register_user import RegisterUser
from app.application.use_cases.retrain_model import RetrainModel
from app.application.use_cases.seed_funding_rounds import SeedFundingRounds
from app.application.use_cases.upload_training_file import UploadTrainingFile
from app.domain.entities import (
    CategoryWarning,
    MAX_TRAINING_FILE_BYTES,
    FundingRound,
    FundingRoundInput,
    ModelMetrics,
    ModelVersion,
    ModelVocabulary,
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
    InsufficientTrainingDataError,
    InvalidInputError,
    InvalidTrainingFileError,
    ModelNotAvailableError,
    RetrainInProgressError,
)
from tests.unit.fakes import (
    FakeFileStorage,
    FakeInspector,
    FakePasswordHasher,
    FakePredictor,
    FakeRetrainLock,
    FakeTokenService,
    FakeTrainer,
    FakeTrainingDataParser,
    InMemoryFundingRoundRepository,
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
    def build(self, rounds=None, log10_value=6.0):
        self.models, self.predictions = InMemoryModelVersionRepository(), InMemoryPredictionRepository()
        self.predictor = FakePredictor(log10_value, lead_freq_map={"Y Combinator": 41})
        self.active = add_active_model(self.models)
        return PredictRound(self.models, self.predictor, self.predictions,
                            InMemoryFundingRoundRepository(rounds))

    def test_predicts_with_active_model_and_saves_history(self):
        predict = self.build()
        request = round_input(lead_investor="Y Combinator", prev_rounds=2, prev_amount_usd=1e6)

        outcome = predict.execute(7, request)

        expected = request.to_features(history=None, lead_freq_map={"Y Combinator": 41})
        assert self.predictor.calls == [(self.active.artifact_path, expected)]
        assert expected["lead_freq"] == 42 and expected["n_prev_rounds"] == 2
        assert outcome.estimate.estimated_amount_usd == pytest.approx(1_000_000)
        assert outcome.model_version == self.active.version
        assert outcome.prediction.id == 1
        saved = self.predictions.items[0]
        assert (saved.user_id, saved.model_version_id) == (7, self.active.id)
        assert saved.estimated_log10 == 6.0
        assert saved.input == request

    def test_unknown_categories_are_reported_as_warnings(self):
        predict = self.build()
        predict.execute(7, round_input(sector="Fintech"))
        self.predictor.vocabularies[self.active.artifact_path] = ModelVocabulary(
            stages=frozenset({"Seed"}), sectors=frozenset({"Fintech"}),
            countries=frozenset({"Chile"}), cities=frozenset(), lead_investors=frozenset(),
            news_sources=frozenset())

        outcome = predict.execute(7, round_input(news_source="TechCrunch", country="Chile"))

        assert outcome.warnings == [
            CategoryWarning(code="unknown_category", field="news_source", value="TechCrunch")]

    def test_known_categories_give_no_warnings(self):
        assert self.build().execute(7, round_input()).warnings == []

    def test_known_company_domain_uses_stored_history(self):
        from datetime import date

        from app.domain.entities import FundingRound

        def stored(year, month, amount, domain="acme.io"):
            return FundingRound(announced_date=date(year, month, 1), year=year,
                                company_domain=domain, founded_year=2020, stage="Seed",
                                amount_status="verified", amount_usd=amount)

        predict = self.build([stored(2023, 1, 1e6), stored(2024, 3, 1e7), stored(2026, 1, 5e7),
                              stored(2024, 6, 9e9, domain="other.io")])

        predict.execute(7, round_input(year=2025, company_domain="acme.io", prev_rounds=9,
                                       prev_amount_usd=5))

        sent = self.predictor.calls[0][1]
        assert sent["n_prev_rounds"] == 2
        assert sent["prev_log_amount"] == pytest.approx(7.0)

    def test_unknown_company_domain_falls_back_to_client_values(self):
        predict = self.build()

        predict.execute(7, round_input(company_domain="new.io", prev_rounds=1, prev_amount_usd=1e5))

        sent = self.predictor.calls[0][1]
        assert (sent["n_prev_rounds"], sent["prev_log_amount"]) == (1, pytest.approx(5.0))

    def test_known_company_without_earlier_rounds_uses_empty_stored_history(self):
        """CodeRabbit review of PR #6: a known domain never falls back to client values."""
        from datetime import date

        from app.domain.entities import FundingRound

        later = FundingRound(announced_date=date(2026, 1, 1), year=2026, company_domain="acme.io",
                             founded_year=2020, stage="Seed", amount_status="verified",
                             amount_usd=5e7)
        predict = self.build([later])

        predict.execute(7, round_input(year=2025, company_domain="acme.io", prev_rounds=9,
                                       prev_amount_usd=1e5))

        sent = self.predictor.calls[0][1]
        assert (sent["n_prev_rounds"], sent["prev_log_amount"]) == (0, None)

    def test_explicit_empty_lead_map_counts_only_the_round_itself(self):
        predict = self.build()
        self.predictor.lead_freq_map = {}
        assert FakePredictor(lead_freq_map={}).lead_freq_map == {}

        predict.execute(7, round_input(lead_investor="Y Combinator"))

        assert self.predictor.calls[0][1]["lead_freq"] == 1

    def test_without_active_model_raises_and_saves_nothing(self):
        predictions = InMemoryPredictionRepository()

        with pytest.raises(ModelNotAvailableError):
            PredictRound(InMemoryModelVersionRepository(), FakePredictor(), predictions,
                         InMemoryFundingRoundRepository()).execute(7, round_input())
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


class TestGetOptions:
    VOCAB = ModelVocabulary(
        stages=frozenset({"Seed", "Angel"}), sectors=frozenset({"Fintech", "AI-native"}),
        countries=frozenset({"Chile", "Brazil"}), cities=frozenset({"Santiago", "Sao Paulo"}),
        lead_investors=frozenset({"Y Combinator", "Sequoia"}),
        news_sources=frozenset({"techcrunch.com"}))

    def build(self, rounds=None, vocabulary=None):
        self.models = InMemoryModelVersionRepository()
        self.active = add_active_model(self.models)
        self.predictor = FakePredictor(vocabulary=vocabulary or self.VOCAB)
        return GetOptions(self.models, self.predictor, InMemoryFundingRoundRepository(rounds))

    def test_lists_come_from_the_active_model_sorted(self):
        options = self.build().execute()

        assert options.stages == ["Angel", "Seed"]
        assert options.sectors == ["AI-native", "Fintech"]
        assert options.countries == ["Brazil", "Chile"]
        assert options.lead_investors == ["Sequoia", "Y Combinator"]
        assert options.news_sources == ["techcrunch.com"]
        assert options.model_version == self.active.version

    def test_values_only_in_the_database_are_not_offered(self):
        options = self.build([funding_round("r1", sector="Biotech", stage="Series Z",
                                            country="Peru", city="Lima")]).execute()

        assert "Biotech" not in options.sectors and "Series Z" not in options.stages
        assert "Peru" not in options.countries and options.cities_by_country == {}

    def test_cities_by_country_pairs_come_from_rounds_filtered_by_the_model(self):
        options = self.build([
            funding_round("r1", country="Chile", city="Santiago"),
            funding_round("r2", country="Chile", city="Santiago"),
            funding_round("r3", country="Brazil", city="Sao Paulo"),
            funding_round("r4", country="Chile", city="Sao Paulo"),
            funding_round("r5", country="Chile", city="Valparaiso"),
            funding_round("r6", country="Peru", city="Santiago"),
            funding_round("r7", country="Chile", city=None),
        ]).execute()

        assert options.cities_by_country == {"Brazil": ["Sao Paulo"],
                                             "Chile": ["Santiago", "Sao Paulo"]}

    def test_follows_the_active_version_after_a_retrain_activates_another(self):
        use_case = self.build()
        new_vocab = ModelVocabulary(
            stages=frozenset({"Seed"}), sectors=frozenset({"Biotech"}),
            countries=frozenset({"Peru"}), cities=frozenset(), lead_investors=frozenset(),
            news_sources=frozenset())
        second = self.models.add(ModelVersion(
            version=2, artifact_path="artifacts/v2.joblib", metrics=metrics(0.35),
            n_train=1, n_test=1))
        self.predictor.vocabularies[second.artifact_path] = new_vocab
        before = use_case.execute()
        self.models.activate(second.id)

        after = use_case.execute()

        assert (before.model_version, after.model_version) == (1, 2)
        assert after.sectors == ["Biotech"] and before.sectors == ["AI-native", "Fintech"]

    def test_without_an_active_model_it_is_unavailable(self):
        use_case = GetOptions(InMemoryModelVersionRepository(), FakePredictor(),
                              InMemoryFundingRoundRepository())

        with pytest.raises(ModelNotAvailableError):
            use_case.execute()


def funding_round(round_id: str | None = "r1", **overrides) -> FundingRound:
    values = dict(announced_date=date(2025, 3, 1), year=2025, company_domain="acme.io",
                  founded_year=2021, stage="Seed", amount_status="verified",
                  amount_usd=1_500_000.0, round_id=round_id)
    return FundingRound(**{**values, **overrides})


def parsed(*round_ids: str | None, rejected: int = 0) -> ParsedRounds:
    return ParsedRounds(rounds=[funding_round(r) for r in round_ids], rejected=rejected)


class TestUploadTrainingFile:
    def setup_method(self):
        self.storage = FakeFileStorage()
        self.parser = FakeTrainingDataParser(parsed("a", "b", "c", rejected=2))
        self.files = InMemoryTrainingFileRepository()
        self.rounds = InMemoryFundingRoundRepository([funding_round("old")])
        self.upload = UploadTrainingFile(self.storage, self.parser, self.files, self.rounds)

    def test_valid_csv_is_stored_parsed_and_its_rows_inserted_as_loaded(self):
        training_file = self.upload.execute(99, "rounds.CSV", b"a,b\n1,2\n")

        assert training_file.id == 1
        assert training_file.status is TrainingFileStatus.LOADED
        assert (training_file.rows, training_file.rows_inserted,
                training_file.rows_rejected) == (5, 3, 2)
        assert training_file.uploaded_by == 99
        assert training_file.filename == "rounds.CSV"
        assert self.storage.files[training_file.stored_path] == b"a,b\n1,2\n"
        assert self.parser.calls == [training_file.stored_path]
        stored = self.files.get_by_id(training_file.id)
        assert stored == training_file
        new_rows = self.rounds.list_all()[1:]
        assert [r.round_id for r in new_rows] == ["a", "b", "c"]
        assert {r.training_file_id for r in new_rows} == {training_file.id}
        assert self.rounds.list_all()[0].training_file_id is None

    def test_duplicate_round_ids_are_ignored_and_not_counted_as_inserted(self):
        self.parser.parsed = parsed("old", "a", None, None)

        training_file = self.upload.execute(99, "rounds.csv", b"x")

        assert training_file.status is TrainingFileStatus.LOADED
        assert (training_file.rows, training_file.rows_inserted,
                training_file.rows_rejected) == (4, 3, 0)
        assert [r.round_id for r in self.rounds.list_all()] == ["old", "a", None, None]

    @pytest.mark.parametrize("filename", ["rounds.txt", "rounds.csv.exe", "rounds", ""])
    def test_non_csv_extension_is_rejected_before_storing(self, filename):
        with pytest.raises(InvalidTrainingFileError, match=".csv"):
            self.upload.execute(99, filename, b"a,b\n")
        assert self.storage.files == {}
        assert self.parser.calls == []
        assert self.files.files == {}

    def test_file_over_20_mb_is_rejected_before_storing(self):
        with pytest.raises(InvalidTrainingFileError, match="20 MB"):
            self.upload.execute(99, "big.csv", b"x" * (MAX_TRAINING_FILE_BYTES + 1))
        assert self.storage.files == {}
        assert self.parser.calls == []

    def test_file_of_exactly_20_mb_is_accepted(self):
        training_file = self.upload.execute(99, "big.csv", b"x" * MAX_TRAINING_FILE_BYTES)

        assert training_file.status is TrainingFileStatus.LOADED

    def test_invalid_file_deletes_the_stored_csv_and_registers_nothing(self):
        self.parser.fail = InvalidTrainingFileError("missing required columns: amount_usd")

        with pytest.raises(InvalidTrainingFileError, match="amount_usd"):
            self.upload.execute(99, "rounds.csv", b"a,b\n")
        assert self.storage.files == {}
        assert self.files.files == {}
        assert self.rounds.count() == 1

    def test_failed_registration_deletes_the_stored_csv(self):
        def broken_add(training_file):
            raise RuntimeError("database down")

        self.files.add = broken_add
        with pytest.raises(RuntimeError, match="database down"):
            self.upload.execute(99, "rounds.csv", b"a\n")
        assert self.storage.files == {}
        assert self.rounds.count() == 1

    def test_failed_insert_never_leaves_a_loaded_record(self):
        def broken_add_many(rounds):
            raise RuntimeError("database down")

        self.rounds.add_many = broken_add_many
        with pytest.raises(RuntimeError, match="database down"):
            self.upload.execute(99, "rounds.csv", b"a\n")
        assert self.storage.files == {}
        assert [f.status for f in self.files.files.values()] == [TrainingFileStatus.FAILED]
        assert self.files.get_latest(TrainingFileStatus.LOADED) is None


class TestRetrainModel:
    PARAMS = {"model__learning_rate": 0.05}

    def build(self, trainer=None, lock=None, with_upload=True):
        self.models = InMemoryModelVersionRepository()
        self.files = InMemoryTrainingFileRepository()
        self.rounds = InMemoryFundingRoundRepository([funding_round("r1"), funding_round("r2")])
        self.trainer = trainer or FakeTrainer()
        self.lock = lock or FakeRetrainLock()
        self.active = add_active_model(self.models)
        self.upload = None
        if with_upload:
            self.upload = self.files.add(TrainingFile(
                filename="new.csv", stored_path="up/new.csv", rows=1, uploaded_by=99,
                status=TrainingFileStatus.LOADED))
            self.rounds.add_many([funding_round("r3", training_file_id=self.upload.id)])
        return RetrainModel(self.rounds, self.files, self.models, self.trainer, self.lock,
                            artifacts_dir="artifacts/", params=self.PARAMS)

    def test_trains_on_every_stored_round_and_activates_if_not_worse(self):
        outcome = self.build(FakeTrainer(mae_log=0.360)).execute()

        [(rounds, params, artifact_path)] = self.trainer.train_calls
        assert [r.round_id for r in rounds] == ["r1", "r2", "r3"]
        assert [r.id for r in rounds] == [1, 2, 3]
        assert (params, artifact_path) == (self.PARAMS, "artifacts/model_v2.joblib")
        assert outcome.activated is True
        assert outcome.model_version.is_active is True
        assert outcome.model_version.version == 2
        assert outcome.model_version.artifact_path == "artifacts/model_v2.joblib"
        assert outcome.model_version.training_file_id == self.upload.id
        assert (outcome.model_version.n_train, outcome.model_version.n_test) == (3, 0)
        assert self.models.get_active().version == 2
        assert self.lock.taken is False

    def test_without_uploads_training_file_id_is_none(self):
        outcome = self.build(with_upload=False).execute()

        assert [r.round_id for r in self.trainer.train_calls[0][0]] == ["r1", "r2"]
        assert outcome.model_version.training_file_id is None

    def test_training_file_id_skips_a_failed_load(self):
        retrain = self.build()
        self.files.add(TrainingFile(filename="bad.csv", stored_path="up/bad.csv", rows=1,
                                    uploaded_by=99, status=TrainingFileStatus.FAILED))

        outcome = retrain.execute()

        assert outcome.model_version.training_file_id == self.upload.id

    def test_activation_compares_with_the_stored_active_mae(self):
        # Active MAE 0.352: 0.362 is within the 0.01 tolerance, 0.3621 is not.
        assert self.build(FakeTrainer(mae_log=0.362)).execute().activated is True
        outcome = self.build(FakeTrainer(mae_log=0.3621)).execute()

        assert outcome.activated is False

    def test_worse_model_is_saved_inactive_and_the_active_one_kept(self):
        outcome = self.build(FakeTrainer(mae_log=0.40)).execute()

        assert outcome.activated is False
        assert outcome.model_version.is_active is False
        assert self.models.get_active().id == self.active.id
        assert len(self.models.list_all()) == 2

    def test_without_an_active_model_the_new_version_is_activated(self):
        retrain = self.build(FakeTrainer(mae_log=0.9))
        self.models.versions.clear()

        outcome = retrain.execute()

        assert outcome.activated is True
        assert outcome.model_version.version == 1
        assert self.models.get_active().id == outcome.model_version.id

    def test_running_retrain_raises_conflict_without_training(self):
        retrain = self.build(lock=FakeRetrainLock(taken=True))

        with pytest.raises(RetrainInProgressError):
            retrain.execute()
        assert self.trainer.train_calls == []
        assert self.lock.released == 0

    def test_insufficient_data_is_a_422_that_keeps_the_active_model(self):
        retrain = self.build(FakeTrainer(fail=InsufficientTrainingDataError("too few rows")))

        with pytest.raises(InsufficientTrainingDataError, match="too few rows"):
            retrain.execute()
        assert self.models.get_active().id == self.active.id
        assert len(self.models.list_all()) == 1
        assert self.lock.taken is False

    def test_training_failure_keeps_the_active_model_and_releases_lock(self):
        retrain = self.build(FakeTrainer(fail=RuntimeError("boom")))

        with pytest.raises(RuntimeError):
            retrain.execute()
        assert self.models.get_active().id == self.active.id
        assert self.files.get_by_id(self.upload.id).status is TrainingFileStatus.LOADED
        assert self.lock.taken is False
        assert len(self.models.list_all()) == 1

    def test_failure_while_saving_the_model_keeps_the_active_model(self):
        retrain = self.build()

        def broken_add(model_version):
            raise RuntimeError("database down")

        self.models.add = broken_add
        with pytest.raises(RuntimeError, match="database down"):
            retrain.execute()
        assert self.models.get_active().id == self.active.id
        assert self.lock.taken is False


class TestSeedFundingRounds:
    def test_loads_the_original_csv_into_an_empty_table(self, tmp_path):
        dataset = tmp_path / "original.csv"
        dataset.write_text("x")
        rounds, parser = InMemoryFundingRoundRepository(), FakeTrainingDataParser(parsed("a", "b"))

        inserted = SeedFundingRounds(rounds, parser).execute(str(dataset))

        assert inserted == 2
        assert parser.calls == [str(dataset)]
        assert [r.round_id for r in rounds.list_all()] == ["a", "b"]
        assert {r.training_file_id for r in rounds.list_all()} == {None}

    def test_does_nothing_if_the_table_has_rows(self, tmp_path):
        dataset = tmp_path / "original.csv"
        dataset.write_text("x")
        rounds = InMemoryFundingRoundRepository([funding_round("old")])
        parser = FakeTrainingDataParser(parsed("a"))

        assert SeedFundingRounds(rounds, parser).execute(str(dataset)) == 0
        assert parser.calls == []
        assert rounds.count() == 1

    @pytest.mark.parametrize("dataset_path", [None, "", "data/does-not-exist.csv"])
    def test_does_nothing_without_an_existing_path(self, dataset_path):
        rounds, parser = InMemoryFundingRoundRepository(), FakeTrainingDataParser(parsed("a"))

        assert SeedFundingRounds(rounds, parser).execute(dataset_path) == 0
        assert parser.calls == []
        assert rounds.count() == 0


def test_teacher_flow_seed_retrain_upload_retrain(tmp_path):
    """Upload data -> retrain -> upload more -> retrain (AGENTS.md section 12, step 9)."""
    dataset = tmp_path / "original.csv"
    dataset.write_text("x")
    rounds, files = InMemoryFundingRoundRepository(), InMemoryTrainingFileRepository()
    models, trainer = InMemoryModelVersionRepository(), FakeTrainer(mae_log=0.35)
    add_active_model(models)
    retrain = RetrainModel(rounds, files, models, trainer, FakeRetrainLock(),
                           artifacts_dir="artifacts", params={"p": 1})

    seeded = SeedFundingRounds(rounds, FakeTrainingDataParser(parsed("o1", "o2"))).execute(
        str(dataset))
    first = retrain.execute()
    upload = UploadTrainingFile(FakeFileStorage(), FakeTrainingDataParser(parsed("u1", "o1")),
                                files, rounds).execute(99, "more.csv", b"x")
    second = retrain.execute()

    assert seeded == 2
    assert (upload.rows, upload.rows_inserted) == (2, 1)
    assert (first.model_version.version, second.model_version.version) == (2, 3)
    assert [[r.round_id for r in call[0]] for call in trainer.train_calls] == [
        ["o1", "o2"], ["o1", "o2", "u1"]]
    assert first.model_version.training_file_id is None
    assert second.model_version.training_file_id == upload.id
    assert models.get_active().version == 3


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


def test_training_file_status_has_no_default():
    """Every load must state its outcome explicitly (CodeRabbit review of PR #4)."""
    with pytest.raises(TypeError):
        TrainingFile(filename="a.csv", stored_path="up/a.csv", rows=1, uploaded_by=1)


class TestEnsureBaselineModel:
    def test_registers_baseline_as_active_version_1_once(self):
        models, inspector = InMemoryModelVersionRepository(), FakeInspector(metrics())
        ensure = EnsureBaselineModel(models, inspector)

        created = ensure.execute("artifacts/modelo_rondas.joblib", n_train=12_758, n_test=3_190)
        again = ensure.execute("artifacts/modelo_rondas.joblib", n_train=12_758, n_test=3_190)

        assert again is None
        assert inspector.calls == ["artifacts/modelo_rondas.joblib"]
        assert models.get_active() == created
        assert (created.version, created.n_train, created.n_test) == (1, 12_758, 3_190)
        assert created.params == {"model__max_iter": 400}
        assert created.training_file_id is None

    def test_baseline_stored_metrics_drive_the_first_retrain_activation(self):
        models = InMemoryModelVersionRepository()
        EnsureBaselineModel(models, FakeInspector(metrics(mae_log=0.3504))).execute(
            "artifacts/modelo_rondas.joblib", n_train=12_758, n_test=3_190)
        retrain = RetrainModel(InMemoryFundingRoundRepository([funding_round()]),
                               InMemoryTrainingFileRepository(), models,
                               FakeTrainer(mae_log=0.3605), FakeRetrainLock(),
                               artifacts_dir="artifacts", params={})

        outcome = retrain.execute()

        assert outcome.activated is False
        assert models.get_active().version == 1

    def test_unreadable_artifact_registers_nothing(self):
        models = InMemoryModelVersionRepository()

        with pytest.raises(ModelNotAvailableError):
            EnsureBaselineModel(models, FakeInspector(fail=ModelNotAvailableError("x"))).execute(
                "artifacts/missing.joblib", n_train=1, n_test=1)
        assert models.list_all() == []
