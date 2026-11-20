import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.adapters.outbound.ml import sklearn_predictor
from app.adapters.outbound.ml.sklearn_predictor import SklearnPredictor
from app.domain.entities import MODEL_FEATURE_COLUMNS, FundingRoundInput
from app.domain.errors import ModelNotAvailableError

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
BASELINE = ARTIFACTS_DIR / "modelo_rondas.joblib"
LEAD_FREQ_MAP = joblib.load(BASELINE)["lead_freq_map"]


@pytest.fixture
def predictor() -> SklearnPredictor:
    return SklearnPredictor(artifacts_dir=str(ARTIFACTS_DIR))


def features(**overrides) -> dict:
    data = {"year": 2025, "founded_year": 2021, "stage": "Seed", "sector": "Fintech",
            "country": "United States"}
    data.update(overrides)
    return FundingRoundInput(**data).to_features(history=None, lead_freq_map=LEAD_FREQ_MAP)


def contract_example() -> dict:
    return features(city="New York", investor_count=3, lead_investor="Y Combinator",
                    news_mentions=2, news_source="TechCrunch", prev_amount_usd=1_500_000)


def test_contract_example_returns_a_plausible_log10(predictor):
    log10_amount = predictor.predict_log10(str(BASELINE), contract_example())

    assert isinstance(log10_amount, float)
    assert 3.0 <= log10_amount <= 10.0


def test_matches_the_pipeline_fed_with_csv_style_nulls(predictor):
    """Nulls must reach the pipeline as NaN, the way read_csv produced them at training time."""
    row = features(lead_investor=None, city=None, prev_amount_usd=None)
    bundle = joblib.load(BASELINE)
    numeric = [c for name, _, cols in bundle["pipeline"].named_steps["prep"].transformers_
               if name == "num" for c in cols]
    frame = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in row.items()}],
                         columns=bundle["columns"]).astype({c: "float64" for c in numeric})
    expected = float(bundle["pipeline"].predict(frame)[0])

    assert predictor.predict_log10(str(BASELINE), row) == pytest.approx(expected)


def test_minimal_input_with_nulls_predicts(predictor):
    assert math.isfinite(predictor.predict_log10(str(BASELINE), features()))


def test_unknown_categories_do_not_break(predictor):
    row = features(stage="Series Z", sector="Quantum Bakery", country="Atlantis",
                   city="Nowhere", lead_investor="Nobody Capital", news_source="Unknown Blog")

    assert math.isfinite(predictor.predict_log10(str(BASELINE), row))


def test_artifact_is_loaded_once_and_cached(predictor, monkeypatch):
    calls = []
    real_load = sklearn_predictor.joblib.load
    monkeypatch.setattr(sklearn_predictor.joblib, "load",
                        lambda path: calls.append(path) or real_load(path))

    predictor.predict_log10(str(BASELINE), contract_example())
    predictor.predict_log10(str(BASELINE), features())

    assert len(calls) == 1


def test_missing_artifact_raises_model_not_available(predictor):
    with pytest.raises(ModelNotAvailableError):
        predictor.predict_log10(str(ARTIFACTS_DIR / "does_not_exist.joblib"), features())


def test_refuses_to_load_files_outside_the_artifacts_dir(tmp_path):
    outside = tmp_path / "evil.joblib"
    joblib.dump({"anything": 1}, outside)
    predictor = SklearnPredictor(artifacts_dir=str(ARTIFACTS_DIR))

    with pytest.raises(ModelNotAvailableError, match="outside"):
        predictor.predict_log10(str(outside), features())


@pytest.mark.parametrize("columns", [None, "year", [], [1, 2], ["year", None],
                                     list(MODEL_FEATURE_COLUMNS[:-1]),
                                     list(reversed(MODEL_FEATURE_COLUMNS))])
def test_invalid_column_metadata_raises_model_not_available(tmp_path, columns):
    pipeline = joblib.load(BASELINE)["pipeline"]
    joblib.dump({"pipeline": pipeline, "columns": columns}, tmp_path / "bad_columns.joblib")
    predictor = SklearnPredictor(artifacts_dir=str(tmp_path))

    with pytest.raises(ModelNotAvailableError, match="columns"):
        predictor.predict_log10(str(tmp_path / "bad_columns.joblib"), features())


def test_malformed_bundle_raises_model_not_available(tmp_path):
    joblib.dump({"pipeline": None}, tmp_path / "broken.joblib")
    predictor = SklearnPredictor(artifacts_dir=str(tmp_path))

    with pytest.raises(ModelNotAvailableError):
        predictor.predict_log10(str(tmp_path / "broken.joblib"), features())


def test_baseline_bundle_matches_the_14_column_contract(predictor):
    bundle = joblib.load(BASELINE)

    assert tuple(bundle["columns"]) == MODEL_FEATURE_COLUMNS
    assert len(predictor.lead_frequency_map(str(BASELINE))) == 8_463
    assert predictor.lead_frequency_map(str(BASELINE)) == bundle["lead_freq_map"]


def test_prediction_raises_no_pandas_warning(predictor):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predictor.predict_log10(str(BASELINE), features(lead_investor=None, city=None))


def test_old_13_column_artifact_is_rejected(tmp_path):
    """CodeRabbit review of PR #6: a stale but self-consistent artifact must not serve
    predictions, or the missing n_prev_rounds / lead_freq would silently become NaN."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    num = ["year", "company_age", "investor_count", "news_mentions", "has_lead_investor",
           "prev_log_amount", "stage_rank"]
    cat = ["stage", "sector", "country", "city", "lead_investor", "news_source"]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({**{c: rng.random(30) for c in num}, **{c: ["a", "b"] * 15 for c in cat}})
    prep = ColumnTransformer([("num", SimpleImputer(), num),
                              ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                     unknown_value=-1), cat)])
    old = Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(max_iter=5))])
    old.fit(frame, rng.random(30))
    joblib.dump({"pipeline": old, "columns": num + cat}, tmp_path / "old.joblib")

    with pytest.raises(ModelNotAvailableError, match="columns"):
        SklearnPredictor(artifacts_dir=str(tmp_path)).predict_log10(str(tmp_path / "old.joblib"),
                                                                    features())
