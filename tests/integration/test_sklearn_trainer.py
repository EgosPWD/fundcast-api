"""SklearnTrainer reproduces the notebook (AGENTS.md sections 1 and 4)."""

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.adapters.outbound.ml import cleaning, features
from app.adapters.outbound.ml.sklearn_trainer import SklearnTrainer, stratified_split
from app.application.ports.trainer import TrainingResult
from app.domain.entities import STAGE_RANK, FundingRound

DATASET = Path(__file__).resolve().parents[2] / "data" / "global_startup_funding_sample.csv"
ARTIFACT_KEYS = {"pipeline", "columns", "criterios", "metricas_test", "parametros",
                 "lead_freq_map", "stage_rank_map"}

requires_dataset = pytest.mark.skipif(not DATASET.exists(),
                                      reason="original Kaggle CSV is not in data/")


@pytest.fixture(scope="module")
def original_frame() -> pd.DataFrame:
    return pd.read_csv(DATASET)


@pytest.fixture(scope="module")
def trained(original_frame, tmp_path_factory) -> tuple[TrainingResult, Path]:
    path = tmp_path_factory.mktemp("artifacts") / "nested" / "model.joblib"
    result = SklearnTrainer().train_from_frame(
        original_frame, SklearnTrainer.WINNING_PARAMS, str(path)
    )
    return result, path


@requires_dataset
def test_training_on_the_original_csv_reproduces_the_notebook_metrics(trained):
    result, path = trained

    assert (result.n_train, result.n_test) == (12758, 3190)
    assert result.artifact_path == str(path)
    assert result.params == SklearnTrainer.WINNING_PARAMS
    m = result.metrics
    assert m.r2 == pytest.approx(0.6153, abs=5e-5)
    assert m.mae_log == pytest.approx(0.3504, abs=5e-5)
    assert m.rmse_log == pytest.approx(0.4948, abs=5e-5)
    assert m.mdape == pytest.approx(0.5294, abs=5e-5)
    assert m.coverage == pytest.approx(0.4727, abs=5e-5)


@requires_dataset
def test_saved_artifact_has_the_baseline_keys_and_metrics(trained):
    _, path = trained

    artifact = joblib.load(path)

    assert set(artifact) == ARTIFACT_KEYS
    assert artifact["columns"] == features.FEATURE_COLUMNS
    assert artifact["criterios"] == {"r2_min": 0.55, "mae_log_max": 0.4}
    assert artifact["parametros"] == SklearnTrainer.WINNING_PARAMS
    assert artifact["stage_rank_map"] == STAGE_RANK
    assert len(artifact["lead_freq_map"]) == 8463
    assert all(type(v) is int for v in artifact["lead_freq_map"].values())
    metrics = artifact["metricas_test"]
    assert metrics["pct_sobreestima"] == pytest.approx(0.5103, abs=5e-5)
    assert metrics["MdAPE"] == pytest.approx(0.5294, abs=5e-5)
    assert metrics["cumple"] is True


@requires_dataset
def test_saved_pipeline_predicts_the_held_out_rows(trained, original_frame):
    _, path = trained
    df = features.usable_rows(features.add_features(cleaning.clean(original_frame)))
    _, X_test, _, y_test = stratified_split(df[features.FEATURE_COLUMNS], df["log_amount"])

    pred = joblib.load(path)["pipeline"].predict(X_test)

    assert pred.shape == (3190,)
    mae = float(np.mean(np.abs(pred - y_test.to_numpy())))
    assert mae == pytest.approx(0.3504, abs=5e-5)


def small_rounds(n: int = 80) -> list[FundingRound]:
    rng = np.random.default_rng(0)
    stages = ["Seed", "Series A", "Series B", "Undisclosed"]
    rounds = []
    for i in range(n):
        stage = stages[i % 4]
        year = 2019 + i % 7
        rounds.append(FundingRound(
            announced_date=date(year, 1 + i % 12, 1), year=year,
            company_domain=f"c{i % 20}.io", founded_year=2015, stage=stage,
            amount_status="verified", round_id=f"r{i}", company_name=f"C{i % 20}",
            country="Chile" if i % 2 else "Peru", city="Santiago",
            sector="Fintech" if i % 3 else "Health",
            amount_usd=float(10 ** (5 + stages.index(stage) + rng.normal(0, 0.3))),
            lead_investor=None if i % 5 == 0 else f"Fund {i % 6}",
            investor_count=1 + i % 4, news_mentions=i % 3, news_source="TechCrunch",
        ))
    return rounds


def test_train_with_funding_round_entities_works_end_to_end(tmp_path):
    path = tmp_path / "model.joblib"

    result = SklearnTrainer().train(small_rounds(), SklearnTrainer.WINNING_PARAMS, str(path))

    assert (result.n_train, result.n_test) == (64, 16)
    assert path.exists()
    artifact = joblib.load(path)
    assert set(artifact) == ARTIFACT_KEYS
    assert artifact["lead_freq_map"]["Fund 1"] == 12
    assert "news_excerpt" not in artifact["columns"]
