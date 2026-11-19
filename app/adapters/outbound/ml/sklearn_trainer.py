"""Training adapter that reproduces the notebook pipeline (AGENTS.md section 4)."""

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from app.adapters.outbound.ml import cleaning, features
from app.application.ports.trainer import TrainingResult
from app.domain.entities import STAGE_RANK, FundingRound, ModelMetrics
from app.domain.errors import InsufficientTrainingDataError

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_STRATA = 10
MIN_ROWS_PER_STRATUM = 2
CRITERIA = {"r2_min": 0.55, "mae_log_max": 0.4}
APE_COVERAGE = 0.5

# FundingRound fields that are CSV columns; news_excerpt is never stored nor used.
_ROUND_COLUMNS = [
    "round_id", "announced_date", "year", "company_name", "company_domain", "country",
    "city", "sector", "founded_year", "stage", "amount_usd", "amount_status",
    "lead_investor", "investor_count", "news_mentions", "news_source", "news_url",
    "fliar_profile_url",
]


def stratified_split(X: pd.DataFrame, y: pd.Series) -> tuple:
    """80/20 split stratified by deciles of y; deciles with equal edges are merged.

    Raises InsufficientTrainingDataError instead of letting sklearn/pandas fail.
    """
    if len(y) < N_STRATA:
        raise InsufficientTrainingDataError(
            f"Not enough usable rows to train: {len(y)} (at least {N_STRATA} needed)"
        )
    strata = pd.qcut(y, q=N_STRATA, labels=False, duplicates="drop")
    smallest = int(strata.value_counts().min())
    if smallest < MIN_ROWS_PER_STRATUM:
        raise InsufficientTrainingDataError(
            f"Not enough usable rows for the stratified split: a decile of the amount "
            f"has {smallest} row(s) (at least {MIN_ROWS_PER_STRATUM} needed)"
        )
    try:
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                stratify=strata)
    except ValueError as exc:
        raise InsufficientTrainingDataError(
            f"Not enough usable rows for the stratified split: {exc}"
        ) from exc


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    """Steps 8-9: every transformer lives inside the Pipeline."""
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("target_enc", TargetEncoder(random_state=RANDOM_STATE)),
    ])
    prep = ColumnTransformer(
        [("num", num, features.COLS_NUM), ("cat", cat, features.COLS_CAT)],
        remainder="drop",
    )
    pipeline = Pipeline([
        ("prep", prep),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])
    return pipeline.set_params(**params)


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Step 10: metrics on log10 and on USD (APE as a fraction)."""
    real, est = 10 ** y_true.to_numpy(), 10 ** y_pred
    ape = np.abs(est - real) / real
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE_log": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MdAPE": float(np.median(ape)),
        "cobertura": float(np.mean(ape <= APE_COVERAGE)),
        "pct_sobreestima": float(np.mean(est > real)),
    }


class SklearnTrainer:
    """Cleans the rounds, fits the winning pipeline on a new stratified split and
    saves an artifact with the same keys as the notebook baseline."""

    WINNING_PARAMS: dict[str, Any] = {
        "model__learning_rate": 0.03,
        "model__max_iter": 400,
        "model__max_leaf_nodes": 31,
        "model__min_samples_leaf": 40,
    }

    def train(
        self, rounds: Sequence[FundingRound], params: dict[str, Any], artifact_path: str
    ) -> TrainingResult:
        """Train from domain rounds, kept in the given order (ordered by id)."""
        rows = [{col: getattr(r, col) for col in _ROUND_COLUMNS} for r in rounds]
        df = pd.DataFrame(rows, columns=_ROUND_COLUMNS)
        df["amount_usd"] = df["amount_usd"].astype(float)
        return self.train_from_frame(df, params, artifact_path)

    def train_from_frame(
        self, df: pd.DataFrame, params: dict[str, Any], artifact_path: str
    ) -> TrainingResult:
        """Train from a frame with the CSV columns, rows in CSV (id) order."""
        prepared = features.add_features(cleaning.clean(df))
        usable = features.usable_rows(prepared)
        X, y = usable[features.FEATURE_COLUMNS], usable[features.TARGET]
        X_train, X_test, y_train, y_test = stratified_split(X, y)

        pipeline = build_pipeline(params).fit(X_train, y_train)
        scores = evaluate(y_test, pipeline.predict(X_test))
        metrics = ModelMetrics(
            mae_log=scores["MAE_log"], r2=scores["R2"], rmse_log=scores["RMSE_log"],
            mdape=scores["MdAPE"], coverage=scores["cobertura"],
        )
        self._save(pipeline, scores, params, prepared, artifact_path)
        return TrainingResult(artifact_path=artifact_path, metrics=metrics,
                              n_train=len(X_train), n_test=len(X_test), params=dict(params))

    @staticmethod
    def _save(pipeline: Pipeline, scores: dict[str, float], params: dict[str, Any],
              prepared: pd.DataFrame, artifact_path: str) -> None:
        meets = scores["R2"] >= CRITERIA["r2_min"] and scores["MAE_log"] <= CRITERIA["mae_log_max"]
        lead_counts = prepared["lead_investor"].value_counts()
        artifact = {
            "pipeline": pipeline,
            "columns": list(features.FEATURE_COLUMNS),
            "criterios": dict(CRITERIA),
            "metricas_test": {**scores, "cumple": bool(meets)},
            "parametros": dict(params),
            "lead_freq_map": {str(k): int(v) for k, v in lead_counts.items()},
            "stage_rank_map": dict(STAGE_RANK),
        }
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
