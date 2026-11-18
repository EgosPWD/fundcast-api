"""Predictor adapter backed by a scikit-learn Pipeline stored with joblib."""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.domain.errors import ModelNotAvailableError

REQUIRED_BUNDLE_KEYS = frozenset({"pipeline", "columns"})


@dataclass(frozen=True, slots=True)
class _LoadedModel:
    pipeline: Any
    columns: list[str]
    numeric_columns: list[str]


class SklearnPredictor:
    """Implements the Predictor port.

    Only artifacts inside `artifacts_dir` are deserialized: joblib.load executes
    pickle code, so it must never touch user-supplied files (AGENTS.md section 8).
    """

    def __init__(self, artifacts_dir: str) -> None:
        self._artifacts_dir = Path(artifacts_dir).resolve()
        self._cache: dict[Path, _LoadedModel] = {}
        self._lock = threading.Lock()

    def predict_log10(self, artifact_path: str, features: dict[str, Any]) -> float:
        model = self._get(artifact_path)
        frame = pd.DataFrame([[features.get(c) for c in model.columns]], columns=model.columns)
        # Nulls must be NaN, as read_csv produced them at training time:
        # SimpleImputer does not treat None as missing and the prediction silently changes.
        frame = frame.astype(object).replace({None: np.nan})
        frame = frame.astype({c: "float64" for c in model.numeric_columns})
        return float(model.pipeline.predict(frame)[0])

    def _get(self, artifact_path: str) -> _LoadedModel:
        path = self._resolve_inside_artifacts_dir(artifact_path)
        with self._lock:
            if path not in self._cache:
                self._cache[path] = self._load(path)
            return self._cache[path]

    def _resolve_inside_artifacts_dir(self, artifact_path: str) -> Path:
        path = Path(artifact_path).resolve()
        if not path.is_relative_to(self._artifacts_dir):
            raise ModelNotAvailableError("Model artifact is outside the artifacts directory")
        if not path.is_file():
            raise ModelNotAvailableError("Model artifact not found")
        return path

    @staticmethod
    def _load(path: Path) -> _LoadedModel:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or not REQUIRED_BUNDLE_KEYS <= bundle.keys():
            raise ModelNotAvailableError("Model artifact has an invalid format")
        pipeline, columns = bundle["pipeline"], list(bundle["columns"])
        if not hasattr(pipeline, "predict"):
            raise ModelNotAvailableError("Model artifact has no usable pipeline")
        return _LoadedModel(pipeline, columns, _numeric_columns(pipeline, columns))


def _numeric_columns(pipeline: Any, columns: list[str]) -> list[str]:
    """Columns routed to the numeric branch ('num') of the ColumnTransformer."""
    try:
        transformers = pipeline.named_steps["prep"].transformers_
    except (AttributeError, KeyError) as exc:
        raise ModelNotAvailableError("Model artifact has no 'prep' ColumnTransformer") from exc
    numeric = [c for name, _, cols in transformers if name == "num" for c in cols]
    if not set(numeric) <= set(columns):
        raise ModelNotAvailableError("Model artifact columns are inconsistent")
    return numeric
