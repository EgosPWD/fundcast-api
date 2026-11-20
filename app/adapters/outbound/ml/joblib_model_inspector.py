"""Reads the stored test metrics and params of a project model artifact."""

from collections.abc import Mapping
from pathlib import Path

import joblib

from app.application.ports.model_inspector import ArtifactMetadata
from app.domain.entities import ModelMetrics
from app.domain.errors import ModelNotAvailableError

# Artifact metric key (notebook naming) -> ModelMetrics field.
METRIC_KEYS = {"MAE_log": "mae_log", "R2": "r2", "RMSE_log": "rmse_log", "MdAPE": "mdape",
               "cobertura": "coverage"}


class JoblibModelInspector:
    """Implements the ModelArtifactInspector port.

    Only artifacts inside `artifacts_dir` are deserialized: joblib.load executes pickle
    code, so it must never touch user-supplied files (AGENTS.md section 8).
    """

    def __init__(self, artifacts_dir: str) -> None:
        self._artifacts_dir = Path(artifacts_dir).resolve()

    def read_metadata(self, artifact_path: str) -> ArtifactMetadata:
        bundle = joblib.load(self._resolve_inside_artifacts_dir(artifact_path))
        scores = bundle.get("metricas_test") if isinstance(bundle, dict) else None
        if not isinstance(scores, Mapping) or not METRIC_KEYS.keys() <= scores.keys():
            raise ModelNotAvailableError("Model artifact has no test metrics")
        params = bundle.get("parametros") or {}
        if not isinstance(params, Mapping):
            raise ModelNotAvailableError("Model artifact has invalid parametros")
        metrics = ModelMetrics(**{field: float(scores[key]) for key, field in METRIC_KEYS.items()})
        return ArtifactMetadata(metrics=metrics, params=dict(params))

    def _resolve_inside_artifacts_dir(self, artifact_path: str) -> Path:
        path = Path(artifact_path).resolve()
        if not path.is_relative_to(self._artifacts_dir):
            raise ModelNotAvailableError("Model artifact is outside the artifacts directory")
        if not path.is_file():
            raise ModelNotAvailableError("Model artifact not found")
        return path
