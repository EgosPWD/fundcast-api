"""JoblibModelInspector reads the stored metrics of project artifacts only."""

from pathlib import Path

import joblib
import pytest

from app.adapters.outbound.ml.joblib_model_inspector import JoblibModelInspector
from app.application.ports.model_inspector import ArtifactMetadata
from app.domain.entities import ModelMetrics
from app.domain.errors import ModelNotAvailableError

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "artifacts" / "modelo_rondas.joblib"


def test_reads_the_baseline_metrics_and_params():
    metadata = JoblibModelInspector(str(BASELINE.parent)).read_metadata(str(BASELINE))

    assert isinstance(metadata, ArtifactMetadata)
    m = metadata.metrics
    assert m.r2 == pytest.approx(0.6153, abs=5e-5)
    assert m.mae_log == pytest.approx(0.3504, abs=5e-5)
    assert m.rmse_log == pytest.approx(0.4948, abs=5e-5)
    assert m.mdape == pytest.approx(0.5294, abs=5e-5)
    assert m.coverage == pytest.approx(0.4727, abs=5e-5)
    assert metadata.params == {"model__learning_rate": 0.03, "model__max_iter": 400,
                               "model__max_leaf_nodes": 31, "model__min_samples_leaf": 40}


def test_maps_every_metric_by_name(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"metricas_test": {"R2": 0.1, "MAE_log": 0.2, "RMSE_log": 0.3, "MdAPE": 0.4,
                                   "cobertura": 0.5, "cumple": False},
                 "parametros": {"p": 1}}, path)

    metadata = JoblibModelInspector(str(tmp_path)).read_metadata(str(path))

    assert metadata == ArtifactMetadata(
        metrics=ModelMetrics(mae_log=0.2, r2=0.1, rmse_log=0.3, mdape=0.4, coverage=0.5),
        params={"p": 1})


def test_refuses_artifacts_outside_the_artifacts_dir(tmp_path):
    inside, outside = tmp_path / "artifacts", tmp_path / "uploads"
    inside.mkdir()
    outside.mkdir()
    joblib.dump({"metricas_test": {}}, outside / "evil.joblib")

    with pytest.raises(ModelNotAvailableError, match="outside"):
        JoblibModelInspector(str(inside)).read_metadata(str(inside / ".." / "uploads" /
                                                             "evil.joblib"))


def test_missing_artifact_is_not_available(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="not found"):
        JoblibModelInspector(str(tmp_path)).read_metadata(str(tmp_path / "nope.joblib"))


@pytest.mark.parametrize("bundle", [["not", "a", "dict"], {"parametros": {}},
                                    {"metricas_test": {"R2": 0.1}, "parametros": {}}])
def test_artifact_without_metrics_has_an_invalid_format(tmp_path, bundle):
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)

    with pytest.raises(ModelNotAvailableError, match="metrics"):
        JoblibModelInspector(str(tmp_path)).read_metadata(str(path))
