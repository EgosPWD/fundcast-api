"""Training from domain rounds (the funding_rounds path) reproduces the notebook.

The original CSV goes through CsvRoundsParser, exactly as SeedFundingRounds loads it,
and then through SklearnTrainer.train, as RetrainModel calls it (AGENTS.md section 4).
"""

from pathlib import Path

import pytest

from app.adapters.outbound.ml.csv_rounds_parser import CsvRoundsParser
from app.adapters.outbound.ml.sklearn_trainer import SklearnTrainer

DATASET = Path(__file__).resolve().parents[2] / "data" / "global_startup_funding_sample.csv"

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="original Kaggle CSV is not in data/")


@pytest.fixture(scope="module")
def parsed():
    return CsvRoundsParser().parse(str(DATASET))


def test_parser_keeps_every_row_of_the_original_csv(parsed):
    assert parsed.rejected == 0
    assert len(parsed.rounds) == 20_227
    assert len({r.round_id for r in parsed.rounds}) == 20_227


def test_training_from_parsed_rounds_reproduces_the_notebook_metrics(parsed, tmp_path):
    artifact = tmp_path / "model_v2.joblib"

    result = SklearnTrainer().train(parsed.rounds, SklearnTrainer.WINNING_PARAMS, str(artifact))

    assert (result.n_train, result.n_test) == (12758, 3190)
    assert artifact.is_file()
    m = result.metrics
    assert m.r2 == pytest.approx(0.6153, abs=5e-5)
    assert m.mae_log == pytest.approx(0.3504, abs=5e-5)
    assert m.rmse_log == pytest.approx(0.4948, abs=5e-5)
    assert m.mdape == pytest.approx(0.5294, abs=5e-5)
    assert m.coverage == pytest.approx(0.4727, abs=5e-5)
