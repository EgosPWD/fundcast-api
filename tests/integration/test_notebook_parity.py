"""The prediction path reproduces the notebook estimates (section 5 load test).

Each reference row is predicted as a new round: its history is the company's earlier
rounds, and the active model's `lead_freq_map` without the row itself.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.adapters.outbound.ml import cleaning, features
from app.adapters.outbound.ml.sklearn_predictor import SklearnPredictor
from app.domain.entities import CompanyHistory, FundingRoundInput

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "global_startup_funding_sample.csv"
ARTIFACTS = ROOT / "artifacts"
BASELINE = ARTIFACTS / "modelo_rondas.joblib"

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="original Kaggle CSV is not in data/")

# CSV row index -> notebook estimate in USD (rounded to thousands).
NOTEBOOK_ESTIMATES = {18181: 616_000, 17937: 52_608_000, 7512: 52_704_000, 4350: 92_685_000}


def _value(v):
    return None if pd.isna(v) else v


@pytest.fixture(scope="module")
def prepared():
    return features.add_features(cleaning.clean(pd.read_csv(DATASET)))


@pytest.fixture(scope="module")
def predictor():
    return SklearnPredictor(str(ARTIFACTS))


@pytest.mark.parametrize("index", NOTEBOOK_ESTIMATES)
def test_prediction_matches_the_notebook(prepared, predictor, index):
    row = prepared.loc[index]
    round_input = FundingRoundInput(
        year=int(row.year), founded_year=int(row.founded_year), stage=row.stage,
        sector=row.sector, country=row.country, city=_value(row.city),
        investor_count=int(row.investor_count), lead_investor=_value(row.lead_investor),
        news_mentions=int(row.news_mentions), news_source=_value(row.news_source),
        company_domain=row.company_domain,
    )
    company = prepared[prepared.company_domain == row.company_domain]
    earlier = company[company.announced_date < row.announced_date].sort_values("announced_date")
    # from_rounds reads only `amount_usd`.
    history = CompanyHistory.from_rounds(
        [SimpleNamespace(amount_usd=_value(a)) for a in earlier.amount_usd])
    lead_freq_map = dict(predictor.lead_frequency_map(str(BASELINE)))
    if round_input.lead_investor is not None:
        lead_freq_map[round_input.lead_investor] -= 1

    log10 = predictor.predict_log10(str(BASELINE), round_input.to_features(history, lead_freq_map))

    assert 10 ** log10 == pytest.approx(NOTEBOOK_ESTIMATES[index], rel=1e-3)
