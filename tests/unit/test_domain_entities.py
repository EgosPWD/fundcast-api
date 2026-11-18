import math

import pytest

from app.domain.entities import (
    MODEL_FEATURE_COLUMNS,
    REQUIRED_TRAINING_COLUMNS,
    STAGE_RANK,
    Estimate,
    FundingRoundInput,
    ModelMetrics,
    should_activate,
)
from app.domain.errors import InvalidInputError


def make_input(**overrides) -> FundingRoundInput:
    data = {
        "year": 2025,
        "founded_year": 2021,
        "stage": "Seed",
        "sector": "Fintech",
        "country": "United States",
    }
    data.update(overrides)
    return FundingRoundInput(**data)


class TestFundingRoundInputToFeatures:
    def test_full_input_derives_every_model_column(self):
        round_input = make_input(
            city="New York",
            investor_count=3,
            lead_investor="Y Combinator",
            news_mentions=2,
            news_source="TechCrunch",
            prev_amount_usd=1_500_000,
        )

        features = round_input.to_features()

        assert features == {
            "year": 2025,
            "company_age": 4,
            "investor_count": 3,
            "news_mentions": 2,
            "has_lead_investor": 1,
            "prev_log_amount": pytest.approx(math.log10(1_500_000)),
            "stage_rank": 2.0,
            "stage": "Seed",
            "sector": "Fintech",
            "country": "United States",
            "city": "New York",
            "lead_investor": "Y Combinator",
            "news_source": "TechCrunch",
        }

    def test_features_have_exactly_the_contract_columns_in_order(self):
        assert tuple(make_input().to_features()) == MODEL_FEATURE_COLUMNS
        assert len(MODEL_FEATURE_COLUMNS) == 13

    def test_minimal_input_keeps_optional_fields_null(self):
        features = make_input().to_features()

        assert features["company_age"] == 4
        assert features["has_lead_investor"] == 0
        assert features["prev_log_amount"] is None
        assert features["city"] is None
        assert features["lead_investor"] is None
        assert features["news_source"] is None
        assert features["investor_count"] is None
        assert features["news_mentions"] is None

    @pytest.mark.parametrize("stage", ["Undisclosed", "Debt", "Bridge", "Unknown stage"])
    def test_stage_without_rank_maps_to_null(self, stage):
        assert make_input(stage=stage).to_features()["stage_rank"] is None

    @pytest.mark.parametrize(
        ("stage", "rank"),
        [
            ("Grant", 0), ("Angel", 0), ("Crowdfunding", 0), ("Pre-seed", 1),
            ("Seed", 2), ("Pre-Series A", 3), ("Series A", 4), ("Pre-Series B", 5),
            ("Series B", 6), ("Pre-Series C", 7), ("Series C", 8), ("Series D", 10),
            ("Series E", 11), ("Series F", 12), ("Series G", 13), ("Series H", 14),
        ],
    )
    def test_stage_rank_map_matches_contract(self, stage, rank):
        assert STAGE_RANK[stage] == rank
        assert make_input(stage=stage).to_features()["stage_rank"] == float(rank)

    @pytest.mark.parametrize("amount", [0, -10])
    def test_non_positive_previous_amount_is_rejected(self, amount):
        with pytest.raises(InvalidInputError, match="prev_amount_usd"):
            make_input(prev_amount_usd=amount)

    @pytest.mark.parametrize("field", ["stage", "sector", "country"])
    def test_blank_required_text_is_rejected(self, field):
        with pytest.raises(InvalidInputError, match=field):
            make_input(**{field: "   "})

    def test_text_fields_are_stripped(self):
        features = make_input(country="  Chile ", lead_investor=" YC ").to_features()

        assert features["country"] == "Chile"
        assert features["lead_investor"] == "YC"
        assert features["has_lead_investor"] == 1

    def test_blank_lead_investor_counts_as_missing(self):
        features = make_input(lead_investor="  ").to_features()

        assert features["lead_investor"] is None
        assert features["has_lead_investor"] == 0


class TestEstimate:
    def test_converts_log10_to_usd_with_range_factor_1_5(self):
        estimate = Estimate.from_log10(6.0, stage="Seed")

        assert estimate.estimated_log10 == 6.0
        assert estimate.estimated_amount_usd == pytest.approx(1_000_000)
        assert estimate.range_usd[0] == pytest.approx(1_000_000 / 1.5)
        assert estimate.range_usd[1] == pytest.approx(1_500_000)
        assert estimate.warning is None

    def test_undisclosed_stage_carries_a_warning(self):
        estimate = Estimate.from_log10(6.0, stage="Undisclosed")

        assert estimate.warning is not None
        assert "Undisclosed" in estimate.warning


class TestShouldActivate:
    def metrics(self, mae_log: float) -> ModelMetrics:
        return ModelMetrics(mae_log=mae_log, r2=0.6, rmse_log=0.5, mdape=0.5, coverage=0.48)

    def test_activates_when_there_is_no_active_model(self):
        assert should_activate(self.metrics(0.9), active=None) is True

    @pytest.mark.parametrize(
        ("new_mae", "expected"),
        [(0.30, True), (0.352, True), (0.362, True), (0.3621, False), (0.40, False)],
    )
    def test_activates_only_within_tolerance_of_active_mae(self, new_mae, expected):
        assert should_activate(self.metrics(new_mae), active=self.metrics(0.352)) is expected


def test_required_training_columns_exclude_leaky_and_optional_columns():
    assert "news_excerpt" not in REQUIRED_TRAINING_COLUMNS
    assert "company_domain" in REQUIRED_TRAINING_COLUMNS
    assert len(REQUIRED_TRAINING_COLUMNS) == 14
