import math
from datetime import date

import pytest

from app.domain.entities import (
    MODEL_FEATURE_COLUMNS,
    REQUIRED_TRAINING_COLUMNS,
    STAGE_RANK,
    CompanyHistory,
    Estimate,
    FundingRound,
    FundingRoundInput,
    ModelMetrics,
    should_activate,
)
from app.domain.errors import InvalidInputError

LEAD_FREQ = {"Y Combinator": 41, "Sequoia": 7}


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


def features(round_input=None, history=None, lead_freq_map=LEAD_FREQ):
    return (round_input or make_input()).to_features(history=history,
                                                     lead_freq_map=lead_freq_map)


def stored_round(month: int, amount: float | None, year: int = 2023) -> FundingRound:
    return FundingRound(announced_date=date(year, month, 1), year=year, company_domain="acme.io",
                        founded_year=2020, stage="Seed", amount_status="verified",
                        amount_usd=amount)


class TestFundingRoundInputToFeatures:
    def test_full_input_derives_the_14_model_columns_in_contract_order(self):
        round_input = make_input(
            city="New York", investor_count=3, lead_investor="Y Combinator", news_mentions=2,
            news_source="TechCrunch", prev_rounds=1, prev_amount_usd=1_500_000,
        )

        result = features(round_input)

        assert result == {
            "year": 2025,
            "company_age": 4,
            "investor_count": 3,
            "news_mentions": 2,
            "n_prev_rounds": 1,
            "prev_log_amount": pytest.approx(math.log10(1_500_000)),
            "lead_freq": 42,
            "stage_rank": 2.0,
            "stage": "Seed",
            "sector": "Fintech",
            "country": "United States",
            "city": "New York",
            "lead_investor": "Y Combinator",
            "news_source": "TechCrunch",
        }
        assert tuple(result) == MODEL_FEATURE_COLUMNS
        assert len(MODEL_FEATURE_COLUMNS) == 14
        assert "has_lead_investor" not in MODEL_FEATURE_COLUMNS

    def test_minimal_input_defaults(self):
        result = features()

        assert result["n_prev_rounds"] == 0
        assert result["prev_log_amount"] is None
        assert result["lead_freq"] is None
        assert result["city"] is None and result["news_source"] is None
        assert result["investor_count"] is None and result["news_mentions"] is None

    def test_unknown_lead_investor_counts_its_own_round(self):
        assert features(make_input(lead_investor="Nobody Capital"))["lead_freq"] == 1

    def test_company_history_overrides_client_values(self):
        history = CompanyHistory(n_prev_rounds=3, prev_log_amount=6.5)
        round_input = make_input(prev_rounds=9, prev_amount_usd=10)

        result = features(round_input, history=history)

        assert (result["n_prev_rounds"], result["prev_log_amount"]) == (3, 6.5)

    @pytest.mark.parametrize("stage", ["Undisclosed", "Debt", "Bridge", "Unknown stage"])
    def test_stage_without_rank_maps_to_null(self, stage):
        assert features(make_input(stage=stage))["stage_rank"] is None

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
        assert features(make_input(stage=stage))["stage_rank"] == float(rank)

    @pytest.mark.parametrize("amount", [0, -10])
    def test_non_positive_previous_amount_is_rejected(self, amount):
        with pytest.raises(InvalidInputError, match="prev_amount_usd"):
            make_input(prev_amount_usd=amount)

    def test_negative_previous_rounds_are_rejected(self):
        with pytest.raises(InvalidInputError, match="prev_rounds"):
            make_input(prev_rounds=-1)

    @pytest.mark.parametrize("field", ["stage", "sector", "country"])
    def test_blank_required_text_is_rejected(self, field):
        with pytest.raises(InvalidInputError, match=field):
            make_input(**{field: "   "})

    def test_text_fields_are_stripped_and_blank_optionals_become_null(self):
        round_input = make_input(country="  Chile ", lead_investor=" Sequoia ",
                                 company_domain="  ", news_source=" ")

        result = features(round_input)

        assert result["country"] == "Chile"
        assert (result["lead_investor"], result["lead_freq"]) == ("Sequoia", 8)
        assert round_input.company_domain is None and result["news_source"] is None


class TestCompanyHistory:
    def test_counts_rounds_and_takes_last_positive_amount_by_date(self):
        rounds = [stored_round(1, 1_000_000), stored_round(5, 10_000_000), stored_round(9, None)]

        history = CompanyHistory.from_rounds(rounds)

        assert history.n_prev_rounds == 3
        assert history.prev_log_amount == pytest.approx(7.0)

    def test_without_any_positive_amount_prev_log_amount_is_null(self):
        history = CompanyHistory.from_rounds([stored_round(1, None), stored_round(2, 0)])

        assert (history.n_prev_rounds, history.prev_log_amount) == (2, None)


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
