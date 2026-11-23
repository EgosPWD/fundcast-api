"""Cleaning and feature rules of AGENTS.md section 4 on small hand-made frames."""

import math

import numpy as np
import pandas as pd
import pytest

from app.adapters.outbound.ml import cleaning, features
from app.adapters.outbound.ml.sklearn_trainer import SklearnTrainer, stratified_split
from app.domain.errors import InsufficientTrainingDataError, InvalidTrainingFileError


def frame(rows: list[dict]) -> pd.DataFrame:
    """Build a frame with every CSV column the pipeline needs, filling defaults."""
    defaults = {
        "round_id": None, "announced_date": "2024-01-01", "year": 2024,
        "company_name": "Acme", "company_domain": "acme.io", "country": "Chile",
        "city": "Santiago", "sector": "Fintech", "founded_year": 2020, "stage": "Seed",
        "amount_usd": 1_000_000.0, "amount_status": "verified", "lead_investor": "YC",
        "investor_count": 2, "news_mentions": 1, "news_source": "TechCrunch",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


# --- cleaning -----------------------------------------------------------------


def test_text_is_stripped_and_spaces_collapsed_without_changing_case():
    df = frame([{"country": "  United   States ", "company_name": "ACME  inc",
                 "lead_investor": " Y\tCombinator", "stage": "Series A ",
                 "sector": "Fin  Tech", "city": " New  York"}])

    out = cleaning.normalize_text(df)

    row = out.iloc[0]
    assert row["country"] == "United States"
    assert row["company_name"] == "ACME inc"
    assert row["lead_investor"] == "Y Combinator"
    assert row["stage"] == "Series A"
    assert row["sector"] == "Fin Tech"
    assert row["city"] == "New York"


def test_text_normalization_keeps_nulls():
    out = cleaning.normalize_text(frame([{"lead_investor": None, "city": np.nan}]))

    assert pd.isna(out.loc[0, "lead_investor"])
    assert pd.isna(out.loc[0, "city"])


def test_city_variants_take_the_most_frequent_spelling_of_their_key():
    df = frame([{"city": "New York"}, {"city": "New York"}, {"city": "new-york"},
                {"city": "NEW YORK"}, {"city": None}])

    out = cleaning.unify_city(df)

    assert out["city"].iloc[:4].tolist() == ["New York"] * 4
    assert pd.isna(out["city"].iloc[4])


def test_city_mode_tie_takes_the_alphabetically_first_spelling():
    df = frame([{"city": "sao paulo"}, {"city": "Sao Paulo"}])

    out = cleaning.unify_city(df)

    assert out["city"].tolist() == ["Sao Paulo", "Sao Paulo"]


def test_announced_date_is_parsed_to_datetime():
    out = cleaning.parse_announced_date(frame([{"announced_date": "2024-03-05"}]))

    assert out.loc[0, "announced_date"] == pd.Timestamp("2024-03-05")


def test_year_that_does_not_match_announced_date_raises():
    df = cleaning.parse_announced_date(
        frame([{"announced_date": "2024-03-05", "year": 2024},
               {"announced_date": "2023-12-31", "year": 2024}])
    )

    with pytest.raises(InvalidTrainingFileError, match="year"):
        cleaning.check_year_matches_date(df)


def test_suspect_low_flag_marks_amounts_below_1000():
    df = frame([{"amount_usd": 999.0}, {"amount_usd": 1000.0}, {"amount_usd": None}])

    out = cleaning.add_quality_flags(df)

    assert out["amount_suspect_low"].tolist() == [True, False, False]


def test_outlier_high_flag_uses_iqr_on_log10_of_positive_amounts():
    amounts = [1e6, 1e6, 1e6, 1e6, 1e7, 1e7, 1e7, 1e7, 1e12, 0.0]
    df = frame([{"amount_usd": a} for a in amounts])

    out = cleaning.add_quality_flags(df)

    assert out["amount_outlier_high"].tolist() == [False] * 8 + [True, False]


def test_dup_same_amount_flags_repeated_domain_stage_amount_except_undisclosed():
    df = frame([
        {"stage": "Seed", "amount_usd": 5e5}, {"stage": "Seed", "amount_usd": 5e5},
        {"stage": "Undisclosed", "amount_usd": 7e5}, {"stage": "Undisclosed", "amount_usd": 7e5},
        {"stage": "Series A", "amount_usd": None}, {"stage": "Series A", "amount_usd": None},
        {"stage": "Series B", "amount_usd": 9e6},
    ])

    out = cleaning.add_quality_flags(df)

    assert out["dup_same_amount"].tolist() == [True, True, False, False, False, False, False]


def test_clean_applies_every_step_and_does_not_drop_rows():
    df = frame([{"city": " santiago", "announced_date": "2024-02-02"},
                {"city": "Santiago"}, {"city": "Santiago"}, {"amount_usd": 10.0}])

    out = cleaning.clean(df)

    assert len(out) == 4
    assert out["city"].tolist() == ["Santiago"] * 4
    assert out["amount_suspect_low"].tolist() == [False, False, False, True]


# --- features -----------------------------------------------------------------


def history_frame() -> pd.DataFrame:
    return cleaning.clean(frame([
        {"company_domain": "b.io", "announced_date": "2021-05-01", "year": 2021,
         "amount_usd": 1e6},
        {"company_domain": "a.io", "announced_date": "2022-01-01", "year": 2022,
         "amount_usd": None, "amount_status": "not_disclosed"},
        {"company_domain": "a.io", "announced_date": "2020-01-01", "year": 2020,
         "amount_usd": 1e5},
        {"company_domain": "a.io", "announced_date": "2023-01-01", "year": 2023,
         "amount_usd": 1e7},
        {"company_domain": "b.io", "announced_date": "2019-05-01", "year": 2019,
         "amount_usd": 0.0},
    ]))


def test_prev_log_amount_uses_only_earlier_rounds_and_forward_fills_missing_amounts():
    out = features.add_features(history_frame())

    prev = out["prev_log_amount"]
    assert math.isnan(prev.iloc[2])  # a.io 2020: first round
    assert prev.iloc[1] == pytest.approx(5.0)  # a.io 2022: previous is 1e5
    assert prev.iloc[3] == pytest.approx(5.0)  # a.io 2023: 2022 has no amount, ffill 1e5
    assert math.isnan(prev.iloc[4])  # b.io 2019: first round
    assert math.isnan(prev.iloc[0])  # b.io 2021: previous amount is 0, no log


def test_n_prev_rounds_counts_earlier_rounds_of_the_same_company():
    out = features.add_features(history_frame())

    assert out["n_prev_rounds"].tolist() == [1, 1, 0, 2, 0]


def test_log_amount_only_for_positive_amounts_and_frame_order_is_kept():
    out = features.add_features(history_frame())

    assert out["log_amount"].iloc[0] == pytest.approx(6.0)
    assert out["log_amount"].iloc[[1, 4]].isna().all()
    assert out["year"].tolist() == [2021, 2022, 2020, 2023, 2019]


def test_lead_freq_counts_the_row_itself_and_is_nan_without_leader():
    df = cleaning.clean(frame([{"lead_investor": "YC"}, {"lead_investor": "YC"},
                               {"lead_investor": "Sequoia"}, {"lead_investor": None}]))

    out = features.add_features(df)

    assert out["lead_freq"].iloc[:3].tolist() == [2, 2, 1]
    assert math.isnan(out["lead_freq"].iloc[3])


def test_company_age_and_stage_rank_with_unknown_stage_as_nan():
    df = cleaning.clean(frame([{"stage": "Series A", "founded_year": 2019},
                               {"stage": "Undisclosed"}, {"stage": "Bridge"}]))

    out = features.add_features(df)

    assert out["company_age"].tolist() == [5, 4, 4]
    assert out["stage_rank"].iloc[0] == 4
    assert out["stage_rank"].iloc[1:].isna().all()


def test_usable_rows_keep_only_verified_amounts_of_at_least_1000():
    df = features.add_features(cleaning.clean(frame([
        {"amount_usd": 5e5}, {"amount_usd": 999.0}, {"amount_usd": None},
        {"amount_usd": 5e5, "amount_status": "failed_check"}, {"amount_usd": 1000.0},
    ])))

    out = features.usable_rows(df)

    assert out.index.tolist() == [0, 4]


def test_feature_columns_match_the_14_column_model_contract():
    assert features.FEATURE_COLUMNS == [
        "year", "company_age", "investor_count", "news_mentions", "n_prev_rounds",
        "prev_log_amount", "lead_freq", "stage_rank", "stage", "sector", "country",
        "city", "lead_investor", "news_source",
    ]


# --- split --------------------------------------------------------------------


def test_repeated_amounts_do_not_break_the_stratified_split():
    y = pd.Series([6.0] * 60 + list(np.linspace(5, 8, 40)))
    X = pd.DataFrame({"a": range(100)})

    X_train, X_test, y_train, y_test = stratified_split(X, y)

    assert (len(X_train), len(X_test)) == (80, 20)


@pytest.mark.parametrize("n_rows", [0, 1, 3, 9])
def test_too_few_rows_raise_insufficient_training_data(n_rows):
    y = pd.Series(np.linspace(5, 8, n_rows))
    X = pd.DataFrame({"a": range(n_rows)})

    with pytest.raises(InsufficientTrainingDataError):
        stratified_split(X, y)


def test_train_from_frame_with_too_few_usable_rows_raises_domain_error(tmp_path):
    df = frame([{"amount_usd": 1e6 + i} for i in range(5)])

    with pytest.raises(InsufficientTrainingDataError):
        SklearnTrainer().train_from_frame(df, SklearnTrainer.WINNING_PARAMS,
                                          str(tmp_path / "m.joblib"))

    assert not (tmp_path / "m.joblib").exists()


def test_city_unification_with_only_null_cities_keeps_them_null():
    out = cleaning.unify_city(frame([{"city": None}, {"city": np.nan}]))

    assert out["city"].isna().all()
