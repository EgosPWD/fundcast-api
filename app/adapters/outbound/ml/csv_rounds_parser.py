"""Training CSV parser (AGENTS.md sections 2, 6 and 13).

The file is read as plain data with pandas: nothing in it is executed and it is never
deserialized with joblib/pickle. Values are kept as written (whitespace and case
included): normalization belongs to `cleaning`, at training time, as in the notebook.
"""

import csv
import math
from typing import Any

import pandas as pd

from app.adapters.outbound.ml import cleaning
from app.application.ports.training_data_parser import ParsedRounds
from app.domain.entities import REQUIRED_TRAINING_COLUMNS, FundingRound
from app.domain.errors import InvalidTrainingFileError

LEAKING_COLUMN = "news_excerpt"
INT_COLUMNS = ("year", "founded_year", "investor_count", "news_mentions")
FLOAT_COLUMNS = ("amount_usd",)
TEXT_COLUMNS = (
    "round_id", "company_name", "company_domain", "country", "city", "sector", "stage",
    "amount_status", "lead_investor", "news_source", "news_url", "fliar_profile_url",
)
# Domain fields without a default: a row missing one cannot be stored.
REQUIRED_TEXT_VALUES = ("company_domain", "stage", "amount_status")


def _read(path: str) -> pd.DataFrame:
    try:
        # Everything as text first: ids such as "007" keep their zeros, and the numeric
        # columns are converted below with explicit errors.
        return pd.read_csv(path, dtype=str, encoding="utf-8")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError,
            csv.Error) as exc:
        raise InvalidTrainingFileError(f"file is not a readable CSV: {exc}") from exc


def _check_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_TRAINING_COLUMNS - set(df.columns))
    if missing:
        raise InvalidTrainingFileError(f"missing required columns: {', '.join(missing)}")


def _to_numbers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (*INT_COLUMNS, *FLOAT_COLUMNS):
        if col not in out:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        bad = values.isna() & out[col].notna()
        if col in INT_COLUMNS:
            bad |= values.notna() & (values % 1 != 0)
        if bad.any():
            raise InvalidTrainingFileError(f"column {col} has {int(bad.sum())} non-numeric "
                                           f"or non-integer value(s)")
        out[col] = values.astype(float)
    return out


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        out = cleaning.parse_announced_date(df)
    except (ValueError, TypeError) as exc:
        raise InvalidTrainingFileError(f"announced_date has an unreadable date: {exc}") from exc
    cleaning.check_year_matches_date(out)
    return out


def _value(raw: Any, as_int: bool = False) -> Any:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    return int(raw) if as_int else raw


def _to_round(row: dict[str, Any]) -> FundingRound:
    fields: dict[str, Any] = {col: _value(row.get(col)) for col in TEXT_COLUMNS}
    fields.update({col: _value(row.get(col), as_int=True) for col in INT_COLUMNS})
    fields["amount_usd"] = _value(row.get("amount_usd"))
    return FundingRound(announced_date=row["announced_date"].date(), **fields)


class CsvRoundsParser:
    """Implements the TrainingDataParser port."""

    def parse(self, path: str) -> ParsedRounds:
        df = _read(path)
        _check_columns(df)
        df = df.drop(columns=[LEAKING_COLUMN], errors="ignore")
        df = _parse_dates(_to_numbers(df))

        rejected = (
            df["founded_year"].isna()
            | (df["year"] - df["founded_year"] < 0)
            | df[list(REQUIRED_TEXT_VALUES)].isna().any(axis=1)
        )
        kept = df.loc[~rejected]
        rounds = [_to_round(row) for row in kept.to_dict(orient="records")]
        return ParsedRounds(rounds=rounds, rejected=int(rejected.sum()))
