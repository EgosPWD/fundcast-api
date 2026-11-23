"""Row cleaning and quality flags (AGENTS.md section 4, steps 1-4).

Replicates the notebook literally: nothing here drops rows or fixes values beyond
whitespace and city spelling, so the training split stays reproducible.
"""

import numpy as np
import pandas as pd

from app.domain.errors import InvalidTrainingFileError

TEXT_COLUMNS = ("country", "city", "sector", "stage", "company_name", "lead_investor")
SUSPECT_LOW_USD = 1000


def _all_null(s: pd.Series) -> bool:
    """An all-null column has no string dtype, so `.str` cannot be used on it."""
    return bool(s.isna().all())


def parse_announced_date(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: parse `announced_date` into a datetime column."""
    out = df.copy()
    out["announced_date"] = pd.to_datetime(out["announced_date"])
    return out


def check_year_matches_date(df: pd.DataFrame) -> None:
    """Step 1: every `year` must equal the year of `announced_date` (never corrected)."""
    mismatched = df["announced_date"].dt.year != df["year"]
    if mismatched.any():
        raise InvalidTrainingFileError(
            f"{int(mismatched.sum())} row(s) have a year that does not match announced_date"
        )


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: strip and collapse inner whitespace, keeping case and nulls."""
    out = df.copy()
    for col in TEXT_COLUMNS:
        if col in out and not _all_null(out[col]):
            out[col] = out[col].str.strip().str.replace(r"\s+", " ", regex=True)
    return out


def unify_city(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: spellings sharing a key (lowercase, alphanumeric only) take the key's
    most frequent spelling; `mode()` sorts ties, so the alphabetically first wins."""
    out = df.copy()
    if _all_null(out["city"]):
        return out
    key = out["city"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    canonical = out["city"].groupby(key).agg(lambda s: s.mode()[0])
    out["city"] = key.map(canonical)
    return out


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: flags over all rows. Only `amount_suspect_low` excludes rows later."""
    out = df.copy()
    amount = out["amount_usd"]
    out["amount_suspect_low"] = amount < SUSPECT_LOW_USD

    log_positive = np.log10(amount[amount > 0])
    q1, q3 = log_positive.quantile([0.25, 0.75])
    out["amount_outlier_high"] = amount > 10 ** (q3 + 1.5 * (q3 - q1))

    duplicated = out.duplicated(["company_domain", "stage", "amount_usd"], keep=False)
    out["dup_same_amount"] = duplicated & amount.notna() & (out["stage"] != "Undisclosed")
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply steps 1-4 in order; the row order of `df` is preserved."""
    out = parse_announced_date(df)
    check_year_matches_date(out)
    return add_quality_flags(unify_city(normalize_text(out)))
