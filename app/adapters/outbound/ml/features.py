"""Model features and usable rows (AGENTS.md section 4, steps 5-6)."""

import numpy as np
import pandas as pd

from app.domain.entities import STAGE_RANK

COLS_NUM = ["year", "company_age", "investor_count", "news_mentions", "n_prev_rounds",
            "prev_log_amount", "lead_freq", "stage_rank"]
COLS_CAT = ["stage", "sector", "country", "city", "lead_investor", "news_source"]
FEATURE_COLUMNS = COLS_NUM + COLS_CAT
TARGET = "log_amount"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model variables over ALL cleaned rows, before filtering `verified`.

    History is sorted by (company_domain, announced_date) with no round_id
    tiebreak, and `prev_log_amount` may come from a non-verified round: both
    replicate the notebook so the baseline metrics reproduce exactly.
    """
    out = df.copy()
    amount = out["amount_usd"]
    out[TARGET] = np.log10(amount.where(amount > 0))

    history = out.sort_values(["company_domain", "announced_date"]).groupby("company_domain")
    # transform/cumcount return values aligned by index, so the frame order is kept.
    out["prev_log_amount"] = history[TARGET].transform(lambda s: s.shift(1).ffill())
    out["n_prev_rounds"] = history.cumcount()

    # The row itself is counted, matching `lead_freq_map + 1` at prediction time.
    out["lead_freq"] = out["lead_investor"].map(out["lead_investor"].value_counts())
    out["company_age"] = out["year"] - out["founded_year"]
    out["stage_rank"] = out["stage"].map(STAGE_RANK)
    return out


def usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: verified rounds with an amount not flagged `amount_suspect_low`."""
    mask = (
        (df["amount_status"] == "verified")
        & df["amount_usd"].notna()
        & ~df["amount_suspect_low"]
    )
    return df[mask]
