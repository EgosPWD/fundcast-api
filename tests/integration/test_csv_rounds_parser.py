"""CsvRoundsParser reads uploaded CSVs as data (AGENTS.md sections 2, 6 and 13)."""

from datetime import date
from pathlib import Path

import pytest

from app.adapters.outbound.ml.csv_rounds_parser import CsvRoundsParser
from app.domain.entities import FundingRound
from app.domain.errors import InvalidTrainingFileError

HEADER = ("round_id,announced_date,year,company_name,company_domain,country,city,sector,"
          "founded_year,stage,amount_usd,amount_status,lead_investor,investor_count,"
          "news_mentions,news_source,news_url,news_excerpt,fliar_profile_url")
ROW_A = ("r1,2025-03-01,2025,Acme, acme.io ,United States,New York,Fintech,2021,Seed,"
         "1500000,verified,Y Combinator,3,2,TechCrunch,https://tc.com/a,raised $1.5M,"
         "https://fliar/acme")
ROW_B = "r2,2024-06-15,2024,,beta.io,Chile,,,2020,Series A,,not_disclosed,,1,0,,,,"


def write(tmp_path: Path, *lines: str, name: str = "rounds.csv") -> str:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_parses_rows_in_file_order_as_domain_rounds(tmp_path):
    parsed = CsvRoundsParser().parse(write(tmp_path, HEADER, ROW_A, ROW_B))

    assert parsed.rejected == 0
    first, second = parsed.rounds
    assert first == FundingRound(
        announced_date=date(2025, 3, 1), year=2025, company_domain=" acme.io ",
        founded_year=2021, stage="Seed", amount_status="verified", round_id="r1",
        company_name="Acme", country="United States", city="New York", sector="Fintech",
        amount_usd=1_500_000.0, lead_investor="Y Combinator", investor_count=3,
        news_mentions=2, news_source="TechCrunch", news_url="https://tc.com/a",
        fliar_profile_url="https://fliar/acme",
    )
    assert type(first.year) is int and type(first.investor_count) is int
    assert type(first.amount_usd) is float
    # NaN becomes None for every optional field.
    assert (second.company_name, second.city, second.sector, second.amount_usd,
            second.lead_investor, second.news_source, second.news_url,
            second.fliar_profile_url) == (None,) * 8
    assert second.round_id == "r2"


def test_news_excerpt_is_dropped(tmp_path):
    parsed = CsvRoundsParser().parse(write(tmp_path, HEADER, ROW_A))

    assert not hasattr(parsed.rounds[0], "news_excerpt")
    assert "raised $1.5M" not in repr(parsed.rounds[0])


def test_optional_columns_may_be_missing(tmp_path):
    header = ("announced_date,year,company_domain,country,city,sector,founded_year,stage,"
              "amount_usd,amount_status,lead_investor,investor_count,news_mentions,news_source")
    row = "2025-03-01,2025,acme.io,Chile,Santiago,Fintech,2021,Seed,100000,verified,,,,"

    [only] = CsvRoundsParser().parse(write(tmp_path, header, row)).rounds

    assert (only.round_id, only.company_name, only.news_url, only.fliar_profile_url) == (
        None, None, None, None)
    assert (only.investor_count, only.news_mentions) == (None, None)


def test_numeric_round_ids_are_kept_as_text(tmp_path):
    parsed = CsvRoundsParser().parse(write(tmp_path, HEADER, "007" + ROW_A[2:]))

    assert parsed.rounds[0].round_id == "007"


def test_missing_required_columns_are_named(tmp_path):
    header = HEADER.replace("amount_usd,", "").replace("company_domain,", "")
    with pytest.raises(InvalidTrainingFileError, match="amount_usd, company_domain"):
        CsvRoundsParser().parse(write(tmp_path, header))


def test_any_year_mismatch_rejects_the_whole_file(tmp_path):
    bad = ROW_B.replace("2024-06-15,2024", "2023-06-15,2024")
    with pytest.raises(InvalidTrainingFileError, match="1 row"):
        CsvRoundsParser().parse(write(tmp_path, HEADER, ROW_A, bad))


def test_unparseable_date_rejects_the_whole_file(tmp_path):
    bad = ROW_B.replace("2024-06-15", "not-a-date")
    with pytest.raises(InvalidTrainingFileError):
        CsvRoundsParser().parse(write(tmp_path, HEADER, bad))


def test_rows_with_null_founded_year_or_negative_age_are_rejected(tmp_path):
    no_founded = ROW_B.replace(",2020,", ",,").replace("r2", "r3")
    negative_age = ROW_B.replace(",2020,", ",2025,").replace("r2", "r4")

    parsed = CsvRoundsParser().parse(write(tmp_path, HEADER, ROW_A, no_founded, negative_age,
                                           ROW_B))

    assert parsed.rejected == 2
    assert [r.round_id for r in parsed.rounds] == ["r1", "r2"]


def test_rows_without_company_domain_stage_or_amount_status_are_rejected(tmp_path):
    no_domain = ROW_B.replace("beta.io", "").replace("r2", "r3")
    no_stage = ROW_B.replace("Series A", "").replace("r2", "r4")
    no_status = ROW_B.replace("not_disclosed", "").replace("r2", "r5")

    parsed = CsvRoundsParser().parse(write(tmp_path, HEADER, no_domain, no_stage, no_status,
                                           ROW_B))

    assert parsed.rejected == 3
    assert [r.round_id for r in parsed.rounds] == ["r2"]


@pytest.mark.parametrize(
    "content",
    [b"", b"\xff\xfe\x00garbage\x00", b'announced_date,year\n"unterminated,2025\n'],
)
def test_unreadable_csv_is_an_invalid_training_file(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_bytes(content)

    with pytest.raises(InvalidTrainingFileError):
        CsvRoundsParser().parse(str(path))


def test_non_numeric_value_in_a_numeric_column_is_an_invalid_training_file(tmp_path):
    with pytest.raises(InvalidTrainingFileError, match="investor_count"):
        CsvRoundsParser().parse(write(tmp_path, HEADER, ROW_B.replace(",1,0,", ",many,0,")))


def test_header_only_file_has_no_rounds(tmp_path):
    parsed = CsvRoundsParser().parse(write(tmp_path, HEADER))

    assert (parsed.rounds, parsed.rejected) == ([], 0)
