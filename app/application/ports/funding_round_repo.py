from typing import Protocol

from app.domain.entities import FundingRound


class FundingRoundRepository(Protocol):
    def add_many(self, rounds: list[FundingRound]) -> int:
        """Insert in the given order, skipping rows whose round_id already exists.

        Returns how many rows were inserted.
        """
        ...

    def list_all(self) -> list[FundingRound]:
        """Every row ordered by id: the stable order that makes the 80/20 split reproducible."""
        ...

    def count(self) -> int: ...

    def has_company(self, company_domain: str) -> bool:
        """Whether any round of this company is stored, whatever its year."""
        ...

    def history(self, company_domain: str, up_to_year: int) -> list[FundingRound]:
        """Rounds of one company with year <= up_to_year, ordered by announced_date."""
        ...

    def city_countries(self) -> list[tuple[str, str]]:
        """Distinct (city, country) pairs where both are non-null (any order)."""
        ...
