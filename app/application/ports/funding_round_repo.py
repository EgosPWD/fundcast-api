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

    def history(self, company_domain: str, up_to_year: int) -> list[FundingRound]:
        """Rounds of one company with year <= up_to_year, ordered by announced_date."""
        ...
