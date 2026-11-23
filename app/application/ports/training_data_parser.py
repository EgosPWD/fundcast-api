from dataclasses import dataclass, field
from typing import Protocol

from app.domain.entities import FundingRound


@dataclass(frozen=True, slots=True)
class ParsedRounds:
    """Rows of a training CSV ready for funding_rounds, in file order."""

    rounds: list[FundingRound] = field(default_factory=list)
    rejected: int = 0


class TrainingDataParser(Protocol):
    def parse(self, path: str) -> ParsedRounds:
        """Read a training CSV as data only (never executes its content).

        Raises InvalidTrainingFileError (HTTP 422) for an unreadable file, missing
        required columns or any row whose year differs from its announced_date year.
        Rows with a null founded_year or a negative company age are rejected (counted,
        not returned). `news_excerpt` is dropped: it is never stored.
        """
        ...
