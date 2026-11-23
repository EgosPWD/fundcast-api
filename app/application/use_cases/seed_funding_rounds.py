from pathlib import Path

from app.application.ports.funding_round_repo import FundingRoundRepository
from app.application.ports.training_data_parser import TrainingDataParser


class SeedFundingRounds:
    """At startup, load the original Kaggle CSV into an empty funding_rounds table, in
    file order and with training_file_id NULL (AGENTS.md section 6, "Arranque")."""

    def __init__(self, rounds: FundingRoundRepository, parser: TrainingDataParser) -> None:
        self._rounds, self._parser = rounds, parser

    def execute(self, dataset_path: str | None) -> int:
        """Return how many rows were inserted (0 if skipped)."""
        if not dataset_path or not Path(dataset_path).is_file():
            return 0
        if self._rounds.count() > 0:
            return 0
        parsed = self._parser.parse(dataset_path)
        return self._rounds.add_many(list(parsed.rounds))
