from app.application.ports.funding_round_repo import FundingRoundRepository
from app.domain.entities import CategoryOptions


class GetOptions:
    """Values the prediction form can offer: the categories the stored rounds contain."""

    def __init__(self, rounds: FundingRoundRepository) -> None:
        self._rounds = rounds

    def execute(self) -> CategoryOptions:
        found = self._rounds.category_options()
        return CategoryOptions(sectors=sorted(found.sectors), stages=sorted(found.stages),
                               countries=sorted(found.countries))
