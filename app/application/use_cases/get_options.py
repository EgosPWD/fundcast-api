from app.application.ports.funding_round_repo import FundingRoundRepository
from app.application.ports.model_repo import ModelVersionRepository
from app.application.ports.predictor import Predictor
from app.domain.entities import FormOptions
from app.domain.errors import ModelNotAvailableError


class GetOptions:
    """Values the prediction form can offer: the vocabulary of the ACTIVE model.

    The model matches categories by exact text, so only what it saw in training is useful.
    Read on every call: when a retrain activates another version the lists change with it.
    City-to-country pairs come from funding_rounds (the model has no such link), keeping
    only cities and countries the model knows.
    """

    def __init__(self, models: ModelVersionRepository, predictor: Predictor,
                 rounds: FundingRoundRepository) -> None:
        self._models, self._predictor, self._rounds = models, predictor, rounds

    def execute(self) -> FormOptions:
        active = self._models.get_active()
        if active is None:
            raise ModelNotAvailableError("No active model version")
        known = self._predictor.vocabulary(active.artifact_path)
        cities: dict[str, set[str]] = {}
        for city, country in self._rounds.city_countries():
            if city in known.cities and country in known.countries:
                cities.setdefault(country, set()).add(city)
        return FormOptions(
            stages=sorted(known.stages), sectors=sorted(known.sectors),
            countries=sorted(known.countries),
            cities_by_country={c: sorted(v) for c, v in sorted(cities.items())},
            lead_investors=sorted(known.lead_investors),
            news_sources=sorted(known.news_sources), model_version=active.version)
