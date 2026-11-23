from dataclasses import dataclass, field

from app.application.ports.funding_round_repo import FundingRoundRepository
from app.application.ports.model_repo import ModelVersionRepository
from app.application.ports.prediction_repo import PredictionRepository
from app.application.ports.predictor import Predictor
from app.domain.entities import (
    CategoryWarning,
    CompanyHistory,
    Estimate,
    FundingRoundInput,
    Prediction,
)
from app.domain.errors import ModelNotAvailableError


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    prediction: Prediction
    estimate: Estimate
    model_version: int
    warnings: list[CategoryWarning] = field(default_factory=list)


class PredictRound:
    def __init__(
        self,
        models: ModelVersionRepository,
        predictor: Predictor,
        predictions: PredictionRepository,
        funding_rounds: FundingRoundRepository,
    ) -> None:
        self._models, self._predictor, self._predictions = models, predictor, predictions
        self._funding_rounds = funding_rounds

    def execute(self, user_id: int, round_input: FundingRoundInput) -> PredictionOutcome:
        active = self._models.get_active()
        if active is None or active.id is None:
            raise ModelNotAvailableError("No active model version")
        features = round_input.to_features(
            history=self._company_history(round_input),
            lead_freq_map=self._predictor.lead_frequency_map(active.artifact_path),
        )
        log10_value = self._predictor.predict_log10(active.artifact_path, features)
        estimate = Estimate.from_log10(log10_value, stage=round_input.stage)
        warnings = self._predictor.vocabulary(active.artifact_path).warnings_for(round_input)
        prediction = self._predictions.add(
            Prediction(
                user_id=user_id,
                model_version_id=active.id,
                input=round_input,
                estimated_amount_usd=estimate.estimated_amount_usd,
                estimated_log10=estimate.estimated_log10,
            )
        )
        return PredictionOutcome(prediction, estimate, active.version, warnings)

    def _company_history(self, round_input: FundingRoundInput) -> CompanyHistory | None:
        """Stored rounds win over the client's values when the company is known.

        A known company with no rounds up to `year` has an empty history (0, None):
        the client's values are used only for companies that are not stored at all.
        """
        domain = round_input.company_domain
        if domain is None or not self._funding_rounds.has_company(domain):
            return None
        return CompanyHistory.from_rounds(self._funding_rounds.history(domain, round_input.year))
