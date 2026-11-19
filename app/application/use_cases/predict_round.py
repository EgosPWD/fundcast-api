from dataclasses import dataclass

from app.application.ports.funding_round_repo import FundingRoundRepository
from app.application.ports.model_repo import ModelVersionRepository
from app.application.ports.prediction_repo import PredictionRepository
from app.application.ports.predictor import Predictor
from app.domain.entities import CompanyHistory, Estimate, FundingRoundInput, Prediction
from app.domain.errors import ModelNotAvailableError


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    prediction: Prediction
    estimate: Estimate
    model_version: int


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
        prediction = self._predictions.add(
            Prediction(
                user_id=user_id,
                model_version_id=active.id,
                input=round_input,
                estimated_amount_usd=estimate.estimated_amount_usd,
                estimated_log10=estimate.estimated_log10,
            )
        )
        return PredictionOutcome(prediction, estimate, active.version)

    def _company_history(self, round_input: FundingRoundInput) -> CompanyHistory | None:
        """Stored rounds win over the client's values when the company is known."""
        if round_input.company_domain is None:
            return None
        rounds = self._funding_rounds.history(round_input.company_domain, round_input.year)
        return CompanyHistory.from_rounds(rounds) if rounds else None
