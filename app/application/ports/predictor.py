from collections.abc import Mapping
from typing import Any, Protocol


class Predictor(Protocol):
    """Runs a trained model artifact on one row of model features."""

    def predict_log10(self, artifact_path: str, features: dict[str, Any]) -> float:
        """Return the predicted log10(amount_usd) for the 14 contract columns."""
        ...

    def lead_frequency_map(self, artifact_path: str) -> Mapping[str, int]:
        """Rounds led per investor, as counted when this artifact was trained."""
        ...
