from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.domain.entities import FundingRound, ModelMetrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    artifact_path: str
    metrics: ModelMetrics
    n_train: int
    n_test: int
    params: dict[str, Any] = field(default_factory=dict)


class Trainer(Protocol):
    """Cleans the data, fits the pipeline and evaluates it on the held-out test split."""

    def train(
        self, rounds: Sequence[FundingRound], params: dict[str, Any], artifact_path: str
    ) -> TrainingResult:
        """Train on `rounds` (ordered by id) with a new stratified 80/20 split and fixed
        params (no hyperparameter search), then save the artifact.

        Raises InsufficientTrainingDataError (HTTP 422) if the split is not possible.
        """
        ...
