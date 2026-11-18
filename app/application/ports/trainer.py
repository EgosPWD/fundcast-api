from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.domain.entities import ModelMetrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    artifact_path: str
    metrics: ModelMetrics
    n_train: int
    n_test: int
    params: dict[str, Any] = field(default_factory=dict)


class Trainer(Protocol):
    """Cleans the data, fits the pipeline and evaluates it on the held-out test split."""

    def validate_dataset(self, path: str) -> int:
        """Check the CSV columns and return its row count.

        Raises InvalidTrainingFileError. Never executes file content.
        """
        ...

    def train(
        self, source_paths: Sequence[str], params: dict[str, Any], artifact_path: str
    ) -> TrainingResult:
        """Train with fixed params (no hyperparameter search) and save the artifact."""
        ...
