from dataclasses import dataclass

from app.application.ports.model_repo import ModelVersionRepository
from app.domain.entities import ModelVersion


@dataclass(frozen=True, slots=True)
class MetricsReport:
    active: ModelVersion | None
    versions: list[ModelVersion]


class GetMetrics:
    def __init__(self, models: ModelVersionRepository) -> None:
        self._models = models

    def execute(self) -> MetricsReport:
        return MetricsReport(active=self._models.get_active(), versions=self._models.list_all())
