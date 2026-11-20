from dataclasses import dataclass, field
from typing import Any, Protocol

from app.domain.entities import ModelMetrics


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    metrics: ModelMetrics
    params: dict[str, Any] = field(default_factory=dict)


class ModelArtifactInspector(Protocol):
    def read_metadata(self, artifact_path: str) -> ArtifactMetadata:
        """Read the test metrics and params stored inside a model artifact."""
        ...
