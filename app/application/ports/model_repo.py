from typing import Protocol

from app.domain.entities import ModelVersion


class ModelVersionRepository(Protocol):
    def add(self, model_version: ModelVersion) -> ModelVersion: ...

    def get_active(self) -> ModelVersion | None: ...

    def list_all(self) -> list[ModelVersion]:
        """Every version, newest first."""
        ...

    def activate(self, model_version_id: int) -> None:
        """Mark one version active and every other inactive, atomically."""
        ...

    def next_version_number(self) -> int: ...
