from app.application.ports.model_inspector import ModelArtifactInspector
from app.application.ports.model_repo import ModelVersionRepository
from app.domain.entities import ModelVersion


class EnsureBaselineModel:
    """At startup, register the versioned baseline artifact as active version 1 when the
    model registry is empty, so the API predicts from the first boot (idempotent).

    Its metrics are the ones stored in the artifact by the notebook: they are what later
    retrainings are compared with (AGENTS.md sections 4 and 6).
    """

    def __init__(self, models: ModelVersionRepository, inspector: ModelArtifactInspector) -> None:
        self._models, self._inspector = models, inspector

    def execute(self, artifact_path: str, n_train: int, n_test: int) -> ModelVersion | None:
        """Return the registered version, or None if any version already exists."""
        if self._models.list_all():
            return None
        metadata = self._inspector.read_metadata(artifact_path)
        return self._models.add(
            ModelVersion(
                version=self._models.next_version_number(),
                artifact_path=artifact_path,
                metrics=metadata.metrics,
                n_train=n_train,
                n_test=n_test,
                params=metadata.params,
                is_active=True,
            )
        )
