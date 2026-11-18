from dataclasses import dataclass
from typing import Any

from app.application.ports.model_repo import ModelVersionRepository
from app.application.ports.retrain_lock import RetrainLock
from app.application.ports.trainer import Trainer
from app.application.ports.training_file_repo import TrainingFileRepository
from app.domain.entities import ModelVersion, TrainingFile, TrainingFileStatus, should_activate
from app.domain.errors import NotFoundError, RetrainInProgressError


@dataclass(frozen=True, slots=True)
class RetrainOutcome:
    model_version: ModelVersion
    activated: bool


class RetrainModel:
    """Retrain on the original CSV plus an uploaded file, with the fixed winning params.

    The new version goes live only if MAE_log_new <= MAE_log_active + 0.01;
    otherwise it is stored inactive (AGENTS.md section 4).
    """

    def __init__(
        self,
        files: TrainingFileRepository,
        models: ModelVersionRepository,
        trainer: Trainer,
        lock: RetrainLock,
        baseline_dataset_path: str,
        artifacts_dir: str,
        params: dict[str, Any],
    ) -> None:
        self._files, self._models, self._trainer, self._lock = files, models, trainer, lock
        self._baseline_dataset_path = baseline_dataset_path
        self._artifacts_dir = artifacts_dir.rstrip("/")
        self._params = params

    def execute(self, file_id: int | None) -> RetrainOutcome:
        if not self._lock.acquire():
            raise RetrainInProgressError("A retraining is already in progress")
        try:
            training_file = self._find_file(file_id)
            return self._retrain(training_file)
        finally:
            self._lock.release()

    def _find_file(self, file_id: int | None) -> TrainingFile:
        found = self._files.get_latest() if file_id is None else self._files.get_by_id(file_id)
        if found is None or found.id is None:
            raise NotFoundError("Training file not found")
        return found

    def _retrain(self, training_file: TrainingFile) -> RetrainOutcome:
        assert training_file.id is not None
        version = self._models.next_version_number()
        try:
            result = self._trainer.train(
                [self._baseline_dataset_path, training_file.stored_path],
                self._params,
                f"{self._artifacts_dir}/model_v{version}.joblib",
            )
        except Exception:
            self._files.update_status(training_file.id, TrainingFileStatus.FAILED)
            raise
        active = self._models.get_active()
        activate = should_activate(result.metrics, active.metrics if active else None)
        saved = self._models.add(
            ModelVersion(
                version=version,
                artifact_path=result.artifact_path,
                metrics=result.metrics,
                n_train=result.n_train,
                n_test=result.n_test,
                params=result.params,
                training_file_id=training_file.id,
            )
        )
        if activate and saved.id is not None:
            self._models.activate(saved.id)
        self._files.update_status(training_file.id, TrainingFileStatus.USED)
        return RetrainOutcome(model_version=saved, activated=activate)
