from dataclasses import dataclass, replace
from typing import Any

from app.application.ports.funding_round_repo import FundingRoundRepository
from app.application.ports.model_repo import ModelVersionRepository
from app.application.ports.retrain_lock import RetrainLock
from app.application.ports.trainer import Trainer
from app.application.ports.training_file_repo import TrainingFileRepository
from app.domain.entities import ModelVersion, TrainingFileStatus, should_activate
from app.domain.errors import RetrainInProgressError


@dataclass(frozen=True, slots=True)
class RetrainOutcome:
    model_version: ModelVersion
    activated: bool


class RetrainModel:
    """Retrain on every row of funding_rounds with a new stratified 80/20 split and the
    fixed winning params (AGENTS.md section 4, "Flujo de /retrain").

    The new version goes live only if MAE_log_new <= MAE_log_active + 0.01, compared with
    the stored metrics of the active version; otherwise it is stored inactive. A failed
    retraining never touches the active model.
    """

    def __init__(
        self,
        rounds: FundingRoundRepository,
        files: TrainingFileRepository,
        models: ModelVersionRepository,
        trainer: Trainer,
        lock: RetrainLock,
        artifacts_dir: str,
        params: dict[str, Any],
    ) -> None:
        self._rounds, self._files, self._models = rounds, files, models
        self._trainer, self._lock = trainer, lock
        self._artifacts_dir = artifacts_dir.rstrip("/")
        self._params = params

    def execute(self) -> RetrainOutcome:
        if not self._lock.acquire():
            raise RetrainInProgressError("A retraining is already in progress")
        try:
            return self._retrain()
        finally:
            self._lock.release()

    def _retrain(self) -> RetrainOutcome:
        # Read before training: the version records the last load included in its data.
        latest = self._files.get_latest(TrainingFileStatus.LOADED)
        rounds = self._rounds.list_all()
        version = self._models.next_version_number()
        result = self._trainer.train(
            rounds, self._params, f"{self._artifacts_dir}/model_v{version}.joblib"
        )
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
                training_file_id=latest.id if latest else None,
            )
        )
        if activate and saved.id is not None:
            self._models.activate(saved.id)
            # Repositories return copies: reflect the activation in the returned value.
            saved = replace(saved, is_active=True)
        return RetrainOutcome(model_version=saved, activated=activate)
