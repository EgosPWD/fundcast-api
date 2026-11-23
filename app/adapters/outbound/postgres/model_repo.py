from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.adapters.outbound.postgres.models import ModelVersionRow
from app.domain.entities import ModelMetrics, ModelVersion
from app.domain.errors import NotFoundError


def _to_domain(row: ModelVersionRow) -> ModelVersion:
    metrics = ModelMetrics(mae_log=row.mae_log, r2=row.r2, rmse_log=row.rmse_log,
                           mdape=row.mdape, coverage=row.coverage)
    return ModelVersion(version=row.version, artifact_path=row.artifact_path, metrics=metrics,
                        n_train=row.n_train, n_test=row.n_test, params=dict(row.params),
                        is_active=row.is_active, training_file_id=row.training_file_id,
                        id=row.id, created_at=row.created_at)


class SqlAlchemyModelVersionRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    def add(self, model_version: ModelVersion) -> ModelVersion:
        m = model_version.metrics
        with self._sessions.begin() as session:
            row = ModelVersionRow(
                version=model_version.version, artifact_path=model_version.artifact_path,
                is_active=model_version.is_active, mae_log=m.mae_log, r2=m.r2,
                rmse_log=m.rmse_log, mdape=m.mdape, coverage=m.coverage,
                n_train=model_version.n_train, n_test=model_version.n_test,
                params=model_version.params, training_file_id=model_version.training_file_id,
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return _to_domain(row)

    def get_active(self) -> ModelVersion | None:
        with self._sessions() as session:
            row = session.scalar(select(ModelVersionRow).where(ModelVersionRow.is_active))
            return _to_domain(row) if row else None

    def list_all(self) -> list[ModelVersion]:
        with self._sessions() as session:
            rows = session.scalars(select(ModelVersionRow).order_by(ModelVersionRow.version.desc()))
            return [_to_domain(r) for r in rows]

    def activate(self, model_version_id: int) -> None:
        # One transaction: deactivate first so the single-active unique index never conflicts.
        with self._sessions.begin() as session:
            if session.get(ModelVersionRow, model_version_id) is None:
                raise NotFoundError("Model version not found")
            session.execute(update(ModelVersionRow).where(ModelVersionRow.is_active)
                            .values(is_active=False))
            session.execute(update(ModelVersionRow)
                            .where(ModelVersionRow.id == model_version_id)
                            .values(is_active=True))

    def next_version_number(self) -> int:
        with self._sessions() as session:
            return (session.scalar(select(func.max(ModelVersionRow.version))) or 0) + 1
