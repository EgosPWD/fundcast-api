from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.adapters.outbound.postgres.models import TrainingFileRow
from app.domain.entities import TrainingFile, TrainingFileStatus


def _to_domain(row: TrainingFileRow) -> TrainingFile:
    return TrainingFile(filename=row.filename, stored_path=row.stored_path, rows=row.rows,
                        uploaded_by=row.uploaded_by, status=TrainingFileStatus(row.status),
                        rows_inserted=row.rows_inserted, rows_rejected=row.rows_rejected,
                        id=row.id, created_at=row.created_at)


class SqlAlchemyTrainingFileRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    def add(self, training_file: TrainingFile) -> TrainingFile:
        with self._sessions.begin() as session:
            row = TrainingFileRow(filename=training_file.filename,
                                  stored_path=training_file.stored_path, rows=training_file.rows,
                                  uploaded_by=training_file.uploaded_by,
                                  status=training_file.status.value,
                                  rows_inserted=training_file.rows_inserted,
                                  rows_rejected=training_file.rows_rejected)
            session.add(row)
            session.flush()
            session.refresh(row)
            return _to_domain(row)

    def get_by_id(self, training_file_id: int) -> TrainingFile | None:
        with self._sessions() as session:
            row = session.get(TrainingFileRow, training_file_id)
            return _to_domain(row) if row else None

    def get_latest(self) -> TrainingFile | None:
        with self._sessions() as session:
            row = session.scalar(select(TrainingFileRow).order_by(TrainingFileRow.id.desc()))
            return _to_domain(row) if row else None

    def update_status(self, training_file_id: int, status: TrainingFileStatus) -> None:
        with self._sessions.begin() as session:
            session.execute(update(TrainingFileRow)
                            .where(TrainingFileRow.id == training_file_id)
                            .values(status=status.value))
