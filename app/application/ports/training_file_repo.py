from typing import Protocol

from app.domain.entities import TrainingFile, TrainingFileStatus


class TrainingFileRepository(Protocol):
    def add(self, training_file: TrainingFile) -> TrainingFile: ...

    def get_by_id(self, training_file_id: int) -> TrainingFile | None: ...

    def get_latest(self, status: TrainingFileStatus | None = None) -> TrainingFile | None:
        """Newest file (highest id), optionally only among files with `status`."""
        ...

    def update_status(
        self, training_file_id: int, status: TrainingFileStatus, rows_inserted: int | None = None
    ) -> None:
        """Set the load status and, when given, the number of rows inserted."""
        ...
