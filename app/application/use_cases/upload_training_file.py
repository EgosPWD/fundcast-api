from app.application.ports.file_storage import FileStorage
from app.application.ports.trainer import Trainer
from app.application.ports.training_file_repo import TrainingFileRepository
from app.domain.entities import MAX_TRAINING_FILE_BYTES, TrainingFile
from app.domain.errors import InvalidTrainingFileError


class UploadTrainingFile:
    """Validate extension and size, store the CSV, check its columns, register it."""

    def __init__(
        self, storage: FileStorage, trainer: Trainer, files: TrainingFileRepository
    ) -> None:
        self._storage, self._trainer, self._files = storage, trainer, files

    def execute(self, uploader_id: int, filename: str, content: bytes) -> TrainingFile:
        if not (filename or "").lower().endswith(".csv"):
            raise InvalidTrainingFileError("file must have a .csv extension")
        if len(content) > MAX_TRAINING_FILE_BYTES:
            raise InvalidTrainingFileError("file exceeds the 20 MB limit")
        stored_path = self._storage.save(filename, content)
        try:
            rows = self._trainer.validate_dataset(stored_path)
        except Exception:
            self._storage.delete(stored_path)
            raise
        return self._files.add(
            TrainingFile(
                filename=filename, stored_path=stored_path, rows=rows, uploaded_by=uploader_id
            )
        )
