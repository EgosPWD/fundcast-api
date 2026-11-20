from dataclasses import replace

from app.application.ports.file_storage import FileStorage
from app.application.ports.funding_round_repo import FundingRoundRepository
from app.application.ports.training_data_parser import TrainingDataParser
from app.application.ports.training_file_repo import TrainingFileRepository
from app.domain.entities import MAX_TRAINING_FILE_BYTES, TrainingFile, TrainingFileStatus
from app.domain.errors import InvalidTrainingFileError


class UploadTrainingFile:
    """Validate extension and size, store the CSV, parse it and insert its rows into
    funding_rounds (AGENTS.md section 6)."""

    def __init__(
        self,
        storage: FileStorage,
        parser: TrainingDataParser,
        files: TrainingFileRepository,
        rounds: FundingRoundRepository,
    ) -> None:
        self._storage, self._parser, self._files, self._rounds = storage, parser, files, rounds

    def execute(self, uploader_id: int, filename: str, content: bytes) -> TrainingFile:
        if not (filename or "").lower().endswith(".csv"):
            raise InvalidTrainingFileError("file must have a .csv extension")
        if len(content) > MAX_TRAINING_FILE_BYTES:
            raise InvalidTrainingFileError("file exceeds the 20 MB limit")
        stored_path = self._storage.save(filename, content)
        try:
            return self._load(uploader_id, filename, stored_path)
        except Exception:
            # No orphan files: whatever fails after saving removes the stored CSV.
            self._storage.delete(stored_path)
            raise

    def _load(self, uploader_id: int, filename: str, stored_path: str) -> TrainingFile:
        parsed = self._parser.parse(stored_path)
        # Registered as failed first: the rows need its id, and the file must never be
        # reported as loaded before they reach funding_rounds.
        registered = self._files.add(
            TrainingFile(
                filename=filename,
                stored_path=stored_path,
                rows=len(parsed.rounds) + parsed.rejected,
                uploaded_by=uploader_id,
                status=TrainingFileStatus.FAILED,
                rows_rejected=parsed.rejected,
            )
        )
        if registered.id is None:
            raise RuntimeError("training file was registered without an id")
        inserted = self._rounds.add_many(
            [replace(r, training_file_id=registered.id) for r in parsed.rounds]
        )
        self._files.update_status(registered.id, TrainingFileStatus.LOADED, rows_inserted=inserted)
        return replace(registered, status=TrainingFileStatus.LOADED, rows_inserted=inserted)
