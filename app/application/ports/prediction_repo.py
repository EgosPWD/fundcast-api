from typing import Protocol

from app.domain.entities import Page, Prediction


class PredictionRepository(Protocol):
    def add(self, prediction: Prediction) -> Prediction:
        """Persist and return the prediction with its id and created_at."""
        ...

    def list(self, user_id: int | None, page: int, size: int) -> Page[Prediction]:
        """List predictions, newest first. user_id=None lists every user's."""
        ...
