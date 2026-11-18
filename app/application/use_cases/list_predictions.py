from app.application.ports.prediction_repo import PredictionRepository
from app.application.ports.token_service import TokenClaims
from app.domain.entities import Page, Prediction, Role
from app.domain.errors import InvalidInputError

MAX_PAGE_SIZE = 100


class ListPredictions:
    """History: a `user` sees only their own predictions, an `admin` sees all."""

    def __init__(self, predictions: PredictionRepository) -> None:
        self._predictions = predictions

    def execute(self, requester: TokenClaims, page: int, size: int) -> Page[Prediction]:
        if page < 1:
            raise InvalidInputError("page must be >= 1")
        if not 1 <= size <= MAX_PAGE_SIZE:
            raise InvalidInputError(f"size must be between 1 and {MAX_PAGE_SIZE}")
        user_id = None if requester.role is Role.ADMIN else requester.user_id
        return self._predictions.list(user_id=user_id, page=page, size=size)
