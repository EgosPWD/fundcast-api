from dataclasses import asdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.adapters.outbound.postgres.models import PredictionRow
from app.domain.entities import FundingRoundInput, Page, Prediction


def _to_domain(row: PredictionRow) -> Prediction:
    return Prediction(user_id=row.user_id, model_version_id=row.model_version_id,
                      input=FundingRoundInput.from_stored(row.input),
                      estimated_amount_usd=row.estimated_amount_usd,
                      estimated_log10=row.estimated_log10, id=row.id, created_at=row.created_at)


class SqlAlchemyPredictionRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    def add(self, prediction: Prediction) -> Prediction:
        with self._sessions.begin() as session:
            row = PredictionRow(user_id=prediction.user_id,
                                model_version_id=prediction.model_version_id,
                                input=asdict(prediction.input),
                                estimated_amount_usd=prediction.estimated_amount_usd,
                                estimated_log10=prediction.estimated_log10)
            session.add(row)
            session.flush()
            session.refresh(row)
            return _to_domain(row)

    def list(self, user_id: int | None, page: int, size: int) -> Page[Prediction]:
        query = select(PredictionRow)
        count = select(func.count()).select_from(PredictionRow)
        if user_id is not None:
            query = query.where(PredictionRow.user_id == user_id)
            count = count.where(PredictionRow.user_id == user_id)
        query = (query.order_by(PredictionRow.created_at.desc(), PredictionRow.id.desc())
                 .offset((page - 1) * size).limit(size))
        with self._sessions() as session:
            items = [_to_domain(r) for r in session.scalars(query)]
            return Page(items=items, total=session.scalar(count) or 0, page=page, size=size)
