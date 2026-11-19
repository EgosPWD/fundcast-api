from dataclasses import asdict

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from app.adapters.outbound.postgres.models import FundingRoundRow
from app.domain.entities import FundingRound

# 20 columns per row; chunks keep each INSERT far below PostgreSQL's 65,535 parameter limit.
CHUNK_SIZE = 1_000
_STORED_FIELDS = tuple(
    name for name in FundingRound.__dataclass_fields__ if name not in ("id", "created_at")
)


def _to_domain(row: FundingRoundRow) -> FundingRound:
    return FundingRound(**{name: getattr(row, name) for name in _STORED_FIELDS},
                        id=row.id, created_at=row.created_at)


class SqlAlchemyFundingRoundRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    def add_many(self, rounds: list[FundingRound]) -> int:
        inserted = 0
        with self._sessions.begin() as session:
            for start in range(0, len(rounds), CHUNK_SIZE):
                values = [
                    {k: v for k, v in asdict(r).items() if k in _STORED_FIELDS}
                    for r in rounds[start : start + CHUNK_SIZE]
                ]
                # Ids follow the VALUES order, so list_all() keeps the file order.
                statement = (
                    insert(FundingRoundRow)
                    .values(values)
                    .on_conflict_do_nothing(index_elements=["round_id"])
                    .returning(FundingRoundRow.id)
                )
                inserted += len(session.execute(statement).all())
        return inserted

    def list_all(self) -> list[FundingRound]:
        with self._sessions() as session:
            rows = session.scalars(select(FundingRoundRow).order_by(FundingRoundRow.id))
            return [_to_domain(r) for r in rows]

    def count(self) -> int:
        with self._sessions() as session:
            return session.scalar(select(func.count()).select_from(FundingRoundRow)) or 0

    def history(self, company_domain: str, up_to_year: int) -> list[FundingRound]:
        query = (
            select(FundingRoundRow)
            .where(FundingRoundRow.company_domain == company_domain,
                   FundingRoundRow.year <= up_to_year)
            .order_by(FundingRoundRow.announced_date, FundingRoundRow.id)
        )
        with self._sessions() as session:
            return [_to_domain(r) for r in session.scalars(query)]
