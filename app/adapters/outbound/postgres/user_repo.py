from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.adapters.outbound.postgres.models import UserRow
from app.domain.entities import Role, User


def _to_domain(row: UserRow) -> User:
    return User(email=row.email, password_hash=row.password_hash, role=Role(row.role),
                id=row.id, created_at=row.created_at)


class SqlAlchemyUserRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    def add(self, user: User) -> User:
        with self._sessions.begin() as session:
            row = UserRow(email=user.email, password_hash=user.password_hash, role=user.role.value)
            session.add(row)
            session.flush()
            session.refresh(row)
            return _to_domain(row)

    def get_by_email(self, email: str) -> User | None:
        with self._sessions() as session:
            row = session.scalar(select(UserRow).where(UserRow.email == email))
            return _to_domain(row) if row else None

    def get_by_id(self, user_id: int) -> User | None:
        with self._sessions() as session:
            row = session.get(UserRow, user_id)
            return _to_domain(row) if row else None
