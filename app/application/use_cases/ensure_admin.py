from app.application.ports.password_hasher import PasswordHasher
from app.application.ports.user_repo import UserRepository
from app.application.use_cases.register_user import normalize_email
from app.domain.entities import Role, User


class EnsureAdmin:
    """Create the admin account at startup if it does not exist yet (idempotent)."""

    def __init__(self, users: UserRepository, hasher: PasswordHasher) -> None:
        self._users, self._hasher = users, hasher

    def execute(self, email: str, password: str) -> User:
        email = normalize_email(email)
        existing = self._users.get_by_email(email)
        if existing is not None:
            return existing
        return self._users.add(
            User(email=email, password_hash=self._hasher.hash(password), role=Role.ADMIN)
        )
