from app.application.ports.password_hasher import PasswordHasher
from app.application.ports.user_repo import UserRepository
from app.domain.entities import Role, User, normalize_email
from app.domain.errors import AdminProvisioningError


class EnsureAdmin:
    """Create the admin account at startup if it does not exist yet (idempotent)."""

    def __init__(self, users: UserRepository, hasher: PasswordHasher) -> None:
        self._users, self._hasher = users, hasher

    def execute(self, email: str, password: str) -> User:
        email = normalize_email(email)
        existing = self._users.get_by_email(email)
        if existing is not None:
            if existing.role is not Role.ADMIN:
                # Never promote a public account silently: it keeps a password the
                # operator does not know. Resolving it requires a trusted manual action.
                raise AdminProvisioningError(
                    f"{email} is registered as a non-admin account; "
                    "resolve it manually before starting the service"
                )
            return existing
        return self._users.add(
            User(email=email, password_hash=self._hasher.hash(password), role=Role.ADMIN)
        )
