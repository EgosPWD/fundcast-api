from app.application.ports.password_hasher import PasswordHasher
from app.application.ports.user_repo import UserRepository
from app.domain.entities import Role, User, normalize_email
from app.domain.errors import EmailAlreadyRegisteredError, InvalidInputError

MIN_PASSWORD_LENGTH = 8


class RegisterUser:
    """Public sign-up. Always creates role `user`: admins come only from EnsureAdmin."""

    def __init__(self, users: UserRepository, hasher: PasswordHasher) -> None:
        self._users, self._hasher = users, hasher

    def execute(self, email: str, password: str) -> User:
        email = normalize_email(email)
        if "@" not in email:
            raise InvalidInputError("email is not valid")
        if len(password or "") < MIN_PASSWORD_LENGTH:
            raise InvalidInputError(f"password must have at least {MIN_PASSWORD_LENGTH} characters")
        if self._users.get_by_email(email) is not None:
            raise EmailAlreadyRegisteredError("email is already registered")
        return self._users.add(
            User(email=email, password_hash=self._hasher.hash(password), role=Role.USER)
        )
