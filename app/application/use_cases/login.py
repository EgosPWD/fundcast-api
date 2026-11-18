from app.application.ports.password_hasher import PasswordHasher
from app.application.ports.token_service import TokenClaims, TokenService
from app.application.ports.user_repo import UserRepository
from app.domain.entities import normalize_email
from app.domain.errors import AuthenticationError


class Login:
    def __init__(self, users: UserRepository, hasher: PasswordHasher, tokens: TokenService):
        self._users, self._hasher, self._tokens = users, hasher, tokens

    def execute(self, email: str, password: str) -> str:
        user = self._users.get_by_email(normalize_email(email))
        # Same message for unknown email and wrong password: do not reveal which accounts exist.
        if user is None or user.id is None or not self._hasher.verify(password, user.password_hash):
            raise AuthenticationError("Invalid email or password")
        return self._tokens.issue(TokenClaims(user_id=user.id, role=user.role))
