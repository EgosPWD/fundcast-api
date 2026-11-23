"""PasswordHasher adapter backed by bcrypt (AGENTS.md section 8)."""

import bcrypt

from app.domain.errors import InvalidInputError

# bcrypt only uses the first 72 bytes; bcrypt >= 5 refuses longer input outright.
BCRYPT_MAX_BYTES = 72


class BcryptPasswordHasher:
    def __init__(self, rounds: int = 12) -> None:
        self._rounds = rounds

    def hash(self, plain_password: str) -> str:
        password = plain_password.encode("utf-8")
        if len(password) > BCRYPT_MAX_BYTES:
            raise InvalidInputError(f"password must be at most {BCRYPT_MAX_BYTES} bytes")
        return bcrypt.hashpw(password, bcrypt.gensalt(rounds=self._rounds)).decode("ascii")

    def verify(self, plain_password: str, password_hash: str) -> bool:
        password = plain_password.encode("utf-8")
        if len(password) > BCRYPT_MAX_BYTES:
            return False
        try:
            return bcrypt.checkpw(password, password_hash.encode("utf-8"))
        except ValueError:
            # Malformed stored hash: never authenticate, never crash the request.
            return False
