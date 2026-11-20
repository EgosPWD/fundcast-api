"""TokenService adapter: signed JWT with an expiry (AGENTS.md section 8)."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import jwt

from app.application.ports.token_service import TokenClaims
from app.domain.entities import Role
from app.domain.errors import AuthenticationError


def _utc_now() -> datetime:
    return datetime.now(UTC)


class JwtTokenService:
    def __init__(
        self,
        secret: str,
        algorithm: str = "HS256",
        expire_minutes: int = 60,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._secret, self._algorithm = secret, algorithm
        self._ttl = timedelta(minutes=expire_minutes)
        self._clock = clock

    def issue(self, claims: TokenClaims) -> str:
        now = self._clock()
        payload = {"sub": str(claims.user_id), "role": claims.role.value,
                   "iat": now, "exp": now + self._ttl}
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def decode(self, token: str) -> TokenClaims:
        try:
            # A fixed algorithm list blocks "alg: none" and algorithm-confusion tokens.
            payload = jwt.decode(token, self._secret, algorithms=[self._algorithm],
                                 options={"require": ["exp", "sub"]})
        except jwt.ExpiredSignatureError as error:
            raise AuthenticationError("Token has expired") from error
        except jwt.PyJWTError as error:
            raise AuthenticationError("Invalid token") from error
        try:
            return TokenClaims(user_id=int(payload["sub"]), role=Role(payload["role"]))
        except (KeyError, TypeError, ValueError) as error:
            raise AuthenticationError("Invalid token") from error
