from dataclasses import dataclass
from typing import Protocol

from app.domain.entities import Role


@dataclass(frozen=True, slots=True)
class TokenClaims:
    user_id: int
    role: Role


class TokenService(Protocol):
    def issue(self, claims: TokenClaims) -> str: ...

    def decode(self, token: str) -> TokenClaims:
        """Raise AuthenticationError if the token is invalid or expired."""
        ...
