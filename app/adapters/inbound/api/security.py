"""Bearer token parsing and role checks (JWT in `Authorization: Bearer <token>`)."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.adapters.inbound.api import deps
from app.application.ports.token_service import TokenClaims, TokenService
from app.domain.entities import Role
from app.domain.errors import AuthenticationError, PermissionDeniedError

# auto_error=False: a missing or non-Bearer header becomes AuthenticationError (401 with
# {"detail": ...}) instead of FastAPI's default 403.
bearer_scheme = HTTPBearer(auto_error=False, description="JWT returned by POST /auth/login")


def current_claims(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    tokens: Annotated[TokenService, Depends(deps.token_service)],
) -> TokenClaims:
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Not authenticated")
    return tokens.decode(credentials.credentials)


def require_role(*roles: Role) -> Callable[[TokenClaims], TokenClaims]:
    allowed = frozenset(roles)
    message = " or ".join(r.value.capitalize() for r in roles) + " role required"

    def dependency(claims: Annotated[TokenClaims, Depends(current_claims)]) -> TokenClaims:
        if claims.role not in allowed:
            raise PermissionDeniedError(message)
        return claims

    return dependency


AnyUser = Annotated[TokenClaims, Depends(require_role(Role.USER, Role.ADMIN))]
AdminUser = Annotated[TokenClaims, Depends(require_role(Role.ADMIN))]
