from typing import Annotated

from fastapi import APIRouter, Depends

from app.adapters.inbound.api import deps
from app.adapters.inbound.api.schemas import (
    ErrorResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.adapters.inbound.api.use_case_ports import LoginUseCase, RegisterUserUseCase

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=201, response_model=UserResponse,
             responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def register(body: RegisterRequest,
             register_user: Annotated[RegisterUserUseCase, Depends(deps.register_user)]):
    """Public sign-up. Always creates role `user`."""
    user = register_user.execute(body.email, body.password)
    return UserResponse(id=user.id, email=user.email, role=user.role.value,
                        created_at=user.created_at)


@router.post("/login", response_model=TokenResponse,
             responses={401: {"model": ErrorResponse}})
def login(body: LoginRequest, login_use_case: Annotated[LoginUseCase, Depends(deps.login)]):
    return TokenResponse(access_token=login_use_case.execute(body.email, body.password))
