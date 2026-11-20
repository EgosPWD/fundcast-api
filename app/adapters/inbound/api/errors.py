"""Map errors to HTTP responses. Every error body is {"detail": "<message>"}."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.domain.errors import (
    AuthenticationError,
    DomainError,
    EmailAlreadyRegisteredError,
    InvalidInputError,
    ModelNotAvailableError,
    NotFoundError,
    PermissionDeniedError,
    RetrainInProgressError,
)

logger = logging.getLogger(__name__)

# Matched by base class, first hit wins: subclasses (e.g. InvalidTrainingFileError,
# InsufficientTrainingDataError) inherit their parent's status code.
STATUS_BY_ERROR: tuple[tuple[type[DomainError], int], ...] = (
    (InvalidInputError, 422),
    (AuthenticationError, 401),
    (PermissionDeniedError, 403),
    (NotFoundError, 404),
    (EmailAlreadyRegisteredError, 409),
    (RetrainInProgressError, 409),
    (ModelNotAvailableError, 503),
)
FALLBACK_DOMAIN_STATUS = 400


def status_for(error: DomainError) -> int:
    return next((code for kind, code in STATUS_BY_ERROR if isinstance(error, kind)),
                FALLBACK_DOMAIN_STATUS)


def _detail(status_code: int, message: str, headers: dict[str, str] | None = None):
    return JSONResponse(status_code=status_code, content={"detail": message}, headers=headers)


async def _domain_error(_: Request, error: DomainError) -> JSONResponse:
    status_code = status_for(error)
    headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
    return _detail(status_code, str(error), headers)


def _format_validation(error: RequestValidationError) -> str:
    parts = []
    for item in error.errors():
        location = ".".join(str(p) for p in item.get("loc", ()) if p != "body")
        parts.append(f"{location}: {item.get('msg')}" if location else str(item.get("msg")))
    return "; ".join(parts) or "Invalid request"


async def _validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
    return _detail(422, _format_validation(error))


async def _http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
    return _detail(error.status_code, str(error.detail), getattr(error, "headers", None))


async def _unexpected_error(_: Request, error: Exception) -> JSONResponse:
    logger.exception("Unhandled error", exc_info=error)
    return _detail(500, "Internal server error")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, _domain_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected_error)
