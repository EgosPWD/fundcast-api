"""Build the FastAPI app from already-wired use cases (the composition root lives in main)."""

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.adapters.inbound.api.errors import register_error_handlers
from app.adapters.inbound.api.routers import auth, health, metrics, predict, training
from app.adapters.inbound.api.use_case_ports import (
    GetMetricsUseCase,
    ListPredictionsUseCase,
    LoginUseCase,
    PredictRoundUseCase,
    RegisterUserUseCase,
    RetrainModelUseCase,
    UploadTrainingFileUseCase,
)
from app.application.ports.token_service import TokenService


@dataclass(frozen=True, slots=True)
class ApiUseCases:
    register_user: RegisterUserUseCase
    login: LoginUseCase
    predict_round: PredictRoundUseCase
    list_predictions: ListPredictionsUseCase
    get_metrics: GetMetricsUseCase
    upload_training_file: UploadTrainingFileUseCase
    retrain_model: RetrainModelUseCase
    token_service: TokenService


def create_app(
    use_cases: ApiUseCases,
    cors_origins: Sequence[str],
    on_startup: Callable[[], None] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if on_startup is not None:
            on_startup()
        yield

    app = FastAPI(
        title="Startup Funding API",
        version="0.1.0",
        description="Estimates the USD amount of a startup funding round.",
        lifespan=lifespan,
    )
    app.state.use_cases = use_cases
    # Bearer tokens travel in a header, not cookies: no credentials mode is needed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    register_error_handlers(app)
    for module in (health, auth, predict, metrics, training):
        app.include_router(module.router)
    return app
