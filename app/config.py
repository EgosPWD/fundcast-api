"""Settings read from environment variables or `.env` (AGENTS.md section 8).

Secrets and the database URL have no default: the service refuses to start without them.
"""

from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

MIN_JWT_SECRET_LENGTH = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
                                      extra="ignore")

    db_password: str = ""
    database_url: str
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    admin_email: str
    admin_password: str
    artifacts_dir: str = "./artifacts"
    uploads_dir: str = "./data/uploads"
    cors_origins: Annotated[list[str], NoDecode] = []
    seed_dataset_path: str | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("seed_dataset_path", mode="before")
    @classmethod
    def _empty_path_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("jwt_secret")
    @classmethod
    def _secret_is_long_enough(cls, value: str) -> str:
        if len(value) < MIN_JWT_SECRET_LENGTH:
            raise ValueError(f"JWT_SECRET must have at least {MIN_JWT_SECRET_LENGTH} characters")
        return value
