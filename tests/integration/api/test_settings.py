"""Settings come from environment variables (AGENTS.md section 8)."""

import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = {"JWT_SECRET": "a" * 32, "ADMIN_EMAIL": "admin@example.com",
            "ADMIN_PASSWORD": "admin-password",
            "DATABASE_URL": "postgresql+psycopg://u:p@db:5432/startups"}


@pytest.fixture
def env(monkeypatch):
    for name in ("DB_PASSWORD", "DATABASE_URL", "JWT_SECRET", "JWT_ALGORITHM",
                 "JWT_EXPIRE_MINUTES", "ADMIN_EMAIL", "ADMIN_PASSWORD", "ARTIFACTS_DIR",
                 "UPLOADS_DIR", "CORS_ORIGINS", "SEED_DATASET_PATH"):
        monkeypatch.delenv(name, raising=False)
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def load() -> Settings:
    return Settings(_env_file=None)


def test_reads_every_variable_from_the_environment(env):
    env.setenv("DB_PASSWORD", "pw")
    env.setenv("JWT_ALGORITHM", "HS512")
    env.setenv("JWT_EXPIRE_MINUTES", "15")
    env.setenv("ARTIFACTS_DIR", "/srv/artifacts")
    env.setenv("UPLOADS_DIR", "/srv/uploads")
    env.setenv("CORS_ORIGINS", "http://localhost:5173, https://app.example.com ,")
    env.setenv("SEED_DATASET_PATH", "/srv/data/seed.csv")

    settings = load()

    assert settings.db_password == "pw"
    assert settings.database_url == REQUIRED["DATABASE_URL"]
    assert settings.jwt_secret == "a" * 32
    assert settings.jwt_algorithm == "HS512"
    assert settings.jwt_expire_minutes == 15
    assert settings.admin_email == "admin@example.com"
    assert settings.admin_password == "admin-password"
    assert settings.artifacts_dir == "/srv/artifacts"
    assert settings.uploads_dir == "/srv/uploads"
    assert settings.cors_origins == ["http://localhost:5173", "https://app.example.com"]
    assert settings.seed_dataset_path == "/srv/data/seed.csv"


def test_defaults(env):
    settings = load()

    assert settings.jwt_algorithm == "HS256"
    assert settings.jwt_expire_minutes == 60
    assert settings.cors_origins == []
    assert settings.seed_dataset_path is None


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_seed_dataset_path_means_none(env, value):
    env.setenv("SEED_DATASET_PATH", value)

    assert load().seed_dataset_path is None


@pytest.mark.parametrize("missing", ["JWT_SECRET", "ADMIN_EMAIL", "ADMIN_PASSWORD",
                                     "DATABASE_URL"])
def test_secrets_and_connection_have_no_default(env, missing):
    env.delenv(missing)

    with pytest.raises(ValidationError):
        load()


def test_short_jwt_secret_is_rejected(env):
    env.setenv("JWT_SECRET", "too-short")

    with pytest.raises(ValidationError, match="32"):
        load()
