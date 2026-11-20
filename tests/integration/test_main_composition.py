"""The composition root wires every endpoint without touching the database at build time."""

from app.config import Settings
from app.main import build_app


def settings(tmp_path) -> Settings:
    return Settings(
        database_url="postgresql+psycopg://u:p@127.0.0.1:1/none",
        jwt_secret="x" * 32,
        admin_email="admin@example.com",
        admin_password="secret123",
        artifacts_dir="./artifacts",
        uploads_dir=str(tmp_path / "uploads"),
        cors_origins=["http://localhost:5173"],
        _env_file=None,  # pyright: ignore[reportCallIssue]
    )


def test_build_app_exposes_every_contract_route(tmp_path):
    app = build_app(settings(tmp_path))

    # FastAPI 0.165 nests included routers; the OpenAPI schema lists every operation.
    paths = app.openapi()["paths"]
    routes = {(method.upper(), path) for path, ops in paths.items() for method in ops}

    assert {
        ("POST", "/auth/register"), ("POST", "/auth/login"), ("POST", "/predict"),
        ("GET", "/predictions"), ("GET", "/metrics"), ("POST", "/training-files"),
        ("POST", "/retrain"), ("GET", "/health"),
    } <= routes
