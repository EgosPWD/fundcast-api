"""docs/openapi.json is the published contract (AGENTS.md section 10): it must match the API."""

import json
from pathlib import Path

from scripts.export_openapi import build_schema

DOCS = Path(__file__).resolve().parents[2] / "docs" / "openapi.json"


def test_published_openapi_matches_the_running_api():
    assert DOCS.is_file(), "run: python -m scripts.export_openapi"
    assert json.loads(DOCS.read_text(encoding="utf-8")) == build_schema(), (
        "docs/openapi.json is stale; run: python -m scripts.export_openapi"
    )


def test_every_422_declares_the_string_error_body_the_api_returns():
    # The global validation handler always answers {"detail": "<message>"}; FastAPI's
    # default 422 schema (HTTPValidationError, a list) would mislead clients.
    schema = build_schema()
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            response = operation["responses"].get("422")
            if response is not None:
                ref = response["content"]["application/json"]["schema"]["$ref"]
                assert ref.endswith("/ErrorResponse"), f"{method.upper()} {path}: {ref}"
    assert "HTTPValidationError" not in schema["components"]["schemas"]
