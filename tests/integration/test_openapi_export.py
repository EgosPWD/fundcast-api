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
