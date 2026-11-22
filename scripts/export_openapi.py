"""Write the API contract to docs/openapi.json (AGENTS.md section 10).

    python -m scripts.export_openapi

Builds the app with placeholder settings: building it never connects to the database.
"""

import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.main import build_app

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"


def build_schema() -> dict[str, Any]:
    settings = Settings(
        database_url="postgresql+psycopg://placeholder@127.0.0.1:1/none",
        jwt_secret="x" * 32, admin_email="admin@example.com", admin_password="placeholder",
        _env_file=None,  # pyright: ignore[reportCallIssue]
    )
    return build_app(settings).openapi()


def main() -> None:
    OUTPUT.write_text(json.dumps(build_schema(), indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
