"""Alembic must accept database URLs whose password is percent-encoded."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_offline_migration_accepts_percent_encoded_password():
    url = "postgresql+psycopg://user:p%40ss%23word@localhost:5432/startups"

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=ROOT, env={**os.environ, "DATABASE_URL": url}, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr[-300:]
    assert "CREATE TABLE funding_rounds" in result.stdout
