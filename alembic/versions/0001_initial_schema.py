"""initial schema: users, training_files, model_versions, predictions

Revision ID: 0001
Revises:
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.CheckConstraint("role IN ('user', 'admin')", name="ck_users_role"),
    )
    op.create_table(
        "training_files",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("stored_path", sa.String(1024), nullable=False),
        sa.Column("rows", sa.Integer, nullable=False),
        sa.Column("uploaded_by", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.CheckConstraint("status IN ('uploaded', 'used', 'failed')",
                           name="ck_training_files_status"),
    )
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("version", sa.Integer, nullable=False, unique=True),
        sa.Column("artifact_path", sa.String(1024), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("mae_log", sa.Float, nullable=False),
        sa.Column("r2", sa.Float, nullable=False),
        sa.Column("rmse_log", sa.Float, nullable=False),
        sa.Column("mdape", sa.Float, nullable=False),
        sa.Column("coverage", sa.Float, nullable=False),
        sa.Column("n_train", sa.Integer, nullable=False),
        sa.Column("n_test", sa.Integer, nullable=False),
        sa.Column("params", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("training_file_id", sa.Integer, sa.ForeignKey("training_files.id"),
                  nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
    )
    op.create_index("uq_model_versions_single_active", "model_versions", ["is_active"],
                    unique=True, postgresql_where=sa.text("is_active"))
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("model_version_id", sa.Integer, sa.ForeignKey("model_versions.id"),
                  nullable=False),
        sa.Column("input", postgresql.JSONB, nullable=False),
        sa.Column("estimated_amount_usd", sa.Float, nullable=False),
        sa.Column("estimated_log10", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
    )
    op.create_index("ix_predictions_user_id_created_at", "predictions",
                    ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_predictions_user_id_created_at", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("uq_model_versions_single_active", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_table("training_files")
    op.drop_table("users")
