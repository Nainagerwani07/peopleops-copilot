"""add ai_audit_logs

Revision ID: 0017_add_ai_audit_logs
Revises: 0016_employee_documents_uploaded_by
Create Date: 2026-10-05
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0017_add_ai_audit_logs"
down_revision: Union[str, None] = "0016_employee_documents_uploaded_by"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ai_audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(50), nullable=True),
        sa.Column("tool_name", sa.String(100), nullable=True),
        sa.Column("action_status", sa.String(30), nullable=False),
        sa.Column("records_accessed", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    for column in ("id", "user_id", "intent", "action_status", "created_at"):
        op.create_index(f"ix_ai_audit_logs_{column}", "ai_audit_logs", [column])


def downgrade() -> None:
    for column in ("id", "user_id", "intent", "action_status", "created_at"):
        op.drop_index(f"ix_ai_audit_logs_{column}", table_name="ai_audit_logs")
    op.drop_table("ai_audit_logs")
