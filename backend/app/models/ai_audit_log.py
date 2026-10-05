from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AiAuditLog(Base):
    __tablename__ = "ai_audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)  # redacted before storing
    intent: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    tool_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    action_status: Mapped[str] = mapped_column(String(30), index=True)
    records_accessed: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list of ids
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
