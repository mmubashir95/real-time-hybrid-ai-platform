from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Identity,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        CheckConstraint(
            "latency_ms >= 0", name="ck_analyses_latency_ms_nonnegative"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    message: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(100))
    urgency: Mapped[str] = mapped_column(String(50))
    summary: Mapped[str] = mapped_column(Text)
    # JSONB preserves the structured entity object and supports later querying.
    entities: Mapped[dict[str, str]] = mapped_column(JSONB)
    sentiment: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    latency_ms: Mapped[int] = mapped_column(Integer)
