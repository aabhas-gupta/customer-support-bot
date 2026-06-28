from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer,
    Float, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# SQLite database file stored in project root
DATABASE_URL = "sqlite:///./support_bot.db"

# Engine = the connection to the database
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Base class that all table models inherit from
Base = declarative_base()

# SessionLocal = factory for creating DB sessions (one per request)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── Table 1: Conversations ────────────────────────────────────────────────────
class Conversation(Base):
    __tablename__ = "conversations"

    id           = Column(String, primary_key=True)       # UUID from frontend
    created_at   = Column(DateTime, default=datetime.utcnow)
    is_escalated = Column(Boolean, default=False)         # flagged for human review
    resolved     = Column(Boolean, default=False)         # escalation resolved by admin

    # One conversation has many messages
    messages     = relationship("Message", back_populates="conversation",
                                cascade="all, delete-orphan")
    escalations  = relationship("Escalation", back_populates="conversation",
                                cascade="all, delete-orphan")


# ── Table 2: Messages ─────────────────────────────────────────────────────────
class Message(Base):
    __tablename__ = "messages"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role            = Column(String, nullable=False)    # "user" or "assistant"
    content         = Column(Text, nullable=False)
    confidence      = Column(Float, nullable=True)      # only set for assistant messages
    created_at      = Column(DateTime, default=datetime.utcnow)

    conversation    = relationship("Conversation", back_populates="messages")


# ── Table 3: Escalations ──────────────────────────────────────────────────────
class Escalation(Base):
    __tablename__ = "escalations"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    reason          = Column(String, nullable=False)    # "low_confidence" | "no_answer"
    resolved        = Column(Boolean, default=False)
    resolved_at     = Column(DateTime, nullable=True)
    admin_notes     = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    conversation    = relationship("Conversation", back_populates="escalations")


def create_tables():
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Dependency injected into FastAPI routes.
    Yields a DB session and always closes it after the request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
