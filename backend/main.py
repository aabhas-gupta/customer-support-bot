from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.database import create_tables, get_db, Conversation, Message, Escalation
from backend.schemas import (
    ChatRequest, ChatResponse, SourceChunk,
    ConversationOut, EscalationOut, ResolveRequest, UploadResponse
)
from backend.rag import SupportRAG


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs at startup.
    Code after `yield` runs at shutdown.
    This is the modern FastAPI way to initialize shared resources.
    """
    create_tables()                  # create SQLite tables if they don't exist
    app.state.rag = SupportRAG()     # load RAG pipeline once (expensive — do it once)
    print("[API] ShopNow Support Bot is ready")
    yield
    print("[API] Shutting down")


app = FastAPI(
    title="ShopNow Customer Support API",
    version="1.0.0",
    lifespan=lifespan
)

# Allow Streamlit (port 8501) to call this API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "ShopNow Support Bot"}


# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Main customer-facing endpoint.
    - Gets or creates a conversation for the session_id
    - Stores the user's message
    - Queries the RAG pipeline
    - Stores the assistant's response
    - Creates an escalation record if confidence is low
    """
    rag: SupportRAG = app.state.rag

    # Get or create the conversation record
    conversation = db.query(Conversation).filter(
        Conversation.id == request.session_id
    ).first()

    if not conversation:
        conversation = Conversation(id=request.session_id)
        db.add(conversation)
        db.commit()

    # Store the user's message
    user_msg = Message(
        conversation_id=request.session_id,
        role="user",
        content=request.message
    )
    db.add(user_msg)
    db.commit()

    # Query the RAG pipeline
    result = rag.query(request.message)

    # Store the assistant's response with confidence score
    assistant_msg = Message(
        conversation_id=request.session_id,
        role="assistant",
        content=result["answer"],
        confidence=result["confidence"]
    )
    db.add(assistant_msg)

    # If low confidence → create escalation and flag the conversation
    if result["escalate"] and not conversation.is_escalated:
        conversation.is_escalated = True
        escalation = Escalation(
            conversation_id=request.session_id,
            reason=result["escalation_reason"] or "low_confidence"
        )
        db.add(escalation)

    db.commit()

    return ChatResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        escalated=result["escalate"],
        escalation_reason=result["escalation_reason"],
        sources=[SourceChunk(**s) for s in result["sources"]]
    )


# ── Admin: upload document ────────────────────────────────────────────────────
@app.post("/admin/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Admin endpoint to add a new document to the knowledge base.
    Accepts: .txt, .pdf, .docx, .csv
    """
    rag: SupportRAG = app.state.rag

    allowed = {".txt", ".pdf", ".docx", ".csv"}
    suffix = "." + file.filename.split(".")[-1].lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {suffix} not supported.")

    content = await file.read()
    chunks_added = rag.add_document(content, file.filename)

    return UploadResponse(
        filename=file.filename,
        chunks_added=chunks_added,
        message=f"Successfully indexed '{file.filename}' ({chunks_added} chunks added)"
    )


# ── Admin: list all conversations ─────────────────────────────────────────────
@app.get("/admin/conversations", response_model=list[ConversationOut])
def get_conversations(db: Session = Depends(get_db)):
    """Returns all conversations ordered by most recent first."""
    return db.query(Conversation).order_by(Conversation.created_at.desc()).all()


# ── Admin: list escalations ───────────────────────────────────────────────────
@app.get("/admin/escalations", response_model=list[EscalationOut])
def get_escalations(resolved: bool = False, db: Session = Depends(get_db)):
    """
    Returns escalations. Pass ?resolved=true to see resolved ones.
    Default: only unresolved escalations.
    """
    return (
        db.query(Escalation)
        .filter(Escalation.resolved == resolved)
        .order_by(Escalation.created_at.desc())
        .all()
    )


# ── Admin: resolve an escalation ─────────────────────────────────────────────
@app.post("/admin/resolve/{escalation_id}", response_model=EscalationOut)
def resolve_escalation(
    escalation_id: int,
    body: ResolveRequest,
    db: Session = Depends(get_db)
):
    """Mark an escalation as resolved and optionally add admin notes."""
    escalation = db.query(Escalation).filter(Escalation.id == escalation_id).first()
    if not escalation:
        raise HTTPException(status_code=404, detail="Escalation not found")

    escalation.resolved = True
    escalation.resolved_at = datetime.utcnow()
    escalation.admin_notes = body.admin_notes

    # Also mark the parent conversation as resolved
    conversation = db.query(Conversation).filter(
        Conversation.id == escalation.conversation_id
    ).first()
    if conversation:
        conversation.resolved = True

    db.commit()
    db.refresh(escalation)
    return escalation
