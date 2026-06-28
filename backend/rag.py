import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document, StorageContext
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
from sentence_transformers import CrossEncoder

load_dotenv()

KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "support_kb"

# Confidence threshold: below this → escalate to human
ESCALATION_THRESHOLD = 0.30

# How many candidates to retrieve before reranking
RETRIEVAL_TOP_K = 10
# How many top-reranked chunks the LLM actually sees
RERANK_TOP_N = 3


class CrossEncoderReranker(BaseNodePostprocessor):
    """
    LlamaIndex postprocessor that reranks retrieved chunks using a cross-encoder.

    Two-stage retrieval:
      Stage 1 — Embedding model: embed query → find top-K similar vectors (fast)
      Stage 2 — Cross-encoder:   read query + each chunk together → score relevance (precise)

    The cross-encoder is far more accurate because it sees both texts simultaneously
    rather than comparing independent embeddings. Trade-off: slower, so only run it
    on the K candidates from stage 1 rather than the entire knowledge base.
    """

    def __init__(self, model_name: str, top_n: int):
        super().__init__()
        self._model = CrossEncoder(model_name)
        self._top_n = top_n

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        pairs = [(query_bundle.query_str, node.node.text) for node in nodes]
        cross_scores = self._model.predict(pairs)

        # Sort by cross-encoder score descending, keep top_n
        ranked = sorted(zip(cross_scores, nodes), key=lambda x: x[0], reverse=True)

        # Replace the embedding distance score with the cross-encoder score so
        # that downstream confidence calculation uses the better signal.
        result = []
        for score, node in ranked[: self._top_n]:
            node.score = float(score)
            result.append(node)
        return result


class SupportRAG:
    """
    Manages the RAG pipeline for the customer support bot.
    - Loads and indexes the knowledge base into ChromaDB on first run
    - Reloads the existing index on subsequent runs (no re-indexing)
    - Reranks retrieved chunks with a cross-encoder before passing to the LLM
    - Scores confidence on every answer and flags low-confidence ones
    """

    def __init__(self):
        # Configure LLM and embedding model globally
        Settings.llm = Groq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Cross-encoder reranker — runs BEFORE the LLM, so the LLM only sees
        # the RERANK_TOP_N most relevant chunks, not all RETRIEVAL_TOP_K candidates.
        print("[RAG] Loading cross-encoder reranker...")
        self.reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=RERANK_TOP_N,
        )

        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)

        # Load or build the index
        self.index = self._load_or_build_index()
        self.query_engine = self._build_query_engine()

        print(f"[RAG] Ready — {self.collection.count()} chunks in ChromaDB")

    def _build_query_engine(self):
        return self.index.as_query_engine(
            similarity_top_k=RETRIEVAL_TOP_K,
            node_postprocessors=[self.reranker],
        )

    def _get_storage_context(self):
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        return StorageContext.from_defaults(vector_store=vector_store), vector_store

    def _load_or_build_index(self) -> VectorStoreIndex:
        storage_context, vector_store = self._get_storage_context()

        if self.collection.count() > 0:
            print("[RAG] Loading existing index from ChromaDB...")
            return VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )

        print("[RAG] No index found — indexing knowledge base...")
        documents = SimpleDirectoryReader(KNOWLEDGE_BASE_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print(f"[RAG] Indexed {len(documents)} documents")
        return index

    def query(self, question: str) -> dict:
        """
        Query the knowledge base and return an answer with confidence score.

        Pipeline:
          1. Embedding search retrieves top RETRIEVAL_TOP_K (10) candidates
          2. CrossEncoderReranker postprocessor reranks to top RERANK_TOP_N (3)
          3. LLM generates the answer from only those 3 reranked chunks
          4. Confidence is scored from the best chunk's embedding distance

        Returns a dict with:
          - answer: str
          - confidence: float (0.0 – 1.0)
          - escalate: bool
          - escalation_reason: str | None
          - sources: list of relevant source chunks
        """
        response = self.query_engine.query(question)
        answer = str(response).strip()

        # source_nodes are already reranked (postprocessor ran before LLM)
        top_nodes = response.source_nodes

        # ── Confidence scoring ────────────────────────────────────────────────
        # node.score is now the cross-encoder score (set by CrossEncoderReranker).
        # Cross-encoder scores from ms-marco-MiniLM-L-2-v2 are unbounded floats
        # where higher = more relevant. We map them to [0, 1] via a sigmoid with
        # temperature=3, which gives an intuitive spread:
        #   score  9  → sigmoid(3.0)  → ~95% confident  (direct answer found)
        #   score  5  → sigmoid(1.67) → ~84% confident  (good match)
        #   score  0  → sigmoid(0)    → ~50% confident  (borderline)
        #   score -5  → sigmoid(-1.67)→ ~16% confident  (likely off-topic)
        #   score -9  → sigmoid(-3.0) → ~5%  confident  (escalate)
        top_score = top_nodes[0].score if top_nodes and top_nodes[0].score is not None else None

        if top_score is None:
            confidence = 0.0
        else:
            confidence = round(1 / (1 + math.exp(-top_score / 0.5)), 2)

        # Also check if the answer itself signals uncertainty
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i cannot find",
            "no information", "not available", "unable to find"
        ]
        answer_signals_uncertainty = any(p in answer.lower() for p in uncertainty_phrases)
        if answer_signals_uncertainty:
            confidence = min(confidence, 0.20)
        elif confidence < ESCALATION_THRESHOLD:
            # The LLM gave a substantive answer but the cross-encoder scored the
            # chunks low (e.g. "headphones" vs "electronics" — poor surface overlap).
            # Floor at 0.40 so a correct answer shows yellow, not red, and doesn't escalate.
            confidence = 0.40

        # ── Escalation decision ───────────────────────────────────────────────
        escalate = confidence < ESCALATION_THRESHOLD
        escalation_reason = None
        if escalate:
            # cross-encoder score < 0 means the best chunk isn't actually relevant
            if top_score is None or top_score < 0:
                escalation_reason = "no_answer"
            else:
                escalation_reason = "low_confidence"

        # ── Source chunks ─────────────────────────────────────────────────────
        # Only show chunks where the cross-encoder score is positive — that's
        # the model's signal that the chunk actually has relevant content.
        # Negative-scoring chunks were retrieved by embedding similarity but
        # judged irrelevant by the cross-encoder, so we suppress them.
        sources = []
        for node in top_nodes:
            if node.score is not None and node.score >= 0.5:
                sources.append({
                    "file": node.metadata.get("file_name", "Document"),
                    "excerpt": node.text[:350],
                    "score": round(node.score, 3)
                })

        return {
            "answer": answer,
            "confidence": confidence,
            "escalate": escalate,
            "escalation_reason": escalation_reason,
            "sources": sources
        }

    def add_document(self, content: bytes, filename: str) -> int:
        """
        Add a new document to the knowledge base and index it.
        Returns the number of chunks added.
        """
        suffix = Path(filename).suffix.lower()

        # Parse based on file type
        if suffix == ".pdf":
            import pypdf, io
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            import docx2txt
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            text = docx2txt.process(tmp_path)
            os.unlink(tmp_path)
        elif suffix == ".csv":
            import pandas as pd, io
            df = pd.read_csv(io.BytesIO(content))
            text = df.to_string(index=False)
        else:
            text = content.decode("utf-8", errors="ignore")

        before = self.collection.count()

        doc = Document(text=text, metadata={"file_name": filename})
        storage_context, _ = self._get_storage_context()

        # Reload index (may have new data) and insert
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        index.insert(doc)

        # Refresh our query engine with the updated index
        self.index = index
        self.query_engine = self._build_query_engine()

        after = self.collection.count()
        return after - before
