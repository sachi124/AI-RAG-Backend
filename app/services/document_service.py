from uuid import uuid4
from sqlalchemy.orm import Session
from app.config import get_settings
from app.models.db_models import DocumentMetadata
from app.services.chunking import split_text
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore


class DocumentService:
    def __init__(self, llm: LLMService, vector_store: VectorStore) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.settings = get_settings()

    def ingest(
        self, db: Session, filename: str, content_type: str, text: str, strategy: str
    ) -> tuple[str, int]:
        if not text.strip():
            raise ValueError("No extractable text found in the uploaded document")
        chunks = split_text(text, strategy, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("Document produced no text chunks")
        embeddings = self.llm.embed_texts(chunks)
        document_id = str(uuid4())

        # Store vectors first. Save metadata only after successful vector upsert.
        self.vector_store.upsert_chunks(document_id, filename, chunks, embeddings)
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            chunking_strategy=strategy,
            chunk_count=len(chunks),
        )
        db.add(metadata)
        db.commit()
        return document_id, len(chunks)
