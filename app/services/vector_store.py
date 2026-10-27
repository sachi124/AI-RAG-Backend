from typing import Any
from uuid import uuid4
from qdrant_client import QdrantClient, models
from app.config import get_settings


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection_name = settings.qdrant_collection
        self.vector_size = settings.embedding_dimensions
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {item.name for item in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
            )
            return
        info = self.client.get_collection(self.collection_name)
        existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
        if existing_size != self.vector_size:
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' has vector size {existing_size}, "
                f"but EMBEDDING_DIMENSIONS is {self.vector_size}. "
                "Use a new QDRANT_COLLECTION name in .env."
            )

    def upsert_chunks(
        self, document_id: str, filename: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding")
        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "text": chunk,
                },
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            with_payload=True,
        ).points
        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "score": float(hit.score),
                "text": str(payload.get("text", "")),
                "document_id": str(payload.get("document_id", "")),
                "filename": str(payload.get("filename", "")),
                "chunk_index": int(payload.get("chunk_index", 0)),
            })
        return results


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
