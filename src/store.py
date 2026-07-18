from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.models import Chunk, EmbeddingRecord

class ChunkRepository:
    """Manages raw Chunk objects independently of their vectors."""
    def __init__(self):
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunk: Chunk) -> None:
        if chunk.chunk_id in self._chunks:
            raise ValueError(f"Chunk with ID '{chunk.chunk_id}' already exists.")
        self._chunks[chunk.chunk_id] = chunk

    def get(self, chunk_id: str) -> Chunk:
        if chunk_id not in self._chunks:
            raise KeyError(f"Chunk with ID '{chunk_id}' not found.")
        return self._chunks[chunk_id]

    def delete(self, chunk_id: str) -> None:
        if chunk_id not in self._chunks:
            raise KeyError(f"Chunk with ID '{chunk_id}' not found.")
        del self._chunks[chunk_id]

    def get_all(self) -> list[Chunk]:
        return list(self._chunks.values())

class BaseVectorStore(ABC):
    """Abstract base class for all vector storage engines."""
    @abstractmethod
    def add(self, record: EmbeddingRecord) -> None: pass
    @abstractmethod
    def add_all(self, records: list[EmbeddingRecord]) -> None: pass
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5, filters: dict | None = None) -> list[tuple[str, float]]: pass
    @abstractmethod
    def delete(self, chunk_id: str) -> None: pass
    @abstractmethod
    def clear(self) -> None: pass

class InMemoryVectorStore(BaseVectorStore):
    """Brute-force vector engine running basic cosine similarities in-memory."""
    def __init__(self):
        self._records: dict[str, EmbeddingRecord] = {}

    def add(self, record: EmbeddingRecord) -> None:
        if record.chunk_id in self._records:
            raise ValueError(f"Chunk with ID '{record.chunk_id}' already exists.")
        self._records[record.chunk_id] = record

    def add_all(self, records: list[EmbeddingRecord]) -> None:
        for record in records:
            self.add(record)

    def search(self, query_embedding: np.ndarray, top_k: int = 5, filters: dict | None = None) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if query_embedding.size == 0:
            raise ValueError("Query embedding cannot be empty.")
        
        results: list[tuple[str, float]] = []
        for record in self._records.values():
            if filters:
                if any(record.metadata.get(key) != value for key, value in filters.items()):
                    continue
            similarity = cosine_similarity(query_embedding.reshape(1, -1), record.embedding.reshape(1, -1))[0][0]
            results.append((record.chunk_id, float(similarity)))
        
        results.sort(key=lambda result: result[1], reverse=True)
        return results[:top_k]

    def delete(self, chunk_id: str) -> None:
        if chunk_id not in self._records:
            raise KeyError(f"Chunk ID '{chunk_id}' not found.")
        del self._records[chunk_id]

    def clear(self) -> None:
        self._records.clear()