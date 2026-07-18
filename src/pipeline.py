import numpy as np
from src.models import Document, Chunk, EmbeddingRecord
from src.chunkers import BaseChunker
from src.store import ChunkRepository, BaseVectorStore
from src.context import BaseContextManager

class RAGPipeline:
    """
    Coordinates the complete end-to-end modular retrieval pipeline.
    Connects the chunker, embedder, repository, vector store, and context manager.
    """
    def __init__(
        self,
        chunker: BaseChunker,
        embedder: object,  # Accepts any compliant embedding generator instance
        repository: ChunkRepository,
        vector_store: BaseVectorStore,
        context_manager: BaseContextManager
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.repository = repository
        self.vector_store = vector_store
        self.context_manager = context_manager

    def index_documents(self, documents: list[Document]) -> None:
        """
        Processes and indexes a collection of documents into the RAG framework.
        """
        if not documents:
            raise ValueError("Document list cannot be empty.")
        
        all_chunks: list[Chunk] = []
        for document in documents:
            chunks = self.chunker.chunk(document)
            all_chunks.extend(chunks)
            
        for chunk in all_chunks:
            self.repository.add(chunk)
            
        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.embed(chunk_texts)

        embedding_records: list[EmbeddingRecord] = []
        for chunk, embedding in zip(all_chunks, embeddings):
            embedding_records.append(
                EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    embedding=embedding,
                    metadata=chunk.metadata
                )
            )
        self.vector_store.add_all(embedding_records)

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None) -> str:
        """
        Retrieves relevant context blocks and builds the final optimized prompt.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")
            
        query_embedding = self.embedder.embed([query])[0]

        # Fetch a broader candidate pool to allow the multi-pass selector to balance diversity
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=max(top_k * 4, 20),
            filters=filters
        )
        
        prompt = self.context_manager.build_context(
            search_results=search_results,
            chunk_repository=self.repository,
            query=query
        )
        return prompt