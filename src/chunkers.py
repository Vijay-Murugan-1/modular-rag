from abc import ABC, abstractmethod
from src.models import Document, Chunk

class BaseChunker(ABC):
    """Abstract Base Class for all chunking strategies."""
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split the document into retrievable chunks."""
        pass

class FixedSizeChunker(BaseChunker):
    """Splits a document using fixed-size characters with overlapping windows."""
    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.to_text()
        chunks = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunk = Chunk(
                chunk_id=f"{document.document_id}_{chunk_index}",
                document_id=document.document_id,
                text=chunk_text,
                metadata={"category": document.category, "product_name": document.name},
                chunk_index=chunk_index
            )
            chunks.append(chunk)
            if end == len(text):
                break
            start = end - self.chunk_overlap
            chunk_index += 1
        return chunks

class FieldAwareChunker(BaseChunker):
    """Splits a structured product document into semantically meaningful chunks based on fields."""
    def chunk(self, document: Document) -> list[Chunk]:
        sections = document.get_sections()
        chunks = []
        for index, (section_name, section_text) in enumerate(sections.items()):
            chunk_text = (
                f"Product Name: {document.name}\n"
                f"Category: {document.category}\n\n"
                f"{section_name.title()}:\n\n" 
                f"{section_text}"
            )            
            chunk = Chunk(
                chunk_id=f"{document.document_id}_{index}",
                document_id=document.document_id,
                text=chunk_text,
                metadata={
                    "category": document.category,
                    "section": section_name,
                    "product_name": document.name
                },
                chunk_index=index
            )
            chunks.append(chunk)
        return chunks