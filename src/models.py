from dataclasses import dataclass
import numpy as np

@dataclass
class Document:
    """
    Represents a structured product document used in the RAG pipeline.
    Serves as the source of truth for raw product information.
    """
    document_id: int
    name: str
    category: str
    features: list[str]
    specifications: dict[str, str]
    description: str

    def to_text(self) -> str:
        features_text = "\n".join(f"- {feature}" for feature in self.features)
        specifications_text = "\n".join(f"{key}: {value}" for key, value in self.specifications.items())
        return f"Product Name: {self.name}\nCategory: {self.category}\nFeatures:\n{features_text}\nSpecifications:\n{specifications_text}\nDescription: {self.description}"

    def get_sections(self) -> dict[str, str]:
        """Returns the document organized into semantic sections for field-aware chunking."""
        features_text = "\n".join(f"- {feature}" for feature in self.features)
        specifications_text = "\n".join(f"{key}: {value}" for key, value in self.specifications.items())
        return {
            "identity": f"Product Name: {self.name}\nCategory: {self.category}",
            "features": features_text,
            "specifications": specifications_text,
            "description": self.description
        }        

@dataclass
class Chunk:
    """Represents a retrievable chunk generated from a product document."""
    chunk_id: str
    document_id: int
    text: str
    metadata: dict[str, str]
    chunk_index: int

@dataclass
class EmbeddingRecord:
    """Stores a chunk and its corresponding dense embedding vector."""
    chunk_id: str
    embedding: np.ndarray
    metadata: dict