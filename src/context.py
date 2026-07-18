from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from src.models import Chunk
from src.store import ChunkRepository

class BaseContextManager(ABC):
    @abstractmethod
    def build_context(self, search_results: list[tuple[str, float]], chunk_repository: ChunkRepository, query: str) -> str:
        pass

class DefaultContextManager(BaseContextManager):
    """Maintains token limitations while ensuring semantic diversity among retrieved contexts."""
    def __init__(self, tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2", context_budget: int = 512):
        if context_budget <= 0:
            raise ValueError("Context budget must be greater than 0.")
        self.context_budget = context_budget
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _select_informative_context(self, search_results: list[tuple[str, float]], chunk_repository: ChunkRepository, top_k: int = 5) -> list[Chunk]:
        chunks = [chunk_repository.get(cid) for cid, _ in search_results]
        selected_chunks: list[Chunk] = []
        seen_products: set[str] = set()
        seen_sections: set[str] = set()

        # PASS 1: Select highest-ranked chunk from each unique product
        for chunk in chunks:
            product = chunk.metadata["product_name"]
            if product not in seen_products:
                selected_chunks.append(chunk)
                seen_products.add(product)
                seen_sections.add(chunk.metadata["section"])
                if len(selected_chunks) == top_k: return selected_chunks

        # PASS 2: Add distinct functional sections
        for chunk in chunks:
            if chunk in selected_chunks: continue
            section = chunk.metadata["section"]
            if section not in seen_sections:
                selected_chunks.append(chunk)
                seen_sections.add(section)
                if len(selected_chunks) == top_k: return selected_chunks

        # PASS 3: Sequential fallback based on standard score order
        for chunk in chunks:
            if chunk in selected_chunks: continue
            selected_chunks.append(chunk)
            if len(selected_chunks) == top_k: break

        return selected_chunks

    def _apply_token_budget(self, chunks: list[Chunk]) -> str:
        context = ""
        current_tokens = 0
        for chunk in chunks:
            token_ids = self.tokenizer.encode(chunk.text, add_special_tokens=False)
            chunk_tokens = len(token_ids)
            remaining_budget = self.context_budget - current_tokens
            
            if remaining_budget <= 0: break
            
            if chunk_tokens <= remaining_budget:
                context += chunk.text + "\n\n"
                current_tokens += chunk_tokens
            else:
                truncated_text = self.tokenizer.decode(token_ids[:remaining_budget], skip_special_tokens=True)
                context += truncated_text + "\n\n"
                break
        return context

    def build_context(self, search_results: list[tuple[str, float]], chunk_repository: ChunkRepository, query: str) -> str:
        if not search_results: raise ValueError("Search results cannot be empty.")
        if not query.strip(): raise ValueError("Query cannot be empty.")
        
        informative_chunks = self._select_informative_context(search_results, chunk_repository)
        context_str = self._apply_token_budget(informative_chunks)
        
        return (
            "You are a helpful AI assistant.\n\n"
            "Use ONLY the provided context to answer the user's question.\n"
            "If the answer is not present in the context, say the information is unavailable.\n\n"
            "Context:\n--------------------------------------------------\n"
            f"{context_str}--------------------------------------------------\n"
            f"Question: {query}\n\nAnswer:"
        )