import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent if "__file__" in locals() else Path.cwd()))

from src import (
    SyntheticDatasetGenerator,
    FieldAwareChunker,
    ChunkRepository,
    InMemoryVectorStore,
    DefaultContextManager,
    RAGPipeline
)
from sentence_transformers import SentenceTransformer

class LocalTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed(self, texts: list[str]):
        return self.model.encode(texts, convert_to_numpy=True)

def run_integration_test():
    print("🚀 Initializing component integration checks...")
    

    generator = SyntheticDatasetGenerator()
    chunker = FieldAwareChunker()
    repository = ChunkRepository()
    vector_store = InMemoryVectorStore()
    context_manager = DefaultContextManager(context_budget=512)
    embedder = LocalTransformerEmbedder()
    
    pipeline = RAGPipeline(
        chunker=chunker,
        embedder=embedder,
        repository=repository,
        vector_store=vector_store,
        context_manager=context_manager
    )
    print("✅ Architecture layout bound successfully.")

    print("📦 Generating mock product evaluations...")
    sample_docs = generator.generate_dataset(10) 
    
    print("⚡ Indexing vectors and chunk metadata...")
    pipeline.index_documents(sample_docs)
    print(f"✅ Pipeline populated. Repository count: {len(repository.get_all())} chunks.")

   
    test_query = "Gaming laptop with RTX 4080 graphics"
    print(f"🔍 Dispatched Semantic Query: '{test_query}'")
    
    compiled_prompt = pipeline.retrieve(test_query, top_k=3)
    
    print("\n================== GENERATED PROMPT OUTPUT ==================\n")
    print(compiled_prompt)
    print("\n=============================================================\n")
    print("🎉 Refactoring Successful! System works end-to-end without breaks.")

if __name__ == "__main__":
    run_integration_test()