# Modular RAG Architecture from Scratch

A production-grade, highly decoupled Retrieval-Augmented Generation (RAG) framework implemented completely from scratch in Python. This system replaces rigid, monolithic RAG scripts with highly modular, isolated components—utilizing Abstract Base Classes (ABCs) to cleanly separate dataset generation, field-aware document chunking, multi-pass retrieval selection, and token-constrained context management.

The architecture is designed to address common failure modes in baseline RAG pipelines, such as semantic context fragmentation, redundant background data crowding, and out-of-boundary prompt token overflows.

---

## Key Features

* **Synthetic Dataset Generator:** Programmatically constructs a structured, relational hardware and specification corpus to provide an objective testbed for semantic retrieval tracking.
* **Field-Aware Semantic Chunker:** Abandons blind character-threshold splitting. It parses documents according to structural boundaries (Identity, Category, Description, and Features), ensuring related technical specifications remain cohesive within the same text block.
* **Multi-Pass Informative Selector:** Evaluates raw vector similarity scores alongside structural catalog diversity. This matrix ensures that the retrieved background context contains diverse product perspectives rather than single-source clusters.
* **Deduplication Filter Matrix:** Dynamically processes overlapping chunk returns to discard redundant semantic fragments before compiling the final payload.
* **Token Budget Manager:** Integrates Hugging Face tokenizers to programmatically monitor and clip text boundaries, completely eliminating downstream LLM prompt overflow risks.
* **Abstract Interface Blueprinting:** Every system layer is built on rigid Python abstraction rules, making it trivial to swap out the custom In-Memory vector store for external databases without altering core orchestration logic.

---

## Tech Stack

* **Language:** Python 3.11+
* **Vector Math Engine:** NumPy, Scikit-Learn
* **Tokenization & Embedding Models:** Hugging Face `transformers` (`AutoTokenizer`), `sentence-transformers`

---

## System Architecture & Processing Pipeline

The pipeline breaks down the classic ingestion and retrieval loop into distinct execution layers to maintain clean state isolation across the lifecycle:

1. **Synthetic Dataset Generation:** The structural corpus is assembled and verified for baseline semantic evaluation.
2. **Field-Aware Document Parsing:** Raw data is split using contextual metadata and field boundaries instead of character counts.
3. **Vector Embedding Conversion:** Chunks are passed through semantic models to generate localized embeddings vector spaces.
4. **Abstract Vector Indexing:** Embeddings are mapped to an interface-driven vector store layout.
5. **Multi-Pass Informative Retrieval:** The search engine sorts records through similarity metrics combined with catalog diversity matrices.
6. **Dynamic Token-Budget Enforcement:** Text structures are dynamically evaluated and clipped using tokenizers.
7. **Instruction-Tuned Prompt Wrapper:** Clean context blocks are compiled into strict execution strings for generation.

---

## Core Code Implementation

### Quick Usage Example

```python
from src.pipeline import RAGPipeline

# Initialize the modular pipeline & ingest the knowledge base
pipeline = RAGPipeline()
pipeline.initialize_knowledge_base()

# Execute an intent-driven structured hardware specification query
prompt = pipeline.retrieve("Gaming laptop with RTX 4080 graphics, 32GB RAM, and 2TB SSD")
print(prompt)

```

---

## Context Benchmarking & Alignment Analysis

* **Semantic Integrity Protection:** By shifting from fixed-size text splitting to field-aware grouping, the pipeline keeps tightly coupled attributes (like RAM, GPU model, and storage specs) in a single block. This completely eliminates instances where critical constraints are cut off mid-sentence.
* **Prompt Optimization Matrix:** Combining deduplication logic with strict token budgeting minimizes empty context filler. The downstream LLM receives a highly dense, unique, and instruction-tuned prompt context wrapper that maximizes retrieval accuracy.

---

## System Verification & Output Proof

When a complex hardware query is dispatched, the retrieval engine builds an instruction-wrapped prompt using only the deduplicated, highly accurate matching specifications found within token bounds. The pipeline safely drops overlapping references or lower-ranked duplicate data, optimizing the attention window for precise execution answers.

---

## Author

Vijay B V