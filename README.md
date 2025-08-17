<div align="center">

# Semantic Code Search & RAG Playground

"Search for what the code does, not just what it is."

This repository accompanies a detailed Medium article exploring the end‑to‑end engine behind modern AI coding assistants (Cursor, Windsurf, Roo Code, GitHub Copilot, etc.). It implements a minimal, transparent version of the semantic code search + RAG pipeline so you can study, extend, and productionize the ideas.

</div>

---
## 1. Why This Matters
Traditional keyword (grep) search fails when the words in your head are not literal substrings in the code. Semantic code search bridges that gap by representing both code and natural language queries as dense vectors in the same concept space, enabling intent‑based lookup and powering higher layers like RAG answers and autonomous code agents.

This project demonstrates the core substrate shared (in more sophisticated form) by tools like Cursor:
1. File discovery & change detection (we model only static ingestion here).
2. Language‑aware chunking (Tree‑sitter style) or structured heuristics.
3. Embedding enrichment (LLM‑generated descriptions + code → hybrid text).
4. Vector storage & similarity (Qdrant, cosine distance).
5. Retrieval for: (a) direct display, (b) RAG context prompting.

---
## 2. High‑Level Architecture

### 2.1 Linear Flow
```text
 Source Code
    │ (scan / split)
    ▼
  Chunking  ──►  (optional) LLM description enrichment  ──►  Embedding
                                        (dense vectors)
    ▼
  Vector Store (Qdrant)  ◄── upsert (vectors + metadata)
    │
    │ query (embed user question with SAME model)
    ▼
  Retrieval (top‑k points) ──► (optional) RAG Prompt Assembly ──► LLM Answer
```

### 2.2 Layered View
```text
┌──────────────────────────────┐
│ Ingestion Layer              │  scan → detect → split (AST / heuristic)         
└──────────────┬───────────────┘
          ▼
┌──────────────────────────────┐
│ Representation Layer         │  hybrid text (description + code) → embedding   
└──────────────┬───────────────┘
          ▼
┌──────────────────────────────┐
│ Storage / Index Layer        │  Qdrant (cosine / HNSW)                          
└──────────────┬───────────────┘
          ▼
┌──────────────────────────────┐
│ Retrieval Layer              │  vector similarity (+ future filters/rerank)     
└──────────────┬───────────────┘
          ▼
┌──────────────────────────────┐
│ Generation Layer (Optional)  │  RAG prompt → LLM                                 
└──────────────────────────────┘
```

### 2.3 Core Invariants
1. Every vector must be traceable back (file path + line span).
2. Query embeddings MUST use the identical model as indexing.
3. Chunk quality (semantic coherence) dominates retrieval quality.
4. Metadata richness enables filtering, reranking, grounding, and citations.

---
## 3. Repository Layout
```
1 - Code Splitting/
  1 - fixed_size_chunking.ipynb
  2 - recursive_character_splitting.ipynb
  3 - ast_based_splitting.ipynb
2 - Vector Embedding/
  1 - create_vector_embedding.ipynb
3 - Indexing/
  1 - store_vector_embeddings.ipynb
4 - Searching/
  1 - code_search.ipynb
docker-compose.yml
README.md
```

| Stage | Notebook | Role | Notes |
|-------|----------|------|-------|
| Chunking | Code Splitting/* | Produce candidate chunks | Multiple strategies demonstrated |
| Embedding | Vector Embedding | Generate hybrid embeddings | Adds optional LLM description |
| Indexing | Indexing | Upsert into Qdrant | Drops/recreates collection for idempotence |
| Retrieval | Searching | Run semantic queries | Two modes: retrieval + RAG |

---
## 4. Chunking / Splitting Strategies
### 4.1 Fixed-Size (Baseline)
Fast, language‑agnostic, but slices semantics arbitrarily. Good as a control.

### 4.2 Recursive Character Splitting
Hierarchical separators (e.g. class → function → blank line → line) to preserve larger semantic units before backing off to smaller granularity.

### 4.3 AST / Parser (Tree-Sitter Style)
Parses the file and extracts structural nodes (functions, classes, methods). Produces the cleanest, semantically self‑contained chunks → highest quality embeddings. Requires careful version alignment (`tree-sitter` + `tree_sitter_languages`).

### 4.4 (Future) Semantic Boundary Detection
Experimental: embed very small units, detect topic shifts via cosine distance deltas, split at semantic cliffs.

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| Fixed | Simple, universal | Low precision | Quick prototype |
| Recursive | Better boundaries | Regex brittleness | Multi-language w/o full AST |
| AST | High fidelity | Parser setup complexity | Primary production path |
| Semantic | Captures sub‑function shifts | Expensive, experimental | Research / refinement |

---
## 5. Embedding Enrichment
Instead of embedding raw code alone, we optionally prepend a concise LLM‑generated intent line ("description") and delimiter it from the code. This hybrid text boosts recall for natural language queries describing *functionality* rather than syntax.

Example composition:
```
Description: Validates a user's authentication token and returns profile.
---
Code:
def authenticate_user(token: str) -> dict:
    ...
```
Model used here: `text-embedding-3-small` (1536 dims). Adjust `VECTOR_SIZE` if you swap models.

Key considerations:
- Consistency: Same prompt style for all descriptions.
- Determinism: Temperature 0 for description generation (if used) to avoid churn.
- Caching: Hash chunk → reuse embedding (not yet implemented here; see Roadmap).

---
## 6. Indexing (Qdrant)
Notebook creates (or recreates) the `semantic_code_search` collection with cosine distance. Each point payload typically contains:
```
{
  "snippet": "def validate_email(email): ...",
  "llm_description": "Validates email format.",
  "context": {"file_path": "src/utils/validators.py", "start": 10, "end": 24, "language": "python"}
}
```
Idempotence: existing collection is dropped before recreation to prevent schema drift or stale vectors.

---
## 7. Retrieval Modes
| Mode | What happens | RAG? | Output |
|------|--------------|------|--------|
| Method 1 | Query embedding → ANN search → display ranked snippets | No | Panels with code + descriptions |
| Method 2 | Retrieval → Construct context prompt → LLM answer | Yes | Synthesized grounded answer |

Prompt Guard Rails (recommended additions):
- Explicit instruction: *"Answer only from provided context; if insufficient, say so."*
- Delimit each snippet with clear markers.

---
## 8. Setup & Quickstart
Prereqs: Python 3.11+, Docker, OpenAI API key.

````bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
docker compose up -d   # launches Qdrant
````
Create `.env`:
```
OPENAI_API_KEY=sk-...
```
Run notebooks in order (1 → 4). Re-run embedding + indexing after any chunking change.

Minimal search demo (after indexing):
```python
from openai import OpenAI; from qdrant_client import QdrantClient
client=OpenAI(); qc=QdrantClient("localhost",6333)
emb=client.embeddings.create(model="text-embedding-3-small", input=["email validation"]).data[0].embedding
pts=qc.query_points(collection_name="semantic_code_search", query=emb, limit=2, with_payload=True).points
for p in pts: print(round(p.score,4), p.payload['llm_description'])
```

---
## 9. Operational Considerations (Inspired by Real Systems like Cursor)
| Concern | Production Tactic | Status Here |
|---------|-------------------|-------------|
| Change Detection | Merkle tree + incremental hashing | Not implemented (static) |
| Embedding Cost | Global hash → embedding cache (DynamoDB, redis) | Planned |
| Privacy | Store only metadata + line spans remotely | Demonstrated (payload minimal) |
| Latency | Local retrieval + streamed LLM | Basic sync version |
| Hybrid Search | Dense + sparse merge + rerank | Roadmap |
| Evaluation | MRR / NDCG benchmark set | Roadmap |

---
## 10. Extending Language Support
1. Add grammar via `tree_sitter_languages.get_parser("<language>")`.
2. Map node kinds to chunk types (e.g. `function_declaration`, `class_declaration`).
3. Normalize metadata (name, start/end lines, language).
4. Add language filter field in payload for downstream filtering.

---
## 11. Security & Hygiene
- `.env` is git‑ignored; never commit secrets.
- Rotate keys if exposure is suspected.
- (Future) Add secret scanning pre-commit.

---
## 12. Roadmap / Next Steps
1. Hybrid (BM25 + Vector) + cross‑encoder rerank.
2. Embedding cache keyed by content hash.
3. Evaluation harness (queries, relevance judgements, metrics).
4. Agentic workflows: multi‑step reasoning over retrieved code.
5. Fine‑tune / domain‑adapt embeddings (optional).
6. Streaming answer UI + citation highlighting.
7. Semantic chunk boundary detection experiment.

---
## 13. Glossary
| Term | Definition |
|------|------------|
| Chunk | Semantically coherent code unit (function/class/etc.). |
| Embedding | High‑dimensional numeric representation of semantics. |
| Vector DB | Specialized store for similarity search over embeddings. |
| RAG | Retrieval-Augmented Generation; retrieved context grounds LLM output. |
| Hybrid Search | Combination of dense (vector) + sparse (keyword) signals. |

---
## 14. Attribution
Content & implementation synthesized for educational purposes, inspired by public discussions of modern code understanding tools and best practices in semantic retrieval.

---
Happy exploring & extending! 🚀

