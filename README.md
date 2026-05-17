# ВЕКТОР — Virtual Expert Knowledge Technical Operations Research

> Local RAG system for L3 support engineers. Answers technical questions from internal documentation in 2–5 seconds.

**Accuracy:** 91.4% on golden test cases · **Questions covered:** 98 · **Data leakage:** 0%

---

## What it does

ВЕКТОР indexes internal technical documentation and uses a hybrid search (deterministic rules + vector similarity + keyword scoring) to find precise answers to L3 support questions — without sending any data to external services.

## Stack

| Component | Technology |
|-----------|-----------|
| LLM | Qwen2.5:14b via [Ollama](https://ollama.ai) |
| Embeddings | nomic-embed-text via Ollama |
| Vector DB | ChromaDB + numpy (vectors.npz) |
| Interface | Streamlit |
| Platform | Windows + Python 3.11 |

## Quick Start

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.ai) installed and running.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull models
ollama pull nomic-embed-text
ollama pull qwen2.5:14b

# 3. Add your documents
# Place PDF files into docs/ folder

# 4. Index documents
python indexer.py

# 5. Launch
streamlit run app.py
```

Or use the batch files:
```
1_setup.bat   — install dependencies + pull models
2_index.bat   — index documents
3_start.bat   — launch UI
4_eval.bat    — run golden test suite (8 cases)
5_mass_eval.bat       — mass evaluation, retrieval-only
6_mass_eval_full.bat  — mass evaluation with LLM generation
```

## Architecture

```
Question
   │
   ▼
reason()  ──→  LLM classifies query type + technical terms
   │
   ▼
retrieve()
   ├── PAGE_TRIGGERS   →  deterministic rules: keyword → PDF pages (27 rules)
   ├── Vector search   →  4,720 chunks, nomic-embed-text embeddings
   └── Keyword scoring →  weighted term matching
   │
   ▼
generate()  ──→  Qwen2.5:14b generates answer with context
   │
   ▼
Answer + source page citations
```

## Security

- **Local-only**: Ollama runs on-premise, no data leaves the company network
- **No PII**: Only public technical manuals are indexed
- **Open source**: Full stack is auditable, no vendor lock-in
- **Auditable**: Every answer shows source document + page number

## Evaluation

Run the golden test suite (8 real L3 cases):
```bash
python evaluate_rag.py --retrieval-only
```

Run full mass evaluation (98 questions, LLM generation):
```bash
python mass_eval.py --generate
```

Results are saved to `reports/` as JSON + Markdown. Markdown reports include copy-paste blocks for Gemini validation against original PDFs.

## Project Structure

```
mango_rag/
├── app.py              # Streamlit UI
├── evaluate_rag.py     # Evaluation pipeline + core RAG logic
├── mass_eval.py        # Mass evaluation (98 questions)
├── indexer.py          # Document indexing
├── requirements.txt    # Python dependencies
├── test_suite.json     # 8 golden test cases (no sensitive data)
├── docs/               # PDF documents — NOT in git (add your own)
└── reports/            # Eval reports — NOT in git
```

## What's NOT in this repo

For security reasons, the following are excluded via `.gitignore`:

- `docs/*.pdf` — internal Mango Office documentation
- `vectors.npz` — vector index (derived from docs)
- `chroma_db/` — ChromaDB index
- `данные для тренеровки.txt` — internal training data (Jira cases)
- `reports/` — evaluation results

To run ВЕКТОР you need to add your own documentation to `docs/` and re-run `2_index.bat`.

---

*Built by SanMog · Mango Office L3 Support · 2026*
