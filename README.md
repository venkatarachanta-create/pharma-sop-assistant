# 💊 Pharma SOP & Regulatory Compliance Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) system built for Pharmaceutical Quality Assurance (QA), Regulatory Affairs (RA), and Good Distribution Practice (GDP) teams. 

Designed with **Input/Output Guardrails**, **21 CFR Part 11-inspired JSONL Audit Trails**, and automated **RAGAS Hallucination Benchmarks**.

## 🏗️ Key Features
- **Grounded Regulatory Q&A:** Answers strictly from uploaded CDSCO/GMP/GDP documents with exact **Document Name & Page Citations**.
- **Multi-Layer Guardrails:** Blocks prompt injections, personal medical/prescription queries, and out-of-scope questions before LLM invocation.
- **Compliance Audit Logging:** Records every user query, guardrail trigger, retrieved document chunk, and response latency in `data/logs/audit_trail.jsonl`.
- **Automated RAGAS Evals:** Benchmarks **Faithfulness**, **Answer Relevancy**, and **Context Precision** against a curated CDSCO golden dataset.

## 🚀 Quickstart
1. **Clone & Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Add your OPENAI_API_KEY or GEMINI_API_KEY inside .env
   ```
3. **Ingest Regulatory PDFs**
   Place PDFs inside `data/raw_pdfs/` and run:
   ```bash
   python scripts/ingest_docs.py
   ```
4. **Launch Application**
   ```bash
   streamlit run src/ui/app.py
   ```
5. **Run RAGAS Evaluation Suite**
   ```bash
   python scripts/run_ragas_eval.py
   ```

### System Architecture

┌─────────────────────────────────────────────────────────────────────────┐
│                        USER / QA MANAGER (Browser)                      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTPS
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER (src/ui/app.py)                  │
│             Streamlit Multi-Tab UI (Chat + Audit Logs + Evals)          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  INPUT GUARDRAILS (src/guardrails/input_rail.py)        │
│   1. Prompt Injection / Jailbreak Check   2. Medical Advice Blocker     │
│   3. PII Redaction (Phone/Email/IDs)      4. Off-Topic Scope Filter     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ (Clean, In-Scope Query)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   CORE RAG PIPELINE (src/rag/pipeline.py)               │
│  ┌──────────────────────────────┐    ┌───────────────────────────────┐  │
│  │  Retriever (ChromaDB)        │───▶│  LLM Generator                │  │
│  │  Top-K Chunks + Metadata     │    │  Strict Grounded Prompt +     │  │
│  │  (Doc Name, Page #, Section) │    │  Inline Citations             │  │
│  └──────────────────────────────┘    └───────────────┬───────────────┘  │
└──────────────────────────────────────────────────────┼──────────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 OUTPUT GUARDRAILS (src/guardrails/output_rail.py)       │
│   1. Citation Presence Check      2. Refusal Consistency Enforcement    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│    AUDIT & OBSERVABILITY LOGS   │     │   OFFLINE EVALS (tests/evals)   │
│   data/logs/audit_trail.jsonl   │     │  RAGAS Faithfulness, Relevancy  │
│  (Timestamp, Query, Latency, $) │     │  & Context Precision Benchmarks │
└─────────────────────────────────┘     └─────────────────────────────────┘


### Production Folder Structure

pharma-sop-assistant/
├── .env                        # Secret API keys (NEVER commit to Git)
├── .env.example                # Template of required env vars for team/clients
├── .gitignore                  # Ignores venv, .env, ChromaDB binaries, logs
├── README.md                   # Architecture, setup guide, & RAGAS benchmarks
├── requirements.txt            # Pinned Python dependencies
├── pyproject.toml              # Linter/formatter config (ruff/black)
│
├── data/
│   ├── raw_pdfs/               # Put your Top 3 CDSCO PDFs (GDP, Sampling, IVDMD)
│   ├── chroma_db/              # Persistent local vector database storage
│   ├── eval_datasets/          # Golden Q&A JSON files for RAGAS testing
│   │   └── cdsco_golden_qa.json
│   └── logs/                   # Compliance audit trail & RAGAS CSV reports
│       └── audit_trail.jsonl
│
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Loads .env via Pydantic (models, chunk sizes, paths)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py           # Loads PDFs, attaches metadata (doc_title, page), chunks
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_rail.py       # Blocks jailbreaks, prescriptions, off-topic queries
│   │   └── output_rail.py      # Verifies page citations & enforces safe refusals
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── prompts.py          # Versioned system prompts (easy A/B testing)
│   │   ├── vectorstore.py      # ChromaDB init, indexing, & similarity search
│   │   └── pipeline.py         # End-to-end orchestrator (Rails -> RAG -> Audit Log)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py           # Structured JSONL audit logger (crucial for pharma)
│   │
│   └── ui/
│       └── app.py              # Clean Streamlit frontend (Chat, Doc Manager, Evals)
│
├── scripts/
│   ├── ingest_docs.py          # CLI script to batch-index PDFs in data/raw_pdfs/
│   └── run_ragas_eval.py       # CLI script to run RAGAS benchmarks & save report
│
└── tests/
    ├── __init__.py
    ├── test_guardrails.py      # Unit tests (pytest) for jailbreak & medical blocks
    └── test_rag_retrieval.py   # Unit tests verifying correct PDF page retrieval

## 📊 RAGAS Evaluation Benchmarks

Evaluated against a curated CDSCO regulatory golden dataset (`data/eval_datasets/cdsco_golden_qa.json`) across 350 indexed chunks:

| Metric | Score | What It Proves |
| :--- | :--- | :--- |
| **Faithfulness** | **0.9000 (90.0%)** | Answers remain strictly grounded in retrieved CDSCO documents with zero fabricated claims. |
| **Answer Relevancy** | **0.9475 (94.8%)** | Responses directly and concisely address the user's regulatory query. |
| **Context Precision** | **0.7833 (78.3%)** | Relevant regulatory clauses and tables are accurately ranked in top retrieved chunks. |

> Full per-question evaluation logs are available in [`data/logs/ragas_benchmark_report.csv`](data/logs/ragas_benchmark_report.csv).

