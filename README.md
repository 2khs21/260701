# CFR RAG — FAA Regulations Q&A

A **retrieval-augmented generation (RAG)** starter built on a corpus of 14 CFR (U.S. Federal Aviation Regulations) PDFs. It includes hybrid search, §-aware chunking, citation validation, token-saving optimizations, and a streaming chat UI.

---

## Project layout

```
rag-starter/
├── documents/                  CFR PDFs (Title 14)
├── indexer.py                  PDF cleanup → § chunking → embed → index.pkl
├── index.pkl                   Generated index (gitignored; rebuild with indexer)
├── backend/
│   ├── app.py                  Flask API — hybrid search, context optimization, Claude
│   └── requirements.txt
├── frontend/                   React + Vite UI
│   └── src/
│       ├── App.jsx             Chat, streaming, citations & warnings
│       ├── SearchParamsPanel.jsx   Tunable search parameters
│       ├── GlossaryPanel.jsx       Abbreviation / definition rollup
│       └── TokenUsageBar.jsx       Token usage & savings display
├── scripts/
│   └── run_benchmark.py        Local API benchmark script
├── docs/architecture/          Mermaid architecture diagrams
├── TECHNIQUES.md               Detailed technique reference
├── upgrade.md                  Improvement log & rubric mapping
└── .env.example
```

---

## Corpus

FAA 14 CFR PDFs in `documents/`:

| File | Contents |
|------|----------|
| `CFR-2025-title14-vol1.pdf` | Title 14 Vol. 1 (full volume) |
| `CFR-2025-title14-vol2-part61.pdf` | Part 61 — Certification: Pilots, Flight Instructors, and Ground Instructors |
| `CFR-2025-title14-vol2-part67.pdf` | Part 67 — Medical Standards and Certification |
| `CFR-2025-title14-vol2-part71.pdf` | Part 71 — Designation of Class A, B, C, D, and E Airspace |
| `CFR-2025-title14-vol2-part73.pdf` | Part 73 — Special Use Airspace |
| `CFR-2025-title14-vol2-part91.pdf` | Part 91 — General Operating and Flight Rules |

Indexing produces roughly **6,400** §-based chunks (exact count depends on corpus and chunking settings).

---

## Key features

**Indexing (`indexer.py`)**
- CFR PDF noise cleanup (page breaks, running headers, hyphenated line breaks)
- `§` section-boundary chunking (~1000 chars, 100-char overlap)
- Metadata enrichment (`section`, `part`, `section_title`)
- Vector + BM25 hybrid index (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim)

**Retrieval & generation (`backend/app.py`)**
- Query expansion, § hint injection, re-ranking
- Vector + BM25 + RRF fusion search
- TOC / noise filtering, context excerpts & trimming
- Adaptive `max_tokens` (short questions → 512)
- Anthropic **Claude Sonnet 4.6** streaming responses
- `[n]` citation parsing & validation, out-of-corpus refusal, prompt-injection hardening

**UI (`frontend/`)**
- Markdown answers, citation sources with excerpts, warning blocks
- Search params panel (`top_k`, `max_distance`, `max_context_chars`, `max_tokens`)
- Token usage & savings breakdown, response timing, session totals

---

## Setup

```bash
# from project root
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env
# set ANTHROPIC_API_KEY in .env
```

The first import of `sentence-transformers` downloads the embedding model (~470 MB) once.

---

## Build the index

```bash
python indexer.py
# Indexing documents from documents/
#   CFR-2025-title14-vol2-part91.pdf: ...
# ✓ Indexed N chunks → index.pkl
```

`index.pkl` is gitignored — run the command above after cloning.

---

## Run

```bash
# Terminal 1 — backend (port 5000)
cd backend
python app.py

# Terminal 2 — frontend (port 5173)
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173> in your browser.

---

## Sample questions

Try these in order of difficulty (same family as `scripts/run_benchmark.py`):

| Difficulty | Question |
|------------|----------|
| Single section | *What aeronautical experience is required for a private pilot certificate with an airplane single-engine rating?* |
| Enumeration | *Which medical conditions disqualify an applicant for a first-class airman medical certificate?* |
| Comparison | *What are the fuel-reserve requirements for VFR flight, day versus night?* |
| Cross-section | *How do operating requirements differ between Class B and Class C airspace?* |
| Out of corpus | *What is the Artemis program?* — should refuse to guess if not in the corpus |

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send a question. NDJSON stream (`token`, `done` events) |
| `/api/config` | GET | Default search parameters |

Example `POST /api/chat` body:

```json
{
  "message": "What are the VFR fuel reserve requirements?",
  "history": [],
  "params": {
    "top_k": 6,
    "max_distance": 0.50,
    "max_context_chars": 4500,
    "max_tokens": 768
  }
}
```

---

## Benchmark

With the backend running:

```bash
python scripts/run_benchmark.py
```

Checks expected § references, sources, forbidden sources, and answer keywords per question.

---

## Documentation

| Doc | Contents |
|-----|----------|
| [TECHNIQUES.md](./TECHNIQUES.md) | RAG techniques, parameters, token-saving strategies |
| [docs/architecture/](./docs/architecture/README.md) | System & pipeline Mermaid diagrams |
| [upgrade.md](./upgrade.md) | Improvement log against evaluation rubric |

---

## Pipeline overview

```
Question → query expansion → hybrid search → § injection & re-ranking
        → noise / TOC filter → context optimization → LLM (streaming) → citations & warnings
```

---

## Swap the corpus

Replace the files in `documents/` with your own PDFs or text, then re-run `python indexer.py`.  
Supported formats: `.md`, `.txt`, `.pdf`. Documents without `§` markers fall back to paragraph / fixed-length chunking.
