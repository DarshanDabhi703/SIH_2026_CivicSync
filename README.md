# CivicSync — Indian Legal RAG Assistant

> **Know Your Rights. Know What To Do Next.**  
> *Built for Smart India Hackathon — Smart Education 🚀*

CivicSync is an evidence-grounded, multilingual legal awareness assistant designed to help citizens in India understand their rights and take informed next steps. It combines **pgvector-powered semantic search** (via Supabase), **domain-specialized legal chunking**, and a **grounded multi-provider LLM reasoning engine** to provide actionable legal rights, remedies, and procedures.

---

### ✨ Key Features
- 🔎 **Evidence-Grounded RAG** for reliable legal information
- 🧠 **Situation Understanding** to identify the user's issue and intent
- 🛡️ **Evidence Qualification** to reduce unsupported AI claims
- ⚡ **Action-First Guidance** with practical next steps
- 🌐 **Multilingual Support** — English, Hindi & Gujarati
- 🎙️ **Voice Input & Output** for accessible interaction
- 🤖 **Multiple LLM Providers** — Gemini, Groq & Ollama
- 📚 **Source-Based Responses** for better transparency

> 💡 **Core Principle**: *AI explains the evidence. The evidence pipeline controls the explanation.*  
> ⚠️ **Disclaimer**: CivicSync is intended for legal awareness and informational guidance, not professional legal advice.

---

## 🏛️ Supported Domains

- **Traffic Rules & Violations** (`traffic`)
- **Labour & Workers' Rights** (`labour`)
- **Women's Safety & Protection** (`women_safety`)
- **Consumer Rights & Protection** (`consumer`)
- **Insurance & Financial Rights** (`insurance`)
- **Land, Property & Tenancy Laws** (`land_property`)

---

## 📐 System Architecture

```text
[ Citizen Query ]
       │
       ▼
┌───────────────────────────────┐
│       React + Vite Frontend    │  (Port 5173)
└──────────────┬────────────────┘
               │  HTTP POST /api/query
               ▼
┌───────────────────────────────┐
│        FastAPI Backend        │  (Port 8000)
│   (backend/main.py)           │
└──────────────┬────────────────┘
               │
       ┌───────┴───────────────────────────────┐
       ▼                                       ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│  Civic Engine Reasoning      │    │  Supabase Vector Database    │
│  - Situation Extraction      │◄──►│  - match_chunks RPC          │
│  - Grounded Retrieval        │    │  - multilingual-e5 embeddings│
│  - Legal Qualification       │    └──────────────────────────────┘
│  - Action Engine Steps       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Multi-Provider LLM Gateway  │
│  (Groq / Gemini / Ollama)    │
└──────────────────────────────┘
```

---

## 📁 Repository Structure

```text
CivicSync Rag/
├── CivicSync_Project_Brief.pdf # Hackathon project brief document
├── backend/                    # FastAPI web server and API routes
│   ├── main.py                 # API endpoints (/api/health, /api/query)
│   └── schemas.py              # Pydantic request & response models
├── frontend/                   # React 19 + TypeScript + Vite user interface
│   ├── src/                    # Components, pages, hooks, styling
│   ├── package.json            # Frontend dependencies and scripts
│   └── .env.example            # Frontend environment variables template
├── src/                        # Core RAG, extraction, and reasoning logic
│   ├── civic_engine.py         # Main orchestration pipeline
│   ├── llm_gateway.py          # Unified LLM provider client (Groq/Gemini/Ollama)
│   ├── retriever.py            # Vector retrieval client via Supabase pgvector
│   ├── embedder.py             # Multilingual sentence transformer embedding generator
│   ├── situation.py            # Intent and scenario classification
│   ├── qualification.py        # Legal eligibility & context analyzer
│   ├── action_engine.py        # Prescriptive citizen step generator
│   ├── chunker.py              # Legal document chunking engine
│   └── pdf_parser.py           # PyPDF extractor
├── data/
│   ├── pdfs/                   # Canonical source legal documents (6 domains)
│   ├── processed/              # Extracted & normalized JSONL document chunks
│   └── embeddings/             # Pre-generated multilingual vector embeddings
├── sql/
│   └── match_chunks.sql        # Supabase pgvector cosine similarity function
├── tests/                      # Pytest test suite (unit + integration + e2e)
├── .env.example                # Backend environment variables template
├── .gitignore                  # Production-ready git ignore configuration
└── requirements.txt            # Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python**: `3.10` or higher (tested with Python 3.13)
- **Node.js**: `18.0` or higher and **npm**
- **Supabase Account**: With a project configured for `pgvector`
- **LLM API Key**: Groq API key (free/fast) or Google Gemini API key, or a local Ollama instance

---

### 2. Environment Setup

#### A. Backend Configuration
Copy `.env.example` in the project root to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and provide your credentials:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SECRET_KEY=your_supabase_service_role_or_anon_key

# LLM Provider Configuration ("groq", "gemini", or "ollama")
LLM_PROVIDER=groq

# Groq API Configuration (if using Groq)
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Gemini API Configuration (if using Gemini)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# Ollama Configuration (if using local Ollama)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Allowed CORS origins
CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173
```

#### B. Frontend Configuration
Copy `frontend/.env.example` to `frontend/.env`:

```bash
cd frontend
cp .env.example .env
cd ..
```

Ensure `frontend/.env` points to the FastAPI backend:
```env
VITE_API_URL=http://127.0.0.1:8000
```

---

### 3. Database Setup (Supabase)

1. Navigate to your Supabase Project Dashboard → **SQL Editor**.
2. Run the SQL script located in [`sql/match_chunks.sql`](sql/match_chunks.sql) to enable the `pgvector` extension and register the `match_chunks` similarity search RPC function.

---

### 4. Running the Backend

From the project root:

```bash
# 1. Create a Python virtual environment (optional but recommended)
python -m venv .venv

# 2. Activate the virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install required Python packages
pip install -r requirements.txt

# 4. Launch the FastAPI server
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

- **API Root**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Probe**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

### 5. Running the Frontend

In a separate terminal window:

```bash
# 1. Change to the frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
```

Open your browser and navigate to: **[http://localhost:5173](http://localhost:5173)**.

---

## 🧪 Running Automated Tests

Run the test suite to verify backend and pipeline functionality:

```bash
# Run all tests
pytest tests/ -v

# Run specific API tests
pytest tests/test_api.py -v

# Run pipeline chunking tests
pytest tests/test_pipeline.py -v
```

To run frontend tests:

```bash
cd frontend
npm test
```

---

## 🔄 Re-generating Chunks & Embeddings (Optional)

If you modify or add legal PDFs to `data/pdfs/`:

1. **Extract and chunk PDFs**:
   ```bash
   python src/pipeline.py
   ```
   Outputs normalized JSONL files into `data/processed/`.

2. **Generate vector embeddings**:
   ```bash
   python src/embedder.py
   ```
   Outputs vector files into `data/embeddings/`.

---

## 🔒 Security & Privacy Notice

- Never commit `.env` files containing real API keys or Supabase secrets to version control.
- All secrets are excluded via `.gitignore`.
- Always use `.env.example` as a template when sharing or deploying the codebase.
