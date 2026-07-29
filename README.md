# Multi-user earnings-call Q&A demo

This project demonstrates a multi-user document search and conversational Q&A system for earnings-call PDFs. Users are restricted to the companies they are authorized to see, and the retrieval boundary is enforced structurally through a metadata filter in the vector-store query.

## What is included

- Streamlit UI with a login screen and per-user chat history
- LangGraph state graph with explicit retrieve -> generate nodes
- Chroma vector store persisted locally at /chroma_db
- Gemini chat + embedding support via Google AI Studio
- Dummy email-based access control using users.json

## Demo users

| User | Email | Authorized companies |
| --- | --- | --- |
| Alice | alice@email.com | CompanyA |
| Bob | bob@email.com | CompanyB, CompanyC |
| Charlie | charlie@email.com | CompanyD, CompanyE |

## Setup
1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

3. Set your Google API key (Streamlit Cloud: add as a secret named `GOOGLE_API_KEY`)

Locally, either export the env var or create a `.env` file containing:

```dotenv
GOOGLE_API_KEY=your_key_here
```

## Ingest the documents

Place your PDFs in the `docs/` folder using filenames that map to company aliases (e.g. `google.pdf`, `netflix.pdf`). Then run:

```bash
python ingest.py
```

Notes:
- The script splits PDFs into ~1000-character chunks and stores them in a local Chroma database under `chroma_db/` by default.
- If your Google embedding quota is exceeded, the script falls back to a deterministic embedding function and still indexes documents (lower-quality embeddings).
- The code now prefers the DuckDB+Parquet Chroma backend in cloud environments to avoid Rust binding lifecycle issues.

## Run the app (local)

```bash
# Run Streamlit locally (port optional)
python -m streamlit run app.py --server.headless true --server.port 8503
```

Smoke test (quick check without Streamlit UI):

```bash
python test_smoke.py
```

## Demo walkthrough

1. Log in as Alice and ask: "What was Company A's revenue in the latest earnings call?"
2. Follow up with: "Summarize the main takeaways from that call."
3. Log out and sign in as Bob, then ask a question about one of Bob's authorized companies.
4. Sign in as Charlie and ask: "What was Company A's Q4 revenue?"
   - The system should return a clear no-authorized-documents response instead of leaking information from CompanyA.

## Access-control notes

The hard boundary is enforced in two places:

- Retrieval is filtered using the metadata expression {"company": {"$in": allowed_companies}} before any document content is returned.
- Each user is assigned a separate LangGraph thread id derived from their email, so conversation history stays isolated.

## Project structure

```text
app.py
graph.py
ingest.py
users.json
requirements.txt
README.md
documents/
  CompanyA/
  CompanyB/
  CompanyC/
  CompanyD/
  CompanyE/

## Streamlit Cloud deployment notes

- In the Streamlit app deploy form use:
  - **Main file path:** `app.py`
  - **Run command:** `python -m streamlit run app.py --server.headless true --server.port $PORT`
  - Add a secret named `GOOGLE_API_KEY` with your API key
- If you prefer a persistent on-disk Chroma DB on the host, set the `CHROMA_DB_DIR` env var to a writable path; otherwise the app uses a project-local `chroma_db/` directory.

## Troubleshooting

- If you see errors from `chromadb` about Rust bindings on Streamlit Cloud, ensure `requirements.txt` includes `duckdb` and `pyarrow` so the DuckDB backend is available. We default to the DuckDB+Parquet implementation to avoid Rust lifecycle issues.
- If ingestion fails due to Gemini embedding quota, re-run `python ingest.py` after quota refresh or rely on the deterministic fallback.
```
