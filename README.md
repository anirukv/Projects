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
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set your Google API key

```bash
set GOOGLE_API_KEY=your_key_here
```

You can also place the key in a .env file. The app fails clearly if the key is missing.

## Ingest the documents

Place your PDFs in the company folders under documents/ and run:

```bash
python ingest.py
```

The ingestion script loads every PDF in the company subfolders, splits the text into chunks of about 1000 characters with 150-character overlap, tags each chunk with the company metadata, and stores the embeddings in the local Chroma database at /chroma_db.

## Run the app

```bash
streamlit run app.py
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
```
