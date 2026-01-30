# QA Bot (FastAPI + LangChain + FAISS)

This project implements a document-grounded question answering backend API.  
Clients upload two files:
1) `questions_file`: a JSON file containing a list of questions
2) `document_file`: a PDF or JSON document to answer from

The service uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant document chunks and answers using an LLM **only with the retrieved context**. If the answer is not present, the system responds with `Not found`.

---

## Tech Stack

- FastAPI (API server)
- LangChain (chunking, embeddings, LLM wrapper, retriever interface)
- FAISS (in-memory vector index)
- OpenAI embeddings + `gpt-4o-mini` for answering

---

## API

### `POST /qa`

**Request**: `multipart/form-data` with:
- `questions_file` (file): JSON list of strings
- `document_file` (file): PDF or JSON

**Response**:
```json
{
  "results": [
    {
      "question": "string",
      "answer": "string",
      "sources": ["page 1", "page 2"]
    }
  ]
}