# Retrieval-Augmented Generation

This project is a modular Retrieval-Augmented Generation (RAG) application. It ingests documents, creates embeddings, stores vectors, retrieves relevant context, and passes that context to an LLM for grounded responses.

## Pipeline

```text
Documents -> loaders -> chunks -> embeddings -> Chroma
                                      |
                              retriever -> LLM response
```

## Repository layout

- `create_database.py` builds the vector store
- `main.py` provides the application entry point
- `app.py` contains the user-facing application layer
- `document loaders/` contains source-specific ingestion logic
- `retrievers/` contains retrieval strategies
- `vector store/` contains persistence and vector-database integration
- `chroma_db/` and `chroma_db_huggingface/` contain local vector-store data

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # if an example file is present
```

Configure the provider keys required by the selected loader, embedding model, and LLM before running the application. Do not commit `.env` files or credentials.

## Typical workflow

```bash
python create_database.py
python main.py
```

If the UI is Streamlit-based, start it with `streamlit run app.py`. The exact provider and model settings should be selected through environment variables so local experiments do not require code changes.

## Improving retrieval quality

Evaluate chunk size, overlap, metadata filters, embedding model, retriever parameters, and answer citations independently. Keep representative documents and question/answer checks so changes can be compared rather than judged only by individual examples.

## Limitations

Generated answers are only as reliable as the retrieved context and configured model. Treat outputs as assisted research and validate important claims against the source documents.
