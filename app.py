from ingestion.loader import load_pdf
from ingestion.chunker import split_documents
from ingestion.embedder import create_embeddings

from retrieval.dense import store_chunks

docs = load_pdf(
    "data/docs/company_policy.pdf"
)

chunks = split_documents(docs)

texts = [
    chunk.page_content
    for chunk in chunks
]

embeddings = create_embeddings(texts)

store_chunks(
    chunks,
    embeddings
)

print("Stored Successfully!")
print(f"Chunks Stored: {len(chunks)}")