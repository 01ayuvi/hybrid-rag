from ingestion.loader import load_pdf
from ingestion.chunker import split_documents
from ingestion.embedder import create_embeddings

docs = load_pdf("data/docs/company_policy.pdf")

chunks = split_documents(docs)

texts = [chunk.page_content for chunk in chunks]

embeddings = create_embeddings(texts)

print(f"Total Chunks: {len(chunks)}")
print(f"Embedding Shape: {embeddings.shape}")

print("\nFirst Vector Preview:\n")
print(embeddings[0][:10])