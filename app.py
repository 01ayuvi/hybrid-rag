from ingestion.loader import load_pdf
from ingestion.chunker import split_documents

docs = load_pdf("data/docs/company_policy.pdf")

print(f"Pages Loaded: {len(docs)}")

chunks = split_documents(docs)

print(f"Chunks Created: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0].page_content)