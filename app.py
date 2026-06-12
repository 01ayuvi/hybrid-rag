from ingestion.loader import load_pdf
from ingestion.chunker import split_documents
from ingestion.embedder import create_embeddings

from retrieval.dense import (
    store_chunks,
    search
)

docs = load_pdf(
    "data/docs/company_policy.pdf"
)

chunks = split_documents(docs)

texts = [
    chunk.page_content
    for chunk in chunks
]

embeddings = create_embeddings(texts)

try:
    store_chunks(chunks, embeddings)
except:
    pass

query = "How do attackers abuse code signing?"

query_embedding = create_embeddings(
    [query]
)[0]

results = search(query_embedding)

print("\nQUESTION:")
print(query)

print("\nTOP RESULTS:\n")

for i, doc in enumerate(results["documents"][0]):

    print(f"\nResult {i+1}")

    if "distances" in results:
        print(
            f"Distance: "
            f"{results['distances'][0][i]:.4f}"
        )

    print(doc[:500])

    print("-" * 60)
results = search(query_embedding)

print(type(results))
print(results.keys())