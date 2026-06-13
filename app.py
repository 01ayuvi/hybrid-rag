from ingestion.loader import load_pdf
from ingestion.chunker import split_documents
from ingestion.embedder import create_embeddings

from retrieval.dense import (
    store_chunks,
    search
)

from retrieval.sparse import (
    BM25Retriever
)

from retrieval.hybrid import (
    hybrid_search
)

from retrieval.reranker import (
    rerank
)

# Load document
docs = load_pdf(
    "data/docs/company_policy.pdf"
)

# Chunk document
chunks = split_documents(docs)

# Extract text
texts = [
    chunk.page_content
    for chunk in chunks
]

# Create embeddings
embeddings = create_embeddings(texts)

# Store in ChromaDB
try:
    store_chunks(
        chunks,
        embeddings
    )
except Exception:
    pass

# Initialize BM25
bm25 = BM25Retriever(texts)

# User query
query = "How do attackers abuse code signing?"

# Query embedding
query_embedding = create_embeddings(
    [query]
)[0]

# Dense retrieval
dense_results = search(
    query_embedding
)

# Sparse retrieval
bm25_results = bm25.search(
    query
)

# Hybrid retrieval
hybrid_results = hybrid_search(
    dense_results,
    bm25_results,
    texts
)

print("\nHYBRID RESULTS:\n")

for doc, score in hybrid_results:

    print(f"\nHybrid Score: {score:.2f}")
    print(doc[:500])
    print("-" * 60)

# Cross Encoder Re-ranking

docs_for_rerank = [
    doc
    for doc, score in hybrid_results
]

reranked = rerank(
    query,
    docs_for_rerank
)

print("\nRERANKED RESULTS:\n")

for doc, score in reranked:

    print(f"\nCrossEncoder Score: {score:.4f}")
    print(doc[:500])
    print("-" * 60)