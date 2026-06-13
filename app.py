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

from generation.llm import (
    generate_answer
)


# ==========================
# LOAD DOCUMENT
# ==========================

docs = load_pdf(
    "data/docs/company_policy.pdf"
)

# ==========================
# CHUNK DOCUMENT
# ==========================

chunks = split_documents(docs)

texts = [
    chunk.page_content
    for chunk in chunks
]

print(f"\nTotal Chunks: {len(texts)}")

# ==========================
# CREATE EMBEDDINGS
# ==========================

embeddings = create_embeddings(texts)

# ==========================
# STORE IN CHROMADB
# ==========================

try:
    store_chunks(
        chunks,
        embeddings
    )
except Exception:
    pass

# ==========================
# BM25 INDEX
# ==========================

bm25 = BM25Retriever(texts)

# ==========================
# USER QUERY
# ==========================

query = input("\nAsk a question: ")

# ==========================
# QUERY EMBEDDING
# ==========================

query_embedding = create_embeddings(
    [query]
)[0]

# ==========================
# DENSE RETRIEVAL
# ==========================

dense_results = search(
    query_embedding
)

# ==========================
# BM25 RETRIEVAL
# ==========================

bm25_results = bm25.search(
    query
)

# ==========================
# HYBRID RETRIEVAL
# ==========================

hybrid_results = hybrid_search(
    dense_results,
    bm25_results,
    texts
)

print("\nHYBRID RESULTS\n")

for doc, score in hybrid_results:

    print(f"\nHybrid Score: {score:.2f}")
    print(doc[:300])
    print("-" * 60)

# ==========================
# RERANKING
# ==========================

docs_for_rerank = [
    doc
    for doc, score in hybrid_results
]

reranked = rerank(
    query,
    docs_for_rerank
)

print("\nRERANKED RESULTS\n")

for doc, score in reranked:

    print(f"\nCrossEncoder Score: {score:.4f}")
    print(doc[:300])
    print("-" * 60)

# ==========================
# BUILD CONTEXT
# ==========================

context = "\n\n".join(
    [
        doc
        for doc, score in reranked[:3]
    ]
)

# ==========================
# GENERATE ANSWER
# ==========================

answer = generate_answer(
    query,
    context
)

print("\n" + "=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print(answer)

print("\n" + "=" * 80)
from evaluation.evaluator import (
    run_evaluation
)

run_evaluation()