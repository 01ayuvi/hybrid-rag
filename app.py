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
    store_chunks(
        chunks,
        embeddings
    )
except:
    pass

bm25 = BM25Retriever(texts)

query = (
    "How do attackers abuse code signing?"
)

query_embedding = (
    create_embeddings([query])[0]
)

dense_results = search(
    query_embedding
)

bm25_results = bm25.search(
    query
)

results = hybrid_search(
    dense_results,
    bm25_results,
    texts
)

for doc, score in results:

    print(
        f"\nScore: {score:.2f}"
    )

    print(doc[:500])

    print("-" * 60)