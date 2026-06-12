from ingestion.loader import load_pdf
from ingestion.chunker import split_documents

from retrieval.sparse import (
    BM25Retriever
)

docs = load_pdf(
    "data/docs/company_policy.pdf"
)

chunks = split_documents(docs)

texts = [
    chunk.page_content
    for chunk in chunks
]

bm25 = BM25Retriever(texts)

results = bm25.search(
    "code signing"
)

for idx, score in results:

    print(
        f"\nScore: {score:.2f}"
    )

    print(texts[idx][:300])

    print("-" * 60)