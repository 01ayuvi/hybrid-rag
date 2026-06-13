from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank(query, docs, top_k=5):

    pairs = [
        (query, doc)
        for doc in docs
    ]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]