def hybrid_search(
    dense_results,
    bm25_results,
    texts,
    top_k=5
):

    combined = {}

    # Dense Retrieval

    for i, doc in enumerate(
        dense_results["documents"][0]
    ):

        score = 1 / (
            dense_results["distances"][0][i]
            + 1e-6
        )

        combined[doc] = (
            combined.get(doc, 0)
            + score
        )

    # BM25 Retrieval

    for idx, score in bm25_results:

        doc = texts[idx]

        combined[doc] = (
            combined.get(doc, 0)
            + score
        )

    ranked = sorted(
        combined.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]