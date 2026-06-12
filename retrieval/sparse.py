from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, texts):

        self.texts = texts

        tokenized_docs = [
            text.lower().split()
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            tokenized_docs
        )

    def search(self, query, top_k=5):

        tokenized_query = (
            query.lower().split()
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]