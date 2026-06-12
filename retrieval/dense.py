import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="knowledge_base"
)


def store_chunks(chunks, embeddings):

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist()
    )
def search(query_embedding, top_k=5):

    return collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    