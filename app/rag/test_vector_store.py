from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import FAISSVectorStore


def main():

    print("=" * 60)
    print("FAISS VECTOR STORE TEST")
    print("=" * 60)

    documents = [
        "Retrieval-Augmented Generation combines retrieval with generation.",
        "RAG retrieves external information before generating an answer.",
        "Embeddings represent text as numerical vectors.",
        "FAISS is used for efficient similarity search.",
        "The weather is sunny today.",
    ]

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(documents)

    print("\nEmbedding shape:")
    print(embeddings.shape)

    # --------------------------------------------------
    # Vector store
    # --------------------------------------------------

    vector_store = FAISSVectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(
        embeddings,
        documents,
    )

    print("\nDocuments indexed:")
    print(len(vector_store.documents))

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    query = "How does RAG retrieve information?"

    query_embedding = embedding_model.encode(
        [query]
    )

    results = vector_store.search(
        query_embedding,
        top_k=3,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\nQuery:")
    print(query)

    print("\nTop results:")

    for i, result in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()