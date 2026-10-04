from app.rag.embeddings import EmbeddingModel


def main():

    print("=" * 60)
    print("EMBEDDING MODEL TEST")
    print("=" * 60)

    model = EmbeddingModel()

    texts = [
        "Retrieval-Augmented Generation combines retrieval with generation.",
        "RAG retrieves external information before generating an answer.",
        "The weather is sunny today.",
    ]

    embeddings = model.encode(texts)

    print("\nEmbedding shape:")
    print(embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0][:10])

    print("\nEmbedding test completed.")


if __name__ == "__main__":
    main()