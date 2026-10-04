from app.rag.loader import load_document
from app.rag.chunker import chunk_text
from app.rag.hybrid_retriever import HybridRetriever


def main():

    print("=" * 60)
    print("HYBRID RETRIEVER TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Load document
    # --------------------------------------------------

    document_path = "data/documents/documents.txt"

    text = load_document(document_path)

    # --------------------------------------------------
    # Chunk document
    # --------------------------------------------------

    chunks = chunk_text(
        text,
        chunk_size=500,
        chunk_overlap=75,
    )

    print(f"\nTotal chunks: {len(chunks)}")

    # --------------------------------------------------
    # Build hybrid retriever
    # --------------------------------------------------

    retriever = HybridRetriever(chunks)

    # --------------------------------------------------
    # Test questions
    # --------------------------------------------------

    questions = [
        "What is Retrieval-Augmented Generation?",
        "How does RAG retrieve information?",
        "What is parametric memory in RAG?",
        "What is non-parametric memory in RAG?",
        "How does RAG update knowledge without retraining?",
    ]

    # --------------------------------------------------
    # Run retrieval
    # --------------------------------------------------

    for question in questions:

        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)

        results = retriever.retrieve(
            query=question,
            top_k=5,
        )

        for i, result in enumerate(
            results,
            start=1,
        ):

            print(f"\n--- Result {i} ---")

            print(
                f"Hybrid Score: "
                f"{result['hybrid_score']:.4f}"
            )

            print(
                f"Semantic Score: "
                f"{result['semantic_score']:.4f}"
            )

            print(
                f"Keyword Score: "
                f"{result['keyword_score']:.4f}"
            )

            print(
                f"Text:\n{result['text']}"
            )


if __name__ == "__main__":
    main()