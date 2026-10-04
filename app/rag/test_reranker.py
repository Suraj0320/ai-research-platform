from app.rag.loader import load_document
from app.rag.chunker import chunk_text
from app.rag.retriever import RAGRetriever
from app.rag.reranker import RAGReranker


DOCUMENT_PATH = "data/documents/documents.txt"


def main():

    print("=" * 60)
    print("RAG RERANKER TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Load document
    # --------------------------------------------------

    text = load_document(DOCUMENT_PATH)

    # --------------------------------------------------
    # Chunk document
    # --------------------------------------------------

    chunks = chunk_text(
        text,
        chunk_size=700,
        chunk_overlap=100,
    )

    print(f"\nTotal chunks: {len(chunks)}")

    # --------------------------------------------------
    # Build dense retriever
    # --------------------------------------------------

    retriever = RAGRetriever(chunks)

    # --------------------------------------------------
    # Build reranker
    # --------------------------------------------------

    reranker = RAGReranker()

    questions = [
        "What is Retrieval-Augmented Generation?",
        "How does RAG retrieve information?",
        "What is parametric memory in RAG?",
        "What is non-parametric memory in RAG?",
        "How does RAG update knowledge without retraining?",
    ]

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    for question in questions:

        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)

        # Dense retrieval
        dense_results = retriever.retrieve(
            question,
            top_k=10,
        )

        print("\nDENSE RETRIEVAL:")
        
        for i, result in enumerate(
            dense_results,
            start=1,
        ):

            print(f"\n--- Result {i} ---")
            print(
                f"Score: {result['score']:.4f}"
            )
            print(
                f"Text:\n{result['text']}"
            )

        # --------------------------------------------------
        # Reranking
        # --------------------------------------------------

        reranked_results = reranker.rerank(
            question,
            dense_results,
            top_k=3,
        )

        print("\n" + "-" * 60)
        print("RERANKED RESULTS:")
        print("-" * 60)

        for i, result in enumerate(
            reranked_results,
            start=1,
        ):

            print(f"\n--- Result {i} ---")

            print(
                f"Dense Score: "
                f"{result['score']:.4f}"
            )

            print(
                f"Rerank Score: "
                f"{result['rerank_score']:.4f}"
            )

            print(
                f"Text:\n{result['text']}"
            )


if __name__ == "__main__":
    main()