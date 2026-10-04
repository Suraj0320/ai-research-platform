from app.rag.pipeline import RAGPipeline


def main():

    document_path = "data/documents/documents.txt"

    # ==================================================
    # RETRIEVAL CONFIGURATION
    # ==================================================

    candidate_k = 30
    top_k = 20
    context_k = 5

    retrieval_method = "hybrid"
    rerank = True

    # ==================================================
    # CREATE PIPELINE
    # ==================================================

    pipeline = RAGPipeline()

    # ==================================================
    # GENERATE ANSWER
    # ==================================================

    result = pipeline.generate(
        question="How does RAG retrieve information?",
        document_path=document_path,

        # Initial hybrid retrieval
        candidate_k=candidate_k,

        # Documents kept after reranking
        top_k=top_k,

        # Documents sent to LLM
        context_k=context_k,

        retrieval_method=retrieval_method,
        rerank=rerank,
    )

    # ==================================================
    # QUESTION
    # ==================================================

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(result["question"])

    # ==================================================
    # ANSWER
    # ==================================================

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    # ==================================================
    # RETRIEVAL CONFIGURATION
    # ==================================================

    print("\n" + "=" * 70)
    print("RETRIEVAL CONFIGURATION")
    print("=" * 70)

    print(
        f"Retrieval Method : "
        f"{result['retrieval_method']}"
    )

    print(
        f"Reranking Enabled : "
        f"{result['reranking_enabled']}"
    )

    print(
        f"Candidate K       : "
        f"{result['candidate_k']}"
    )

    print(
        f"Reranked Top K    : "
        f"{result['top_k']}"
    )

    print(
        f"Final Context K   : "
        f"{result['context_k']}"
    )

    # ==================================================
    # RERANKED DOCUMENTS
    # ==================================================

    print("\n" + "=" * 70)
    print("RERANKED DOCUMENTS")
    print("=" * 70)

    reranked_documents = result[
        "retrieved_documents"
    ]

    print(
        f"Total reranked documents: "
        f"{len(reranked_documents)}"
    )

    for rank, doc in enumerate(
        reranked_documents,
        start=1,
    ):

        print(f"\nRank {rank}")
        print("-" * 70)

        print(
            f"Reranker Score : "
            f"{doc.get('reranker_score', 0.0):.4f}"
        )

        print(
            f"Hybrid Score   : "
            f"{doc.get('hybrid_score', 0.0):.6f}"
        )

        print(
            f"Semantic Score : "
            f"{doc.get('semantic_score', 0.0):.4f}"
        )

        print(
            f"Keyword Score  : "
            f"{doc.get('keyword_score', 0.0):.4f}"
        )

        print(
            f"Semantic Rank  : "
            f"{doc.get('semantic_rank')}"
        )

        print(
            f"Keyword Rank   : "
            f"{doc.get('keyword_rank')}"
        )

        print(
            f"Reranker Rank  : "
            f"{doc.get('reranker_rank')}"
        )

        print("\nText:")
        print(doc["text"])

    # ==================================================
    # FINAL CONTEXT
    # ==================================================

    print("\n" + "=" * 70)
    print("FINAL CONTEXT SENT TO LLM")
    print("=" * 70)

    final_context = result[
        "used_context"
    ]

    print(
        f"Total context documents: "
        f"{len(final_context)}"
    )

    for rank, doc in enumerate(
        final_context,
        start=1,
    ):

        print(f"\nContext {rank}")
        print("-" * 70)

        print(doc["text"])


if __name__ == "__main__":
    main()