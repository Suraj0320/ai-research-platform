from app.rag.loader import load_document
from app.rag.chunker import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import FAISSVectorStore
from app.rag.hybrid_retriever import HybridRetriever


DOCUMENT_PATH = "data/documents/documents.txt"


QUESTIONS = [
    "What is Retrieval-Augmented Generation?",
    "How does RAG retrieve information?",
    "What is parametric memory in RAG?",
    "What is non-parametric memory in RAG?",
    "How does RAG update knowledge without retraining?",
]


GROUND_TRUTH = {
    "What is Retrieval-Augmented Generation?": [
        "retrieval-augmented generation",
        "combine pre-trained parametric and non-parametric memory",
    ],

    "How does RAG retrieve information?": [
        "dense vector index of Wikipedia",
        "pre-trained neural retriever",
    ],

    "What is parametric memory in RAG?": [
        "parametric memory is a pre-trained seq2seq",
        "pre-trained seq2seq transformer",
    ],

    "What is non-parametric memory in RAG?": [
        "non-parametric memory is a dense vector index",
        "dense vector index of Wikipedia",
    ],

    "How does RAG update knowledge without retraining?": [
        "hot-swapped to update the model",
        "without requiring any retraining",
        "non-parametric memory can be replaced",
    ],
}


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(title, results):

    print(f"\n{'-' * 70}")
    print(title)
    print("-" * 70)

    if not results:
        print("No results")
        return

    for i, result in enumerate(results, 1):

        print(f"\nResult {i}")

        if "score" in result:
            print(
                f"Score: {result['score']:.4f}"
            )

        if "semantic_score" in result:
            print(
                f"Semantic Score: "
                f"{result['semantic_score']:.4f}"
            )

        if "keyword_score" in result:
            print(
                f"Keyword Score: "
                f"{result['keyword_score']:.4f}"
            )

        if "normalized_semantic" in result:
            print(
                f"Normalized Semantic: "
                f"{result['normalized_semantic']:.4f}"
            )

        if "normalized_keyword" in result:
            print(
                f"Normalized Keyword: "
                f"{result['normalized_keyword']:.4f}"
            )

        if "hybrid_score" in result:
            print(
                f"Hybrid Score: "
                f"{result['hybrid_score']:.4f}"
            )

        print(f"Text:\n{result['text']}")


# ============================================================
# RELEVANCE CHECK
# ============================================================

def is_relevant(text, ground_truth_terms):

    text_lower = text.lower()

    return any(
        term.lower() in text_lower
        for term in ground_truth_terms
    )


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    results,
    ground_truth_terms,
    k,
):
    """
    Returns:

        1.0 -> at least one relevant chunk
               appears in top-k

        0.0 -> no relevant chunk appears
               in top-k
    """

    top_results = results[:k]

    for result in top_results:

        if is_relevant(
            result["text"],
            ground_truth_terms,
        ):
            return 1.0

    return 0.0


# ============================================================
# RECIPROCAL RANK
# ============================================================

def reciprocal_rank(
    results,
    ground_truth_terms,
):
    """
    Returns reciprocal rank of the
    first relevant result.

    Example:

    Relevant result at rank 1 -> 1.0
    Relevant result at rank 2 -> 0.5
    Relevant result at rank 3 -> 0.333
    No relevant result -> 0.0
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if is_relevant(
            result["text"],
            ground_truth_terms,
        ):

            return 1.0 / rank

    return 0.0


# ============================================================
# EVALUATE ONE METHOD
# ============================================================

def evaluate_method(
    results,
    ground_truth_terms,
):
    """
    Calculate retrieval metrics for one question.
    """

    return {
        "recall@1": recall_at_k(
            results,
            ground_truth_terms,
            1,
        ),

        "recall@3": recall_at_k(
            results,
            ground_truth_terms,
            3,
        ),

        "recall@5": recall_at_k(
            results,
            ground_truth_terms,
            5,
        ),

        "mrr": reciprocal_rank(
            results,
            ground_truth_terms,
        ),
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    method_name,
    metrics,
):

    print("\n" + "-" * 70)
    print(f"{method_name} METRICS")
    print("-" * 70)

    print(
        f"Recall@1 : {metrics['recall@1']:.4f}"
    )

    print(
        f"Recall@3 : {metrics['recall@3']:.4f}"
    )

    print(
        f"Recall@5 : {metrics['recall@5']:.4f}"
    )

    print(
        f"MRR      : {metrics['mrr']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAISS vs BM25 vs HYBRID RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load document
    # --------------------------------------------------------

    text = load_document(
        DOCUMENT_PATH
    )

    # --------------------------------------------------------
    # Chunk document
    # --------------------------------------------------------

    chunks = chunk_text(
        text,
        chunk_size=500,
        chunk_overlap=75,
    )

    print(
        f"\nTotal chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Build embedding model
    # --------------------------------------------------------

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(
        chunks
    )

    # --------------------------------------------------------
    # Build FAISS vector store
    # --------------------------------------------------------

    vector_store = FAISSVectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(
        embeddings,
        chunks,
    )

    # --------------------------------------------------------
    # Build Hybrid Retriever
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    # ========================================================
    # Store metrics for final comparison
    # ========================================================

    all_metrics = {
        "FAISS": [],
        "BM25": [],
        "HYBRID": [],
    }

    # ========================================================
    # Evaluate every question
    # ========================================================

    for question in QUESTIONS:

        print("\n\n")
        print("=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        ground_truth_terms = GROUND_TRUTH[
            question
        ]

        # ====================================================
        # FAISS / SEMANTIC RETRIEVAL
        # ====================================================

        query_embedding = embedding_model.encode(
            [question]
        )

        faiss_results = vector_store.search(
            query_embedding,
            top_k=5,
        )

        print_results(
            "FAISS / SEMANTIC RETRIEVAL",
            faiss_results,
        )

        faiss_metrics = evaluate_method(
            faiss_results,
            ground_truth_terms,
        )

        print_metrics(
            "FAISS / SEMANTIC",
            faiss_metrics,
        )

        all_metrics["FAISS"].append(
            faiss_metrics
        )

        # ====================================================
        # BM25 / KEYWORD RETRIEVAL
        # ====================================================

        bm25_results = (
            hybrid_retriever.keyword_search(
                question,
                top_k=5,
            )
        )

        print_results(
            "BM25 / KEYWORD RETRIEVAL",
            bm25_results,
        )

        bm25_metrics = evaluate_method(
            bm25_results,
            ground_truth_terms,
        )

        print_metrics(
            "BM25 / KEYWORD",
            bm25_metrics,
        )

        all_metrics["BM25"].append(
            bm25_metrics
        )

        # ====================================================
        # HYBRID RETRIEVAL
        # ====================================================

        hybrid_results = (
            hybrid_retriever.retrieve(
                question,
                top_k=5,
            )
        )

        print_results(
            "HYBRID RETRIEVAL",
            hybrid_results,
        )

        hybrid_metrics = evaluate_method(
            hybrid_results,
            ground_truth_terms,
        )

        print_metrics(
            "HYBRID",
            hybrid_metrics,
        )

        all_metrics["HYBRID"].append(
            hybrid_metrics
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("OVERALL RETRIEVAL EVALUATION")
    print("=" * 70)

    for method in [
        "FAISS",
        "BM25",
        "HYBRID",
    ]:

        metrics_list = all_metrics[
            method
        ]

        avg_recall_1 = sum(
            m["recall@1"]
            for m in metrics_list
        ) / len(metrics_list)

        avg_recall_3 = sum(
            m["recall@3"]
            for m in metrics_list
        ) / len(metrics_list)

        avg_recall_5 = sum(
            m["recall@5"]
            for m in metrics_list
        ) / len(metrics_list)

        avg_mrr = sum(
            m["mrr"]
            for m in metrics_list
        ) / len(metrics_list)

        print("\n" + "-" * 70)
        print(method)
        print("-" * 70)

        print(
            f"Average Recall@1 : "
            f"{avg_recall_1:.4f}"
        )

        print(
            f"Average Recall@3 : "
            f"{avg_recall_3:.4f}"
        )

        print(
            f"Average Recall@5 : "
            f"{avg_recall_5:.4f}"
        )

        print(
            f"Mean Reciprocal Rank (MRR): "
            f"{avg_mrr:.4f}"
        )

    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Method':<12}"
        f"{'Recall@1':<15}"
        f"{'Recall@3':<15}"
        f"{'Recall@5':<15}"
        f"{'MRR':<15}"
    )

    print("-" * 70)

    for method in [
        "FAISS",
        "BM25",
        "HYBRID",
    ]:

        metrics_list = all_metrics[
            method
        ]

        avg_recall_1 = sum(
            m["recall@1"]
            for m in metrics_list
        ) / len(metrics_list)

        avg_recall_3 = sum(
            m["recall@3"]
            for m in metrics_list
        ) / len(metrics_list)

        avg_recall_5 = sum(
            m["recall@5"]
            for m in metrics_list
        ) / len(metrics_list)

        avg_mrr = sum(
            m["mrr"]
            for m in metrics_list
        ) / len(metrics_list)

        print(
            f"{method:<12}"
            f"{avg_recall_1:<15.4f}"
            f"{avg_recall_3:<15.4f}"
            f"{avg_recall_5:<15.4f}"
            f"{avg_mrr:<15.4f}"
        )

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()