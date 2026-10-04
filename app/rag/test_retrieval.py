from app.rag.loader import load_document
from app.rag.chunker import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import FAISSVectorStore
from app.rag.hybrid_retriever import HybridRetriever


# ==========================================================
# CONFIGURATION
# ==========================================================

DOCUMENT_PATH = "data/documents/documents.txt"

TOP_K = 5

CANDIDATE_K = 10

CHUNK_SIZE = 500

CHUNK_OVERLAP = 75


# ==========================================================
# QUESTIONS
# ==========================================================

QUESTIONS = [

    "What is Retrieval-Augmented Generation?",

    "How does RAG retrieve information?",

    "What is parametric memory in RAG?",

    "What is non-parametric memory in RAG?",

    "How does RAG update knowledge without retraining?",
]


# ==========================================================
# GROUND TRUTH
# ==========================================================

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


# ==========================================================
# RELEVANCE
# ==========================================================

def is_relevant(
    text,
    ground_truth_terms,
):
    """
    A result is considered relevant if it contains
    at least one ground-truth phrase.
    """

    text_lower = text.lower()

    for term in ground_truth_terms:

        if term.lower() in text_lower:

            return True

    return False


# ==========================================================
# RECALL@K
# ==========================================================

def recall_at_k(
    results,
    ground_truth_terms,
    k,
):
    """
    Recall@K for this evaluation setup.

    Returns:

        1.0 -> relevant result found
        0.0 -> relevant result not found
    """

    for result in results[:k]:

        if is_relevant(
            result["text"],
            ground_truth_terms,
        ):

            return 1.0

    return 0.0


# ==========================================================
# MRR
# ==========================================================

def reciprocal_rank(
    results,
    ground_truth_terms,
):
    """
    Reciprocal rank of the first relevant result.
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


# ==========================================================
# RESULT DISPLAY
# ==========================================================

def print_results(
    title,
    results,
):
    """
    Print retrieval results.
    """

    print("\n" + "-" * 70)

    print(title)

    print("-" * 70)

    if not results:

        print("No results")

        return

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nRank {rank}"
        )

        if "score" in result:

            print(
                f"Score: "
                f"{result['score']:.4f}"
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

        if "semantic_rank" in result:

            print(
                f"Semantic Rank: "
                f"{result['semantic_rank']}"
            )

        if "keyword_rank" in result:

            print(
                f"Keyword Rank: "
                f"{result['keyword_rank']}"
            )

        if "rrf_score" in result:

            print(
                f"RRF Score: "
                f"{result['rrf_score']:.6f}"
            )

        if "hybrid_score" in result:

            print(
                f"Hybrid Score: "
                f"{result['hybrid_score']:.6f}"
            )

        print(
            f"Text:\n{result['text']}"
        )


# ==========================================================
# METRIC SUMMARY
# ==========================================================

def calculate_metrics(
    results,
    ground_truth_terms,
):
    """
    Calculate all retrieval metrics.
    """

    return {

        "Recall@1": recall_at_k(
            results,
            ground_truth_terms,
            1,
        ),

        "Recall@3": recall_at_k(
            results,
            ground_truth_terms,
            3,
        ),

        "Recall@5": recall_at_k(
            results,
            ground_truth_terms,
            5,
        ),

        "MRR": reciprocal_rank(
            results,
            ground_truth_terms,
        ),
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)

    print(
        "FAISS vs BM25 vs RRF HYBRID RETRIEVAL EVALUATION"
    )

    print("=" * 70)

    # ======================================================
    # LOAD DOCUMENT
    # ======================================================

    print("\nLoading document...")

    text = load_document(
        DOCUMENT_PATH
    )

    # ======================================================
    # CHUNK DOCUMENT
    # ======================================================

    print("Chunking document...")

    chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    # ======================================================
    # EMBEDDINGS
    # ======================================================

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(
        chunks
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # ======================================================
    # FAISS
    # ======================================================

    print("\nBuilding FAISS vector store...")

    vector_store = FAISSVectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(
        embeddings,
        chunks,
    )

    # ======================================================
    # HYBRID RETRIEVER
    # ======================================================

    print(
        "\nBuilding BM25 + RRF hybrid retriever..."
    )

    hybrid_retriever = HybridRetriever(

        chunks=chunks,

        embedding_model=embedding_model,

        vector_store=vector_store,

        semantic_weight=0.5,

        keyword_weight=0.5,

        rrf_k=60,
    )

    # ======================================================
    # STORAGE FOR OVERALL METRICS
    # ======================================================

    all_metrics = {

        "FAISS": [],

        "BM25": [],

        "HYBRID": [],
    }

    # ======================================================
    # QUESTION LOOP
    # ======================================================

    for question in QUESTIONS:

        print("\n\n")

        print("=" * 70)

        print(
            f"QUESTION: {question}"
        )

        print("=" * 70)

        ground_truth_terms = GROUND_TRUTH[
            question
        ]

        # ==================================================
        # FAISS
        # ==================================================

        faiss_results = hybrid_retriever.semantic_search(
            question,
            top_k=TOP_K,
        )

        faiss_metrics = calculate_metrics(
            faiss_results,
            ground_truth_terms,
        )

        all_metrics["FAISS"].append(
            faiss_metrics
        )

        print_results(
            "FAISS / SEMANTIC RETRIEVAL",
            faiss_results,
        )

        print("\nFAISS METRICS")

        print(
            f"Recall@1 : "
            f"{faiss_metrics['Recall@1']:.4f}"
        )

        print(
            f"Recall@3 : "
            f"{faiss_metrics['Recall@3']:.4f}"
        )

        print(
            f"Recall@5 : "
            f"{faiss_metrics['Recall@5']:.4f}"
        )

        print(
            f"MRR      : "
            f"{faiss_metrics['MRR']:.4f}"
        )

        # ==================================================
        # BM25
        # ==================================================

        bm25_results = hybrid_retriever.keyword_search(
            question,
            top_k=TOP_K,
        )

        bm25_metrics = calculate_metrics(
            bm25_results,
            ground_truth_terms,
        )

        all_metrics["BM25"].append(
            bm25_metrics
        )

        print_results(
            "BM25 / KEYWORD RETRIEVAL",
            bm25_results,
        )

        print("\nBM25 METRICS")

        print(
            f"Recall@1 : "
            f"{bm25_metrics['Recall@1']:.4f}"
        )

        print(
            f"Recall@3 : "
            f"{bm25_metrics['Recall@3']:.4f}"
        )

        print(
            f"Recall@5 : "
            f"{bm25_metrics['Recall@5']:.4f}"
        )

        print(
            f"MRR      : "
            f"{bm25_metrics['MRR']:.4f}"
        )

        # ==================================================
        # HYBRID / RRF
        # ==================================================

        hybrid_results = hybrid_retriever.retrieve(
            question,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
        )

        hybrid_metrics = calculate_metrics(
            hybrid_results,
            ground_truth_terms,
        )

        all_metrics["HYBRID"].append(
            hybrid_metrics
        )

        print_results(
            "HYBRID / RRF RETRIEVAL",
            hybrid_results,
        )

        print("\nHYBRID METRICS")

        print(
            f"Recall@1 : "
            f"{hybrid_metrics['Recall@1']:.4f}"
        )

        print(
            f"Recall@3 : "
            f"{hybrid_metrics['Recall@3']:.4f}"
        )

        print(
            f"Recall@5 : "
            f"{hybrid_metrics['Recall@5']:.4f}"
        )

        print(
            f"MRR      : "
            f"{hybrid_metrics['MRR']:.4f}"
        )

        # ==================================================
        # RANK COMPARISON
        # ==================================================

        print("\nRANK COMPARISON")

        print(
            f"{'Method':<12}"
            f"{'Relevant Rank':<20}"
        )

        print("-" * 32)

        for name, results in [

            ("FAISS", faiss_results),

            ("BM25", bm25_results),

            ("HYBRID", hybrid_results),

        ]:

            relevant_rank = None

            for rank, result in enumerate(
                results,
                start=1,
            ):

                if is_relevant(
                    result["text"],
                    ground_truth_terms,
                ):

                    relevant_rank = rank

                    break

            if relevant_rank is None:

                relevant_rank = "Not Found"

            print(
                f"{name:<12}"
                f"{str(relevant_rank):<20}"
            )

    # ======================================================
    # OVERALL METRICS
    # ======================================================

    print("\n\n")

    print("=" * 70)

    print(
        "OVERALL RETRIEVAL EVALUATION"
    )

    print("=" * 70)

    overall = {}

    for method in [

        "FAISS",

        "BM25",

        "HYBRID",
    ]:

        print(
            f"\n{'-' * 70}"
        )

        print(method)

        print(
            f"{'-' * 70}"
        )

        metrics_list = all_metrics[
            method
        ]

        average_recall_1 = sum(
            x["Recall@1"]
            for x in metrics_list
        ) / len(metrics_list)

        average_recall_3 = sum(
            x["Recall@3"]
            for x in metrics_list
        ) / len(metrics_list)

        average_recall_5 = sum(
            x["Recall@5"]
            for x in metrics_list
        ) / len(metrics_list)

        average_mrr = sum(
            x["MRR"]
            for x in metrics_list
        ) / len(metrics_list)

        overall[method] = {

            "Recall@1": average_recall_1,

            "Recall@3": average_recall_3,

            "Recall@5": average_recall_5,

            "MRR": average_mrr,
        }

        print(
            f"Average Recall@1 : "
            f"{average_recall_1:.4f}"
        )

        print(
            f"Average Recall@3 : "
            f"{average_recall_3:.4f}"
        )

        print(
            f"Average Recall@5 : "
            f"{average_recall_5:.4f}"
        )

        print(
            f"Mean Reciprocal Rank: "
            f"{average_mrr:.4f}"
        )

    # ======================================================
    # FINAL COMPARISON
    # ======================================================

    print("\n\n")

    print("=" * 70)

    print("FINAL COMPARISON")

    print("=" * 70)

    print()

    print(
        f"{'Method':<12}"
        f"{'Recall@1':<15}"
        f"{'Recall@3':<15}"
        f"{'Recall@5':<15}"
        f"{'MRR':<15}"
    )

    print("-" * 72)

    for method in [

        "FAISS",

        "BM25",

        "HYBRID",
    ]:

        metrics = overall[
            method
        ]

        print(
            f"{method:<12}"
            f"{metrics['Recall@1']:<15.4f}"
            f"{metrics['Recall@3']:<15.4f}"
            f"{metrics['Recall@5']:<15.4f}"
            f"{metrics['MRR']:<15.4f}"
        )

    # ======================================================
    # EVALUATION COMPLETE
    # ======================================================

    print("\n")

    print("=" * 70)

    print(
        "EVALUATION COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()