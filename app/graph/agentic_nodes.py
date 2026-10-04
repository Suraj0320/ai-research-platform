from app.graph.state import ResearchState


def evaluate_evidence_node(
    state: ResearchState,
) -> ResearchState:

    print("\n[AGENTIC RAG] EVIDENCE EVALUATION")

    ranked_sources = state.get(
        "ranked_sources",
        []
    )

    attempt = state.get(
        "retrieval_attempt",
        1
    )

    # Simple initial criterion.
    # We will make this smarter later.
    evidence_sufficient = len(ranked_sources) >= 5

    print(
        f"Retrieval attempt: {attempt}"
    )

    print(
        f"Ranked sources: {len(ranked_sources)}"
    )

    print(
        f"Evidence sufficient: "
        f"{evidence_sufficient}"
    )

    return {
        **state,
        "evidence_sufficient": evidence_sufficient,
        "status": "evidence_evaluated",
    }

def route_after_evaluation(
    state: ResearchState,
):

    if state.get("evidence_sufficient", False):

        print(
            "\n[AGENTIC RAG] Evidence is sufficient"
        )

        return "answer"

    max_attempts = state.get(
        "max_retrieval_attempts",
        2
    )

    current_attempt = state.get(
        "retrieval_attempt",
        1
    )

    if current_attempt >= max_attempts:

        print(
            "\n[AGENTIC RAG] Maximum retrieval attempts reached"
        )

        return "answer"

    print(
        "\n[AGENTIC RAG] Evidence insufficient"
    )

    print(
        "[AGENTIC RAG] Refining retrieval..."
    )

    return "refine"


def refine_query_node(
    state: ResearchState,
) -> ResearchState:

    print("\n[AGENTIC RAG] QUERY REFINEMENT")

    question = state["question"]

    attempt = state.get(
        "retrieval_attempt",
        1
    )

    refinement_query = (
        f"{question} "
        "Provide additional authoritative evidence, "
        "technical details, and supporting sources."
    )

    print(
        f"Refined query:\n{refinement_query}"
    )

    return {
        **state,
        "question": refinement_query,
        "refinement_query": refinement_query,
        "retrieval_attempt": attempt + 1,
        "status": "query_refined",
    }