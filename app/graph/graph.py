from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from app.graph.state import ResearchState

from app.graph.nodes import (
    planner_node,
    search_node,
    evaluate_sources_node,
    rank_sources_node,
    rag_retrieval_node,
    evaluate_context_node,
    retrieval_decision_node,
    refine_query_node,
    web_fallback_node,
    analysis_node,
    answer_node,
)


# ==========================================================
# RETRIEVAL ROUTER
# ==========================================================

def route_after_retrieval(
    state: ResearchState,
) -> str:

    decision = state.get(
        "retrieval_decision",
        "ANSWER",
    )

    if decision == "REFINE":
        return "refine"

    if decision == "WEB_SEARCH":
        return "web"

    return "answer"


# ==========================================================
# INITIALIZE STATE
# ==========================================================

def initialize_state(
    state: ResearchState,
) -> ResearchState:

    """
    Initialize fields required by Agentic RAG.

    This keeps the first retrieval attempt deterministic
    and avoids relying on missing optional state values.
    """

    return {
        **state,

        # Agentic RAG
        "retrieval_attempt": 0,
        "max_retrieval_attempts": state.get(
            "max_retrieval_attempts",
            2,
        ),

        "context_sufficient": False,
        "context_evaluation": "",

        "retrieval_decision": "",
        "retrieval_decision_reason": "",

        "refined_query": "",
        "refinement_history": [],

        "rag_status": "initialized",

        # Web fallback
        "web_fallback_used": False,

        # Output
        "analysis": None,
        "answer": None,

        # Debugging
        "status": "initialized",
        "errors": state.get(
            "errors",
            [],
        ),

        # Metadata
        "metadata": state.get(
            "metadata",
            {},
        ),
    }


# ==========================================================
# BUILD GRAPH
# ==========================================================

def build_graph():

    graph = StateGraph(
        ResearchState
    )

    # ======================================================
    # INITIALIZATION
    # ======================================================

    graph.add_node(
        "initialize",
        initialize_state,
    )

    # ======================================================
    # RESEARCH NODES
    # ======================================================

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_node(
        "search",
        search_node,
    )

    graph.add_node(
        "evaluate_sources",
        evaluate_sources_node,
    )

    graph.add_node(
        "rank_sources",
        rank_sources_node,
    )

    # ======================================================
    # RAG NODES
    # ======================================================

    graph.add_node(
        "rag_retrieval",
        rag_retrieval_node,
    )

    graph.add_node(
        "evaluate_context",
        evaluate_context_node,
    )

    graph.add_node(
        "retrieval_decision",
        retrieval_decision_node,
    )

    graph.add_node(
        "refine_query",
        refine_query_node,
    )

    # ======================================================
    # WEB FALLBACK
    # ======================================================

    graph.add_node(
        "web_fallback",
        web_fallback_node,
    )

    # ======================================================
    # FINAL RESPONSE
    # ======================================================

    graph.add_node(
        "analysis",
        analysis_node,
    )

    graph.add_node(
        "answer",
        answer_node,
    )

    # ======================================================
    # INITIAL FLOW
    #
    # START
    #   ↓
    # initialize
    #   ↓
    # planner
    #   ↓
    # search
    #   ↓
    # evaluate_sources
    #   ↓
    # rank_sources
    # ======================================================

    graph.add_edge(
        START,
        "initialize",
    )

    graph.add_edge(
        "initialize",
        "planner",
    )

    graph.add_edge(
        "planner",
        "search",
    )

    graph.add_edge(
        "search",
        "evaluate_sources",
    )

    graph.add_edge(
        "evaluate_sources",
        "rank_sources",
    )

    # ======================================================
    # RAG FLOW
    #
    # rank_sources
    #      ↓
    # rag_retrieval
    #      ↓
    # evaluate_context
    #      ↓
    # retrieval_decision
    # ======================================================

    graph.add_edge(
        "rank_sources",
        "rag_retrieval",
    )

    graph.add_edge(
        "rag_retrieval",
        "evaluate_context",
    )

    graph.add_edge(
        "evaluate_context",
        "retrieval_decision",
    )

    # ======================================================
    # AGENTIC ROUTING
    #
    # retrieval_decision
    #
    #       ├── ANSWER
    #       │      ↓
    #       │   analysis
    #       │
    #       ├── REFINE
    #       │      ↓
    #       │   refine_query
    #       │      ↓
    #       │   rag_retrieval
    #       │
    #       └── WEB_SEARCH
    #              ↓
    #          web_fallback
    #              ↓
    #           analysis
    #
    # ======================================================

    graph.add_conditional_edges(
        "retrieval_decision",
        route_after_retrieval,
        {
            "refine": "refine_query",
            "web": "web_fallback",
            "answer": "analysis",
        },
    )

    # ======================================================
    # AGENTIC RETRIEVAL LOOP
    #
    # refine_query
    #      ↓
    # rag_retrieval
    #      ↓
    # evaluate_context
    #      ↓
    # retrieval_decision
    #
    # Maximum attempts are controlled by
    # retrieval_decision_node.
    #
    # ======================================================

    graph.add_edge(
        "refine_query",
        "rag_retrieval",
    )

    # ======================================================
    # WEB FALLBACK
    # ======================================================

    graph.add_edge(
        "web_fallback",
        "analysis",
    )

    # ======================================================
    # FINAL RESPONSE
    #
    # analysis
    #    ↓
    # answer
    #    ↓
    # END
    #
    # answer_node also saves conversation memory.
    #
    # ======================================================

    graph.add_edge(
        "analysis",
        "answer",
    )

    graph.add_edge(
        "answer",
        END,
    )

    # ======================================================
    # COMPILE
    # ======================================================

    return graph.compile()