from app.graph.state import ResearchState as GraphState

from app.agents.state import ResearchState as AgentState
from app.agents.planner import ResearchPlanner
from app.agents.analyzer import ResearchAnalyzer
from app.agents.answer_generator import ResearchAnswerGenerator
from app.agents.source_evaluator import SourceEvaluator
from app.agents.source_ranker import SourceRanker

from app.tools.web_search import WebSearchTool

from app.rag.pipeline import RAGPipeline

from app.memory.memory_manager import MemoryManager


# ==========================================================
# SHARED COMPONENTS
# ==========================================================

planner = ResearchPlanner()

search_tool = WebSearchTool()

source_evaluator = SourceEvaluator()

source_ranker = SourceRanker()

analyzer = ResearchAnalyzer()

answer_generator = ResearchAnswerGenerator()

rag_pipeline = RAGPipeline()

memory_manager = MemoryManager()


# ==========================================================
# AGENTIC RAG CONFIGURATION
# ==========================================================

MAX_RETRIEVAL_ATTEMPTS = 2


# ==========================================================
# MEMORY HELPERS
# ==========================================================

def get_session_id(
    state: GraphState,
) -> str | None:

    metadata = state.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        return None

    session_id = metadata.get(
        "session_id"
    )

    if not session_id:
        return None

    return str(session_id)


def load_session_memory(
    state: GraphState,
) -> tuple[list, str]:

    session_id = get_session_id(
        state
    )

    if not session_id:
        return [], ""

    history = memory_manager.load_memory(
        session_id
    )

    memory_context = memory_manager.format_memory(
        history
    )

    return history, memory_context


def save_session_memory(
    state: GraphState,
    answer: str,
) -> None:

    session_id = get_session_id(
        state
    )

    if not session_id:
        return

    question = state.get(
        "question",
        ""
    )

    if not question or not answer:
        return

    memory_manager.save_memory(
        session_id=session_id,
        question=question,
        answer=answer,
    )


# ==========================================================
# STEP 1
# PLANNER NODE
# ==========================================================

def planner_node(
    state: GraphState,
) -> GraphState:

    question = state["question"]

    print(
        "\n[LANGGRAPH] PLANNER"
    )

    print(
        f"Question: {question}"
    )

    # ------------------------------------------------------
    # Load conversation memory
    # ------------------------------------------------------

    memory_history, memory_context = (
        load_session_memory(state)
    )

    if memory_history:

        print(
            f"[MEMORY] Loaded "
            f"{len(memory_history)} previous "
            f"conversation(s)"
        )

    else:

        print(
            "[MEMORY] No previous conversation"
        )

    # ------------------------------------------------------
    # Convert LangGraph state -> Agent state
    # ------------------------------------------------------

    agent_state = AgentState(
        question=question
    )

    # ------------------------------------------------------
    # Store memory inside Agent metadata
    # ------------------------------------------------------

    agent_state.metadata[
        "memory_history"
    ] = memory_history

    agent_state.metadata[
        "memory_context"
    ] = memory_context

    # ------------------------------------------------------
    # Generate research plan
    # ------------------------------------------------------

    plan = planner.create_plan(
        agent_state
    )

    print(
        "\nGenerated Plan:"
    )

    for i, step in enumerate(
        plan,
        start=1,
    ):

        print(
            f"{i}. {step}"
        )

    # ======================================================
    # MEMORY FIX
    #
    # Put the loaded memory back into the LangGraph state.
    #
    # Previously memory was loaded here but discarded when
    # returning the state. That is why test_graph.py showed:
    #
    # Memory history loaded: 0
    #
    # even though the planner printed:
    #
    # [MEMORY] Loaded 1 previous conversation(s)
    # ======================================================

    metadata = dict(
        state.get(
            "metadata",
            {}
        )
    )

    metadata[
        "memory_history"
    ] = memory_history

    metadata[
        "memory_context"
    ] = memory_context

    return {
        **state,

        "plan":
            plan,

        "metadata":
            metadata,

        "status":
            "planning_completed",
    }


# ==========================================================
# STEP 2
# SEARCH NODE
# ==========================================================

def search_node(
    state: GraphState,
) -> GraphState:

    print(
        "\n[LANGGRAPH] SEARCH"
    )

    results = []

    # ------------------------------------------------------
    # Maximum search objectives
    # ------------------------------------------------------

    search_steps = state.get(
        "plan",
        []
    )[:3]

    for step in search_steps:

        print(
            "\nSearch objective:"
        )

        print(step)

        query = f"""
Research question:
{state["question"]}

Research objective:
{step}

Find relevant, reliable information for this
research objective.

Prefer authoritative and recent sources
when appropriate.
"""

        result = search_tool.search(
            query
        )

        results.append(
            result
        )

    print(
        f"\nSearch completed: "
        f"{len(results)} search results"
    )

    return {
        **state,

        "search_results":
            results,

        "status":
            "search_completed",
    }


# ==========================================================
# STEP 3
# SOURCE EVALUATION
# ==========================================================

def evaluate_sources_node(
    state: GraphState,
) -> GraphState:

    print(
        "\n[LANGGRAPH] SOURCE EVALUATION"
    )

    evaluated = source_evaluator.evaluate(
        state.get(
            "search_results",
            []
        )
    )

    print(
        f"Evaluated sources: "
        f"{len(evaluated)}"
    )

    return {
        **state,

        "evaluated_sources":
            evaluated,

        "status":
            "evaluation_completed",
    }


# ==========================================================
# STEP 4
# SOURCE RANKING
# ==========================================================

def rank_sources_node(
    state: GraphState,
) -> GraphState:

    print(
        "\n[LANGGRAPH] SOURCE RANKING"
    )

    ranked = source_ranker.rank(
        state.get(
            "evaluated_sources",
            []
        )
    )

    print(
        f"Ranked sources: "
        f"{len(ranked)}"
    )

    return {
        **state,

        "ranked_sources":
            ranked,

        "status":
            "ranking_completed",
    }


# ==========================================================
# STEP 5
# RAG RETRIEVAL
# ==========================================================

def rag_retrieval_node(
    state: GraphState,
) -> GraphState:

    print(
        "\n[LANGGRAPH] RAG RETRIEVAL"
    )

    # ------------------------------------------------------
    # Determine retrieval attempt
    # ------------------------------------------------------

    current_attempt = state.get(
        "retrieval_attempt",
        0,
    )

    if current_attempt == 0:

        current_attempt = 1

    # ------------------------------------------------------
    # Determine retrieval query
    # ------------------------------------------------------

    query = (
        state.get(
            "refined_query"
        )
        or state["question"]
    )

    if current_attempt > 1:

        print(
            "[LANGGRAPH] Refined retrieval"
        )

    else:

        print(
            "[LANGGRAPH] Initial retrieval"
        )

    print(
        f"[LANGGRAPH] Retrieval attempt: "
        f"{current_attempt}"
    )

    print(
        f"[LANGGRAPH] Retrieval query:\n"
        f"{query}"
    )

    # ------------------------------------------------------
    # Check document
    # ------------------------------------------------------

    document_path = state.get(
        "document_path"
    )

    if not document_path:

        print(
            "[LANGGRAPH] No document provided."
        )

        return {
            **state,

            "retrieved_documents":
                [],

            "used_context":
                [],

            "context":
                "",

            "retrieval_attempt":
                current_attempt,

            "rag_status":
                "no_document",

            "status":
                "rag_retrieval_skipped",
        }

    # ------------------------------------------------------
    # Run RAG pipeline
    # ------------------------------------------------------

    rag_result = rag_pipeline.generate(
        question=query,
        document_path=document_path,
        candidate_k=30,
        top_k=20,
        context_k=5,
        retrieval_method="hybrid",
        rerank=True,
    )

    retrieved_documents = (
        rag_result.get(
            "retrieved_documents",
            []
        )
    )

    context_results = (
        rag_result.get(
            "used_context",
            []
        )
    )

    context = (
        rag_result.get(
            "context",
            ""
        )
    )

    # ------------------------------------------------------
    # Print retrieval information
    # ------------------------------------------------------

    print(
        f"Retrieved documents: "
        f"{len(retrieved_documents)}"
    )

    print(
        f"Context documents: "
        f"{len(context_results)}"
    )

    print(
        f"Context characters: "
        f"{len(context)}"
    )

    # ------------------------------------------------------
    # Print retrieved chunks
    # ------------------------------------------------------

    for i, document in enumerate(
        context_results,
        start=1,
    ):

        print(
            f"\n--- Retrieved Chunk {i} ---"
        )

        print(
            f"Retriever score: "
            f"{document.get('score', 'N/A')}"
        )

        print(
            f"Reranker score: "
            f"{document.get('reranker_score', 'N/A')}"
        )

        print(
            document.get(
                "text",
                ""
            )
        )

    return {
        **state,

        "retrieved_documents":
            retrieved_documents,

        "used_context":
            context_results,

        "context":
            context,

        "retrieval_attempt":
            current_attempt,

        "rag_status":
            "retrieved",

        "status":
            "rag_retrieval_completed",
    }


# ==========================================================
# STEP 6
# CONTEXT EVALUATION
# ==========================================================

def evaluate_context_node(
    state: GraphState,
) -> GraphState:

    print(
        "\n[LANGGRAPH] CONTEXT EVALUATION"
    )

    context = state.get(
        "context",
        ""
    )

    # ------------------------------------------------------
    # No context
    # ------------------------------------------------------

    if not context.strip():

        print(
            "Context evaluation: "
            "INSUFFICIENT"
        )

        return {
            **state,

            "context_sufficient":
                False,

            "context_evaluation":
                "No useful context was retrieved.",

            "relevant_chunk_count":
                0,

            "status":
                "context_insufficient",
        }

    # ------------------------------------------------------
    # Deterministic signal
    # ------------------------------------------------------

    used_context = state.get(
        "used_context",
        []
    )

    relevant_chunk_count = len(
        used_context
    )

    # ------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------

    evaluation_prompt = f"""
You are evaluating retrieved context
for a research question.

QUESTION:
{state["question"]}

RETRIEVED CONTEXT:
{context}

Determine whether the retrieved context
contains enough relevant information to
answer the question accurately.

Return ONLY one of:

SUFFICIENT

or

INSUFFICIENT

Do not provide an explanation.
"""

    try:

        response = analyzer.llm.generate(
            evaluation_prompt
        )

        decision = (
            response
            .strip()
            .upper()
        )

    except Exception as e:

        print(
            f"Context evaluation error: {e}"
        )

        decision = "INSUFFICIENT"

    # ------------------------------------------------------
    # Normalize result
    # ------------------------------------------------------

    sufficient = (
        decision == "SUFFICIENT"
        or decision.startswith(
            "SUFFICIENT"
        )
    )

    print(
        "Context evaluation:",
        "SUFFICIENT"
        if sufficient
        else "INSUFFICIENT"
    )

    print(
        f"{relevant_chunk_count} "
        f"relevant chunks and "
        f"{len(context)} context characters "
        f"available."
    )

    return {
        **state,

        "context_sufficient":
            sufficient,

        "context_evaluation":
            (
                "SUFFICIENT"
                if sufficient
                else "INSUFFICIENT"
            ),

        "relevant_chunk_count":
            relevant_chunk_count,

        "status":
            "context_evaluated",
    }


# ==========================================================
# STEP 7
# RETRIEVAL DECISION
# ==========================================================

def retrieval_decision_node(
    state: GraphState,
) -> GraphState:

    print(
        "\n[LANGGRAPH] RETRIEVAL DECISION"
    )

    sufficient = state.get(
        "context_sufficient",
        False
    )

    attempt = state.get(
        "retrieval_attempt",
        1
    )

    max_attempts = state.get(
        "max_retrieval_attempts",
        MAX_RETRIEVAL_ATTEMPTS
    )

    # ------------------------------------------------------
    # Context sufficient
    # ------------------------------------------------------

    if sufficient:

        decision = "ANSWER"

        reason = (
            "Retrieved context is sufficient "
            "to answer the current question."
        )

    # ------------------------------------------------------
    # Context insufficient + retry available
    # ------------------------------------------------------

    elif attempt < max_attempts:

        decision = "REFINE"

        reason = (
            "Retrieved context is insufficient "
            "and another retrieval attempt is available."
        )

    # ------------------------------------------------------
    # Retrieval exhausted
    # ------------------------------------------------------

    else:

        decision = "WEB_SEARCH"

        reason = (
            "Local retrieval remains insufficient "
            "after the maximum number of attempts."
        )

    print(
        f"Decision: {decision}"
    )

    print(
        f"Reason: {reason}"
    )

    return {
        **state,

        "retrieval_decision":
            decision,

        "retrieval_decision_reason":
            reason,

        "rag_status":
            (
                "retrieval_completed"
                if decision == "ANSWER"
                else (
                    "refinement_required"
                    if decision == "REFINE"
                    else "web_fallback"
                )
            ),

        "status":
            "retrieval_decision_completed",
    }


# ==========================================================
# STEP 8
# QUERY REFINEMENT
# ==========================================================

def refine_query_node(
    state: GraphState,
) -> GraphState:

    print(
        "\n[LANGGRAPH] QUERY REFINEMENT"
    )

    question = state[
        "question"
    ]

    context = state.get(
        "context",
        ""
    )

    previous_query = (
        state.get(
            "refined_query"
        )
        or question
    )

    refinement_prompt = f"""
You are a retrieval query refinement agent.

Original research question:
{question}

Previous retrieval query:
{previous_query}

Previously retrieved context:
{context}

The previous retrieval did not provide
sufficient information.

Create ONE improved retrieval query that
focuses specifically on the missing
information.

Rules:

- Preserve the original intent.
- Make the query more precise.
- Focus on information missing from the
  retrieved context.
- Do not answer the question.
- Return ONLY the improved query.
"""

    try:

        refined_query = (
            analyzer.llm.generate(
                refinement_prompt
            )
            .strip()
        )

    except Exception as e:

        print(
            f"Query refinement error: {e}"
        )

        refined_query = question

    if not refined_query:

        refined_query = question

    # ------------------------------------------------------
    # Update refinement history
    # ------------------------------------------------------

    refinement_history = list(
        state.get(
            "refinement_history",
            []
        )
    )

    refinement_history.append(
        refined_query
    )

    next_attempt = (
        state.get(
            "retrieval_attempt",
            1
        )
        + 1
    )

    print(
        "Refined query:"
    )

    print(
        refined_query
    )

    print(
        f"Next retrieval attempt: "
        f"{next_attempt}"
    )

    return {
        **state,

        "refined_query":
            refined_query,

        "refinement_history":
            refinement_history,

        "retrieval_attempt":
            next_attempt,

        "status":
            "query_refined",
    }


# ==========================================================
# STEP 9
# WEB FALLBACK
# ==========================================================

def web_fallback_node(
    state: GraphState,
) -> GraphState:

    print(
        "\n[LANGGRAPH] WEB FALLBACK"
    )

    query = (
        state.get(
            "refined_query"
        )
        or state["question"]
    )

    print(
        f"Web fallback query:\n{query}"
    )

    try:

        result = search_tool.search(
            query
        )

        existing_results = list(
            state.get(
                "search_results",
                []
            )
        )

        existing_results.append(
            result
        )

        return {
            **state,

            "search_results":
                existing_results,

            "web_fallback_used":
                True,

            "rag_status":
                "web_fallback_completed",

            "status":
                "web_fallback_completed",
        }

    except Exception as e:

        print(
            f"Web fallback error: {e}"
        )

        errors = list(
            state.get(
                "errors",
                []
            )
        )

        errors.append(
            f"Web fallback error: {e}"
        )

        return {
            **state,

            "web_fallback_used":
                True,

            "errors":
                errors,

            "status":
                "web_fallback_failed",
        }


# ==========================================================
# STEP 10
# ANALYSIS NODE
# ==========================================================

def analysis_node(
    state: GraphState,
) -> GraphState:

    print(
        "\n[LANGGRAPH] ANALYSIS"
    )

    # ------------------------------------------------------
    # Convert LangGraph state -> Agent state
    #
    # IMPORTANT:
    # Only pass fields that ResearchState actually accepts.
    #
    # Do NOT pass:
    # retrieved_documents
    # used_context
    # context
    #
    # Those belong to GraphState and are stored in metadata.
    # ------------------------------------------------------

    agent_state = AgentState(
        question=state["question"],

        plan=state.get(
            "plan",
            []
        ),

        search_results=state.get(
            "search_results",
            []
        ),

        evaluated_sources=state.get(
            "evaluated_sources",
            []
        ),

        ranked_sources=state.get(
            "ranked_sources",
            []
        ),
    )

    # ------------------------------------------------------
    # Load memory
    # ------------------------------------------------------

    memory_history, memory_context = (
        load_session_memory(state)
    )

    # ------------------------------------------------------
    # Add RAG information to metadata
    # ------------------------------------------------------

    agent_state.metadata[
        "rag_context"
    ] = state.get(
        "context",
        ""
    )

    agent_state.metadata[
        "retrieved_documents"
    ] = state.get(
        "retrieved_documents",
        []
    )

    agent_state.metadata[
        "used_context"
    ] = state.get(
        "used_context",
        []
    )

    agent_state.metadata[
        "retrieval_attempt"
    ] = state.get(
        "retrieval_attempt",
        1
    )

    agent_state.metadata[
        "context_sufficient"
    ] = state.get(
        "context_sufficient",
        False
    )

    agent_state.metadata[
        "context_evaluation"
    ] = state.get(
        "context_evaluation",
        ""
    )

    agent_state.metadata[
        "relevant_chunk_count"
    ] = state.get(
        "relevant_chunk_count",
        0
    )

    # ------------------------------------------------------
    # Add memory
    # ------------------------------------------------------

    agent_state.metadata[
        "memory_history"
    ] = memory_history

    agent_state.metadata[
        "memory_context"
    ] = memory_context

    # ======================================================
    # ANALYSIS
    # ======================================================

    print(
        f"[ANALYSIS] Search results: "
        f"{len(state.get('search_results', []))}"
    )

    print(
        f"[ANALYSIS] Ranked sources: "
        f"{len(state.get('ranked_sources', []))}"
    )

    print(
        f"[ANALYSIS] Retrieved documents: "
        f"{len(state.get('retrieved_documents', []))}"
    )

    print(
        f"[ANALYSIS] RAG context characters: "
        f"{len(state.get('context', ''))}"
    )

    analysis = analyzer.analyze(
        agent_state
    )

    return {
        **state,

        "analysis":
            analysis,

        "status":
            "analysis_completed",
    }


# ==========================================================
# STEP 11
# ANSWER NODE
# ==========================================================

def answer_node(
    state: GraphState,
) -> GraphState:

    print(
        "\n[LANGGRAPH] ANSWER GENERATION"
    )

    # ------------------------------------------------------
    # Convert LangGraph state -> Agent statea
    #
    # IMPORTANT:
    # Only pass fields supported by AgentState.
    # RAG-specific information goes into metadata.
    # ------------------------------------------------------

    agent_state = AgentState(
        question=state["question"],

        plan=state.get(
            "plan",
            []
        ),

        search_results=state.get(
            "search_results",
            []
        ),

        evaluated_sources=state.get(
            "evaluated_sources",
            []
        ),

        ranked_sources=state.get(
            "ranked_sources",
            []
        ),

        analysis=state.get(
            "analysis"
        ),
    )

    # ------------------------------------------------------
    # Load memory
    # ------------------------------------------------------

    memory_history, memory_context = (
        load_session_memory(state)
    )

    # ------------------------------------------------------
    # Add Agentic RAG information
    # ------------------------------------------------------

    agent_state.metadata[
        "rag_context"
    ] = state.get(
        "context",
        ""
    )

    agent_state.metadata[
        "retrieved_documents"
    ] = state.get(
        "retrieved_documents",
        []
    )

    agent_state.metadata[
        "used_context"
    ] = state.get(
        "used_context",
        []
    )

    agent_state.metadata[
        "retrieval_attempt"
    ] = state.get(
        "retrieval_attempt",
        1
    )

    agent_state.metadata[
        "context_sufficient"
    ] = state.get(
        "context_sufficient",
        False
    )

    agent_state.metadata[
        "context_evaluation"
    ] = state.get(
        "context_evaluation",
        ""
    )

    agent_state.metadata[
        "relevant_chunk_count"
    ] = state.get(
        "relevant_chunk_count",
        0
    )

    agent_state.metadata[
        "retrieval_decision"
    ] = state.get(
        "retrieval_decision",
        "ANSWER"
    )

    agent_state.metadata[
        "retrieval_decision_reason"
    ] = state.get(
        "retrieval_decision_reason",
        ""
    )

    # ------------------------------------------------------
    # Add memory
    # ------------------------------------------------------

    agent_state.metadata[
        "memory_history"
    ] = memory_history

    agent_state.metadata[
        "memory_context"
    ] = memory_context

    # ======================================================
    # PRINT DEBUG INFORMATION
    # ======================================================

    print(
        f"[ANSWER] Ranked sources: "
        f"{len(state.get('ranked_sources', []))}"
    )

    print(
        f"[ANSWER] Retrieved documents: "
        f"{len(state.get('retrieved_documents', []))}"
    )

    print(
        f"[ANSWER] RAG context characters: "
        f"{len(state.get('context', ''))}"
    )

    print(
        f"[ANSWER] Analysis available: "
        f"{state.get('analysis') is not None}"
    )

    print(
        f"[ANSWER] Memory conversations available: "
        f"{len(memory_history)}"
    )

    # ======================================================
    # GENERATE FINAL ANSWER
    # ======================================================

    answer = answer_generator.generate(
        agent_state
    )

    # ======================================================
    # SAVE CONVERSATION MEMORY
    #
    # Save AFTER generating the answer.
    #
    # This prevents the current question/answer from
    # appearing as previous memory during generation.
    # ======================================================

    save_session_memory(
        state=state,
        answer=answer,
    )

    if get_session_id(state):

        print(
            "\n[MEMORY] Conversation saved."
        )

    else:

        print(
            "\n[MEMORY] No session_id; "
            "conversation not persisted."
        )

    # ======================================================
    # MEMORY FIX
    #
    # Keep the memory that was loaded BEFORE generating
    # this answer in the returned metadata.
    #
    # This allows test_graph.py to verify that Question 2
    # actually loaded Question 1's conversation.
    # ======================================================

    metadata = dict(
        state.get(
            "metadata",
            {}
        )
    )

    metadata[
        "memory_history"
    ] = memory_history

    metadata[
        "memory_context"
    ] = memory_context

    return {
        **state,

        "answer":
            answer,

        "metadata":
            metadata,

        "status":
            "completed",
    }