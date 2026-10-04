from app.graph.graph import build_graph
from app.agents.answer_generator import ResearchAnswerGenerator


# ==========================================================
# CONFIGURATION
# ==========================================================

DOCUMENT_PATH = (
    "data/documents/documents.txt"
)

SESSION_ID = "memory-test-session"


# ==========================================================
# RUN ONE QUESTION
# ==========================================================

def run_question(
    graph,
    question: str,
    session_id: str,
):
    """
    Run one question through the LangGraph pipeline.

    The same session_id is reused so that the
    MemoryManager can load previous conversations.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "QUESTION"
    )

    print(
        "=" * 70
    )

    print(
        question
    )

    print(
        "\nSession ID:"
    )

    print(
        session_id
    )

    # ------------------------------------------------------
    # INITIAL STATE
    # ------------------------------------------------------

    initial_state = {

        # ==================================================
        # INPUT
        # ==================================================

        "question":
            question,

        "document_path":
            DOCUMENT_PATH,

        # ==================================================
        # PLANNING
        # ==================================================

        "plan":
            [],

        # ==================================================
        # WEB SEARCH
        # ==================================================

        "search_results":
            [],

        # ==================================================
        # SOURCE PROCESSING
        # ==================================================

        "evaluated_sources":
            [],

        "ranked_sources":
            [],

        # ==================================================
        # RAG RETRIEVAL
        # ==================================================

        "retrieved_documents":
            [],

        "used_context":
            [],

        "context":
            "",

        # ==================================================
        # AGENTIC RAG
        # ==================================================

        "retrieval_attempt":
            0,

        "max_retrieval_attempts":
            2,

        "context_sufficient":
            False,

        "context_evaluation":
            "",

        "context_score":
            0.0,

        "retrieval_score":
            0.0,

        "reranker_score":
            0.0,

        "relevant_chunk_count":
            0,

        "retrieval_decision":
            "",

        "retrieval_decision_reason":
            "",

        "refined_query":
            "",

        "refinement_history":
            [],

        "rag_status":
            "initialized",

        # ==================================================
        # WEB FALLBACK
        # ==================================================

        "web_fallback_used":
            False,

        # ==================================================
        # ANALYSIS
        # ==================================================

        "analysis":
            None,

        # ==================================================
        # FINAL ANSWER
        # ==================================================

        "answer":
            None,

        # ==================================================
        # CONTROL
        # ==================================================

        "status":
            "initialized",

        "errors":
            [],

        # ==================================================
        # MEMORY / METADATA
        # ==================================================

        "metadata": {
            "session_id":
                session_id,
        },
    }

    # ------------------------------------------------------
    # RUN GRAPH
    # ------------------------------------------------------

    result = graph.invoke(
        initial_state
    )

    # ======================================================
    # FINAL ANSWER
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        result.get(
            "answer",
            "No answer generated.",
        )
    )

    # ======================================================
    # AGENTIC RAG INFORMATION
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "AGENTIC RAG INFORMATION"
    )

    print(
        "=" * 70
    )

    print(
        "RAG Status:",
        result.get(
            "rag_status",
            "N/A",
        ),
    )

    print(
        "Retrieval Attempts:",
        result.get(
            "retrieval_attempt",
            0,
        ),
    )

    print(
        "Context Sufficient:",
        result.get(
            "context_sufficient",
            False,
        ),
    )

    print(
        "Retrieval Decision:",
        result.get(
            "retrieval_decision",
            "N/A",
        ),
    )

    print(
        "Retrieved Documents:",
        len(
            result.get(
                "retrieved_documents",
                [],
            )
        )
    )

    print(
        "Reranked Documents:",
        len(
            result.get(
                "used_context",
                [],
            )
        )
    )

    print(
        "Context Characters:",
        len(
            result.get(
                "context",
                "",
            )
        )
    )

    # ======================================================
    # MEMORY INFORMATION
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MEMORY INFORMATION"
    )

    print(
        "=" * 70
    )

    print(
        "Session ID:",
        session_id,
    )

    print(
        "Memory history loaded:",
        len(
            result.get(
                "metadata",
                {}
            ).get(
                "memory_history",
                []
            )
        ),
    )

    # ======================================================
    # REFINED QUERY
    # ======================================================

    refined_query = result.get(
        "refined_query"
    )

    if refined_query:

        print(
            "\nRefined Query:"
        )

        print(
            refined_query
        )

    # ======================================================
    # ERRORS
    # ======================================================

    errors = result.get(
        "errors",
        []
    )

    if errors:

        print(
            "\nErrors:"
        )

        for error in errors:

            print(
                f"- {error}"
            )

    # ======================================================
    # STATUS
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATUS:",
        result.get(
            "status",
            "N/A",
        ),
    )

    print(
        "=" * 70
    )

    return result


# ==========================================================
# HALLUCINATION DETECTION TEST
# ==========================================================

def test_hallucination_detection(
    answer_generator,
):
    """
    Test the hallucination detection component independently
    from the compiled LangGraph.

    Two cases are tested:

    1. Grounded answer
    2. Hallucinated answer
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HALLUCINATION DETECTION TEST"
    )

    print(
        "=" * 70
    )

    # ======================================================
    # TEST QUESTION
    # ======================================================

    test_question = (
        "What is Retrieval-Augmented Generation?"
    )

    # ======================================================
    # TEST EVIDENCE
    # ======================================================

    test_rag_context = """
Retrieval-Augmented Generation (RAG) combines
pre-trained parametric and non-parametric memory
for language generation.

RAG uses retrieval to provide relevant information
to the generation process.
"""

    print(
        "\nQuestion:"
    )

    print(
        test_question
    )

    print(
        "\nSupplied Evidence:"
    )

    print(
        test_rag_context
    )

    # ======================================================
    # TEST 1
    # GROUNDED ANSWER
    # ======================================================

    grounded_answer = """
Retrieval-Augmented Generation combines parametric
and non-parametric memory and uses retrieved information
during language generation.
"""

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST 1: GROUNDED ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        "\nAnswer:"
    )

    print(
        grounded_answer
    )

    try:

        hallucination_result = (
            answer_generator._detect_hallucination(
                current_question=test_question,
                final_answer=grounded_answer,
                rag_context=test_rag_context,
            )
        )

        print(
            "\nHallucination Detection Result:"
        )

        print(
            hallucination_result
        )

    except Exception as error:

        print(
            "\nERROR during grounded-answer test:"
        )

        print(
            error
        )

    # ======================================================
    # TEST 2
    # HALLUCINATED ANSWER
    # ======================================================

    hallucinated_answer = """
Retrieval-Augmented Generation was invented by OpenAI
in 2022 and uses a proprietary database containing
10 trillion documents. It guarantees 100% factual accuracy.
"""

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST 2: HALLUCINATED ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        "\nAnswer:"
    )

    print(
        hallucinated_answer
    )

    try:

        hallucination_result = (
            answer_generator._detect_hallucination(
                current_question=test_question,
                final_answer=hallucinated_answer,
                rag_context=test_rag_context,
            )
        )

        print(
            "\nHallucination Detection Result:"
        )

        print(
            hallucination_result
        )

    except Exception as error:

        print(
            "\nERROR during hallucinated-answer test:"
        )

        print(
            error
        )

    # ======================================================
    # COMPLETED
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HALLUCINATION DETECTION TEST COMPLETED"
    )

    print(
        "=" * 70
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LANGGRAPH AGENTIC RAG + MEMORY TEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nDocument:"
    )

    print(
        DOCUMENT_PATH
    )

    print(
        "\nSession:"
    )

    print(
        SESSION_ID
    )

    # ======================================================
    # BUILD GRAPH
    # ======================================================

    graph = build_graph()

    # ======================================================
    # BUILD ANSWER GENERATOR
    #
    # IMPORTANT:
    #
    # _detect_hallucination() belongs to
    # ResearchAnswerGenerator.
    #
    # It does NOT belong to CompiledStateGraph.
    # ======================================================

    answer_generator = (
        ResearchAnswerGenerator()
    )

    # ======================================================
    # QUESTION 1
    # ======================================================

    question_1 = (
        "What is Retrieval-Augmented Generation "
        "and how does it work?"
    )

    result_1 = run_question(
        graph=graph,
        question=question_1,
        session_id=SESSION_ID,
    )

    # ======================================================
    # QUESTION 2
    #
    # This question intentionally depends on the
    # previous conversation.
    # ======================================================

    question_2 = (
        "Why is it useful for improving the answers "
        "generated by language models?"
    )

    result_2 = run_question(
        graph=graph,
        question=question_2,
        session_id=SESSION_ID,
    )

    # ======================================================
    # MEMORY VERIFICATION
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MEMORY VERIFICATION"
    )

    print(
        "=" * 70
    )

    memory_history = (
        result_2
        .get(
            "metadata",
            {}
        )
        .get(
            "memory_history",
            []
        )
    )

    if memory_history:

        print(
            f"\nPrevious conversations loaded: "
            f"{len(memory_history)}"
        )

        for i, item in enumerate(
            memory_history,
            start=1,
        ):

            print(
                f"\n--- Memory {i} ---"
            )

            print(
                "Question:"
            )

            print(
                item.get(
                    "question",
                    "",
                )
            )

            print(
                "\nAnswer:"
            )

            print(
                item.get(
                    "answer",
                    "",
                )
            )

    else:

        print(
            "\nNo previous memory was loaded."
        )

    # ======================================================
    # HALLUCINATION DETECTION
    # ======================================================

    test_hallucination_detection(
        answer_generator=answer_generator,
    )

    # ======================================================
    # FINAL STATUS
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ALL TESTS COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nExpected behavior:"
    )

    print(
        "1. Question 1 is answered."
    )

    print(
        "2. Question 1 + Answer 1 are saved."
    )

    print(
        "3. Question 2 uses the same session ID."
    )

    print(
        "4. Previous conversation is loaded."
    )

    print(
        "5. Question 2 is answered."
    )

    print(
        "6. Question 2 + Answer 2 are saved."
    )

    print(
        "7. Grounded answer is detected as "
        "NO_HALLUCINATION."
    )

    print(
        "8. Hallucinated answer is detected as "
        "HALLUCINATION_DETECTED or POSSIBLE_HALLUCINATION."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()