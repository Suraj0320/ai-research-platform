from typing import TypedDict, List, Dict, Optional


class ResearchState(TypedDict, total=False):

    # ======================================================
    # INPUT
    # ======================================================

    question: str
    document_path: str

    # Session identifier used by MemoryStore
    session_id: str

    # ======================================================
    # MEMORY
    # ======================================================

    # Previous conversation loaded for this session
    memory: List[Dict]

    # Formatted previous conversation passed to agents/LLM
    memory_context: str

    # ======================================================
    # PLANNING
    # ======================================================

    plan: List[str]

    # ======================================================
    # WEB SEARCH
    # ======================================================

    search_results: List[Dict]

    # ======================================================
    # SOURCE PROCESSING
    # ======================================================

    evaluated_sources: List[Dict]

    ranked_sources: List[Dict]

    # ======================================================
    # RAG RETRIEVAL
    # ======================================================

    retrieved_documents: List[Dict]

    # Documents actually selected after retrieval/reranking
    used_context: List[Dict]

    # Combined text passed to analysis/answer generation
    context: str

    # ======================================================
    # AGENTIC RAG
    # ======================================================

    # Current retrieval attempt
    retrieval_attempt: int

    # Maximum number of retrieval attempts
    max_retrieval_attempts: int

    # ======================================================
    # CONTEXT EVALUATION
    # ======================================================

    context_sufficient: bool

    context_evaluation: str

    # Optional quantitative retrieval signals
    context_score: float

    retrieval_score: float

    reranker_score: float

    # Number of useful chunks
    relevant_chunk_count: int

    # ======================================================
    # RETRIEVAL DECISION
    # ======================================================

    # Expected values:
    #
    # ANSWER
    # REFINE
    # WEB_SEARCH

    retrieval_decision: str

    retrieval_decision_reason: str

    # ======================================================
    # QUERY REFINEMENT
    # ======================================================

    refined_query: str

    refinement_history: List[str]

    # ======================================================
    # RAG STATUS
    # ======================================================

    rag_status: str

    # ======================================================
    # WEB FALLBACK
    # ======================================================

    web_fallback_used: bool

    # ======================================================
    # ANALYSIS
    # ======================================================

    analysis: Optional[str]

    # ======================================================
    # FINAL ANSWER
    # ======================================================

    answer: Optional[str]

    # ======================================================
    # MEMORY SAVE STATUS
    # ======================================================

    # Indicates whether the current Q&A was saved
    memory_saved: bool

    # ======================================================
    # CONTROL / DEBUGGING
    # ======================================================

    status: str

    errors: List[str]

    # ======================================================
    # METADATA
    # ======================================================

    metadata: Dict