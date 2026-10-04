from app.llm.gateway.gateway import LLMGateway
from app.agents.state import ResearchState


class ResearchAnalyzer:

    def __init__(self):
        self.llm = LLMGateway()

    def analyze(
        self,
        state: ResearchState,
    ):

        # ==================================================
        # METADATA
        # ==================================================

        metadata = getattr(
            state,
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        # ==================================================
        # BUILD SEARCH EVIDENCE CONTEXT
        # ==================================================

        research_context = ""

        search_results = getattr(
            state,
            "search_results",
            [],
        )

        for i, result in enumerate(
            search_results,
            1,
        ):

            if not result.get(
                "success",
                True,
            ):

                research_context += f"""
SEARCH RESULT {i}

Query:
{result.get("query", "")}

Search failed:
{result.get(
    "error",
    "Unknown search error",
)}

----------------------------------------
"""

                continue

            research_context += f"""
SEARCH RESULT {i}

Query:
{result.get("query", "")}

Search Answer:
{result.get("answer", "")}

Citations:
{result.get("citations", [])}

----------------------------------------
"""

        # ==================================================
        # BUILD RANKED SOURCE CONTEXT
        # ==================================================

        source_context = ""

        ranked_sources = getattr(
            state,
            "ranked_sources",
            [],
        )

        for source in ranked_sources:

            source_context += f"""
Rank:
{source.get("rank", "UNKNOWN")}

Score:
{source.get("score", "UNKNOWN")}

Title:
{source.get("title", "")}

URL:
{source.get("url", "")}

Authority:
{source.get("authority", "UNKNOWN")}

Relevance:
{source.get("relevance", "UNKNOWN")}

Usefulness:
{source.get("usefulness", "UNKNOWN")}

----------------------------------------
"""

        # ==================================================
        # RAG CONTEXT
        # ==================================================

        rag_context = metadata.get(
            "rag_context",
            "",
        )

        if not rag_context:

            rag_context = (
                "No local RAG context available."
            )

        # ==================================================
        # RAG RETRIEVED DOCUMENTS
        #
        # IMPORTANT:
        # These are stored in AgentState.metadata
        # because AgentState does not have these as
        # constructor fields.
        # ==================================================

        retrieved_documents = metadata.get(
            "retrieved_documents",
            [],
        )

        rag_documents_context = ""

        for i, document in enumerate(
            retrieved_documents,
            1,
        ):

            rag_documents_context += f"""
RAG DOCUMENT {i}

Retriever Score:
{document.get(
    "score",
    "UNKNOWN",
)}

Reranker Score:
{document.get(
    "reranker_score",
    "UNKNOWN",
)}

Content:
{document.get(
    "text",
    "",
)}

----------------------------------------
"""

        if not rag_documents_context:

            rag_documents_context = (
                "No retrieved RAG documents available."
            )

        # ==================================================
        # USED CONTEXT
        # ==================================================

        used_context = metadata.get(
            "used_context",
            [],
        )

        used_context_count = len(
            used_context
        )

        # ==================================================
        # AGENTIC RAG INFORMATION
        # ==================================================

        retrieval_attempt = metadata.get(
            "retrieval_attempt",
            0,
        )

        context_sufficient = metadata.get(
            "context_sufficient",
            False,
        )

        context_evaluation = metadata.get(
            "context_evaluation",
            "",
        )

        retrieval_decision = metadata.get(
            "retrieval_decision",
            "",
        )

        retrieval_decision_reason = metadata.get(
            "retrieval_decision_reason",
            "",
        )

        refined_query = metadata.get(
            "refined_query",
            "",
        )

        # ==================================================
        # MEMORY
        # ==================================================

        memory_context = metadata.get(
            "memory_context",
            "",
        )

        # ==================================================
        # ANALYSIS PROMPT
        # ==================================================

        prompt = f"""
You are a rigorous research analyst.

==================================================
CURRENT RESEARCH QUESTION
==================================================

{state.question}


==================================================
PREVIOUS CONVERSATION MEMORY
==================================================

{memory_context}

Use previous conversation only when it is
relevant to understanding the current question.

Do not treat previous conversation as factual
evidence unless the same information is also
supported by the current research evidence.


==================================================
WEB RESEARCH EVIDENCE
==================================================

{research_context}


==================================================
RANKED WEB SOURCES
==================================================

{source_context}


==================================================
LOCAL RAG CONTEXT
==================================================

{rag_context}


==================================================
LOCAL RAG RETRIEVED DOCUMENTS
==================================================

{rag_documents_context}


==================================================
AGENTIC RAG INFORMATION
==================================================

Retrieval attempt:
{retrieval_attempt}

Context sufficient:
{context_sufficient}

Context evaluation:
{context_evaluation}

Retrieval decision:
{retrieval_decision}

Retrieval decision reason:
{retrieval_decision_reason}

Refined query:
{refined_query}

Number of selected context documents:
{used_context_count}


==================================================
TASK
==================================================

Analyze the research question using ONLY the
evidence provided above.

The evidence may come from:

1. Web search
2. Ranked web sources
3. Local RAG documents

==================================================
IMPORTANT RULES
==================================================

1. Do not rely on your general knowledge.

2. Do not invent facts, mechanisms, limitations,
   comparisons, or conclusions.

3. A claim appearing in a search answer is NOT
   automatically established fact.

4. Prefer claims supported by multiple independent
   sources.

5. Higher-ranked sources should receive greater
   consideration, but ranking alone does not prove
   correctness.

6. Treat local RAG evidence as direct document
   evidence.

7. If web evidence and RAG evidence disagree,
   explicitly identify the disagreement.

8. If evidence is insufficient, say:
   "The available evidence is insufficient."

9. Do not infer information that is missing.

10. Do not introduce external information.

11. Distinguish between:

    - well-supported findings
    - partially supported findings
    - uncertain claims
    - evidence gaps

12. Be especially careful with limitations.

13. Do not claim that something is a limitation
    unless the provided evidence supports it.

14. Do not repeat the same claim unnecessarily.

15. If the RAG context directly answers the question,
    prioritize the RAG evidence.

16. Do not describe information as coming from the
    RAG document unless the supplied RAG context
    actually supports that claim.

17. Do not treat retrieval scores as evidence that
    a claim is factually correct.

18. Do not treat source ranking as proof of truth.

19. If a web search failed, do not use the failed
    search as supporting evidence.

20. Preserve uncertainty from the evidence.

==================================================
OUTPUT FORMAT
==================================================

KEY FINDINGS:
- ...

SOURCE COMPARISON:
- ...

EVIDENCE STRENGTH:
- ...

UNCERTAINTIES / CONFLICTS:
- ...

EVIDENCE GAPS:
- ...

CONCLUSION:
- ...

Return ONLY the research analysis.
"""

        return self.llm.generate(
            prompt
        )