from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# DIRECT LLM GENERATION
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Prompt to send to the LLM",
    )


class GenerateResponse(BaseModel):
    success: bool = True

    response: str


# ============================================================
# RESEARCH REQUEST
# ============================================================

class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Research question",
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Optional conversation session ID",
    )


# ============================================================
# CITATION
# ============================================================

class CitationResponse(BaseModel):

    title: str = ""

    url: str = ""


# ============================================================
# SEARCH RESULT DOCUMENT
# ============================================================

class SearchDocumentResponse(BaseModel):

    url: str = ""

    title: str = ""

    content: str = ""

    score: Optional[float] = None

    raw_content: Optional[str] = None

    id: Optional[str] = None


# ============================================================
# SEARCH RESULT
# ============================================================

class SearchResultResponse(BaseModel):

    success: bool = True

    query: str = ""

    answer: str = ""

    citations: List[CitationResponse] = Field(
        default_factory=list
    )

    results: List[SearchDocumentResponse] = Field(
        default_factory=list
    )

    error: Optional[str] = None


# ============================================================
# EVALUATED SOURCE
# ============================================================

class SourceResponse(BaseModel):

    rank: Optional[int] = None

    score: Optional[float] = None

    title: str = ""

    url: str = ""

    query: str = ""

    authority: str = "UNKNOWN"

    relevance: str = "UNKNOWN"

    usefulness: str = "UNKNOWN"

    reason: str = ""


# ============================================================
# RESEARCH RESPONSE
# ============================================================

class ResearchResponse(BaseModel):

    success: bool

    agent: str

    status: str

    question: str

    plan: List[str] = Field(
        default_factory=list
    )

    search_results: List[SearchResultResponse] = Field(
        default_factory=list
    )

    evaluated_sources: List[SourceResponse] = Field(
        default_factory=list
    )

    ranked_sources: List[SourceResponse] = Field(
        default_factory=list
    )

    analysis: Optional[str] = None

    final_answer: Optional[str] = None

    errors: List[str] = Field(
        default_factory=list
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )