from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResearchState:

    # ============================================================
    # REQUEST
    # ============================================================

    question: str

    session_id: Optional[str] = None

    # ============================================================
    # MEMORY
    # ============================================================

    memory: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # ============================================================
    # RESEARCH PIPELINE
    # ============================================================

    plan: List[str] = field(
        default_factory=list
    )

    search_results: List[Dict[str, Any]] = field(
        default_factory=list
    )

    evaluated_sources: List[Dict[str, Any]] = field(
        default_factory=list
    )

    ranked_sources: List[Dict[str, Any]] = field(
        default_factory=list
    )

    analysis: Optional[str] = None

    final_answer: Optional[str] = None

    # ============================================================
    # EXECUTION
    # ============================================================

    status: str = "initialized"

    errors: List[str] = field(
        default_factory=list
    )

    # ============================================================
    # METADATA
    # ============================================================

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )