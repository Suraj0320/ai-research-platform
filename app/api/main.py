from fastapi import FastAPI, HTTPException

from app.llm.gateway.gateway import LLMGateway
from app.agents.research_agent import ResearchAgent

from app.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    ResearchRequest,
    ResearchResponse,
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="AI Research & Analysis Platform",
    description=(
        "Production-oriented agentic AI research platform "
        "for planning, web search, source evaluation, "
        "ranking, analysis, and answer generation."
    ),
    version="0.1.0",
)


# ============================================================
# Services
# ============================================================

gateway = LLMGateway()

research_agent = ResearchAgent()


# ============================================================
# Health Check
# ============================================================

@app.get(
    "/health",
    tags=["System"],
)
def health_check():

    return {
        "status": "healthy",
        "project": "AI Research & Analysis Platform",
        "version": "0.1.0",
    }


# ============================================================
# Direct LLM Generation
# ============================================================

@app.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["LLM"],
)
def generate(
    request: GenerateRequest,
):

    try:

        prompt = request.prompt.strip()

        if not prompt:

            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty.",
            )

        response = gateway.generate(
            prompt
        )

        return GenerateResponse(
            success=True,
            response=response,
        )

    except HTTPException:

        raise

    except Exception as e:

        print(
            f"[ERROR] LLM generation failed: {repr(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="LLM generation failed.",
        )


# ============================================================
# Research Agent
# ============================================================

@app.post(
    "/research",
    response_model=ResearchResponse,
    tags=["Research"],
)
def research(
    request: ResearchRequest,
):

    try:

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        question = request.question.strip()

        if not question:

            raise HTTPException(
                status_code=400,
                detail="Research question cannot be empty.",
            )

        # ----------------------------------------------------
        # Run Research Agent
        # ----------------------------------------------------

        state = research_agent.run(
            question=question,
            session_id=request.session_id,
        )

        # ----------------------------------------------------
        # Handle agent failure
        # ----------------------------------------------------

        if state.status == "failed":

            error_message = (
                state.errors[-1]
                if state.errors
                else "Research agent execution failed."
            )

            raise HTTPException(
                status_code=500,
                detail=error_message,
            )

        # ----------------------------------------------------
        # Build API response
        # ----------------------------------------------------

        return ResearchResponse(
            success=True,

            agent="research_agent",

            status=state.status,

            question=state.question,

            plan=state.plan,

            search_results=state.search_results,

            evaluated_sources=state.evaluated_sources,

            ranked_sources=state.ranked_sources,

            analysis=state.analysis,

            final_answer=state.final_answer,

            errors=state.errors,

            metadata=state.metadata,
        )

    except HTTPException:

        raise

    except Exception as e:

        print(
            f"[ERROR] Research agent failed: {repr(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Research agent execution failed.",
        )


# ============================================================
# Root Endpoint
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
def root():

    return {
        "project": "AI Research & Analysis Platform",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "research": "/research",
            "docs": "/docs",
        },
    }