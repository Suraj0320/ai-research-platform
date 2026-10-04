from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Optional
import tempfile
import os

from app.llm.gateway.gateway import LLMGateway
from app.agents.research_agent import ResearchAgent

from app.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    ResearchResponse,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Research & Analysis Platform",
    description=(
        "AI research platform supporting web research "
        "and document-based RAG."
    ),
    version="0.1.0",
)


# ============================================================
# LAZY-LOADED SERVICES
# ============================================================
#
# Heavy RAG dependencies such as:
#
#   sentence-transformers
#   PyTorch
#   FAISS
#
# are intentionally NOT loaded when FastAPI starts.
#
# RAGPipeline is imported only when a PDF is uploaded.
#
# This helps the Render deployment start within the
# available memory.
# ============================================================

gateway = None
research_agent = None
rag_pipeline = None


# ============================================================
# LLM GATEWAY
# ============================================================

def get_gateway():
    """
    Lazily initialize the LLM Gateway.

    The gateway is created only when an LLM request
    actually needs it.
    """

    global gateway

    if gateway is None:

        print(
            "[INFO] Initializing LLM Gateway..."
        )

        gateway = LLMGateway()

        print(
            "[INFO] LLM Gateway initialized."
        )

    return gateway


# ============================================================
# RESEARCH AGENT
# ============================================================

def get_research_agent():
    """
    Lazily initialize the Research Agent.
    """

    global research_agent

    if research_agent is None:

        print(
            "[INFO] Initializing Research Agent..."
        )

        research_agent = ResearchAgent()

        print(
            "[INFO] Research Agent initialized."
        )

    return research_agent


# ============================================================
# RAG PIPELINE
# ============================================================

def get_rag_pipeline():
    """
    Lazily initialize the RAG Pipeline.

    IMPORTANT:
    RAGPipeline is imported INSIDE this function.

    This prevents sentence-transformers / PyTorch from
    being imported during FastAPI startup.

    The RAG pipeline is loaded only when the user uploads
    a PDF.
    """

    global rag_pipeline

    if rag_pipeline is None:

        print(
            "[INFO] Initializing RAG Pipeline..."
        )

        # IMPORTANT:
        # Do NOT move this import to the top of the file.
        from app.rag.pipeline import RAGPipeline

        rag_pipeline = RAGPipeline()

        print(
            "[INFO] RAG Pipeline initialized."
        )

    return rag_pipeline


# ============================================================
# HEALTH CHECK
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
# DIRECT LLM GENERATION
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

        # ----------------------------------------------------
        # Validate prompt
        # ----------------------------------------------------

        prompt = request.prompt.strip()

        if not prompt:

            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty.",
            )

        # ----------------------------------------------------
        # Lazy-load LLM Gateway
        # ----------------------------------------------------

        llm_gateway = get_gateway()

        # ----------------------------------------------------
        # Generate response
        # ----------------------------------------------------

        response = llm_gateway.generate(
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
# RESEARCH
# ============================================================

@app.post(
    "/research",
    response_model=ResearchResponse,
    tags=["Research"],
)
async def research(

    question: str = Form(...),

    session_id: Optional[str] = Form(
        default=None
    ),

    document: Optional[UploadFile] = File(
        default=None
    ),
):

    document_path = None
    document_name = None

    try:

        # ====================================================
        # VALIDATE QUESTION
        # ====================================================

        question = question.strip()

        if not question:

            raise HTTPException(
                status_code=400,
                detail="Research question cannot be empty.",
            )

        # ====================================================
        # MODE 1 — DOCUMENT / RAG
        # ====================================================

        if document is not None:

            print(
                "[INFO] Document research mode selected."
            )

            # ------------------------------------------------
            # Validate PDF
            # ------------------------------------------------

            filename = document.filename or ""

            if not filename.lower().endswith(".pdf"):

                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are supported.",
                )

            # ------------------------------------------------
            # Read uploaded PDF
            # ------------------------------------------------

            pdf_bytes = await document.read()

            if not pdf_bytes:

                raise HTTPException(
                    status_code=400,
                    detail="Uploaded PDF is empty.",
                )

            # ------------------------------------------------
            # Save temporary PDF
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp_file:

                temp_file.write(
                    pdf_bytes
                )

                document_path = temp_file.name

            document_name = filename

            print(
                f"[INFO] PDF uploaded: "
                f"{document_name}"
            )

            print(
                f"[INFO] Temporary path: "
                f"{document_path}"
            )

            # =================================================
            # LAZY-LOAD RAG PIPELINE
            # =================================================

            pipeline = get_rag_pipeline()

            # =================================================
            # RUN RAG PIPELINE
            # =================================================

            rag_result = pipeline.generate(

                question=question,

                document_path=document_path,

                retrieval_method="hybrid",

                candidate_k=30,

                top_k=20,

                context_k=5,

                rerank=True,
            )

            # =================================================
            # RAG RESPONSE
            # =================================================

            metadata = {

                "mode": "document_rag",

                "document_uploaded": True,

                "document_name": document_name,

                "retrieval_method": rag_result.get(
                    "retrieval_method",
                    "hybrid",
                ),

                "reranking_enabled": rag_result.get(
                    "reranking_enabled",
                    True,
                ),

                "candidate_k": rag_result.get(
                    "candidate_k",
                    30,
                ),

                "top_k": rag_result.get(
                    "top_k",
                    20,
                ),

                "context_k": rag_result.get(
                    "context_k",
                    5,
                ),

                "retrieved_documents": len(
                    rag_result.get(
                        "retrieved_documents",
                        [],
                    )
                ),
            }

            return ResearchResponse(

                success=True,

                agent="rag_pipeline",

                status="completed",

                question=question,

                plan=[
                    "Load uploaded document",
                    "Split document into chunks",
                    "Generate embeddings",
                    "Retrieve relevant document content",
                    "Rerank retrieved content",
                    "Generate grounded answer",
                ],

                search_results=[],

                evaluated_sources=[],

                ranked_sources=[],

                analysis=(
                    "Answer generated using the "
                    "uploaded document through the "
                    "RAG pipeline."
                ),

                final_answer=rag_result.get(
                    "answer",
                    "",
                ),

                errors=[],

                metadata=metadata,
            )

        # ====================================================
        # MODE 2 — WEB RESEARCH
        # ====================================================

        print(
            "[INFO] Web research mode selected."
        )

        # ----------------------------------------------------
        # Lazy-load Research Agent
        # ----------------------------------------------------

        agent = get_research_agent()

        # ----------------------------------------------------
        # Run Research Agent
        # ----------------------------------------------------

        state = agent.run(

            question=question,

            session_id=session_id,
        )

        # ====================================================
        # HANDLE FAILURE
        # ====================================================

        if state.status == "failed":

            error_message = (

                state.errors[-1]

                if state.errors

                else (
                    "Research agent "
                    "execution failed."
                )
            )

            raise HTTPException(
                status_code=500,
                detail=error_message,
            )

        # ====================================================
        # METADATA
        # ====================================================

        metadata = dict(
            state.metadata
            if state.metadata
            else {}
        )

        metadata["mode"] = "web_research"

        metadata["document_uploaded"] = False

        metadata["document_name"] = None

        # ====================================================
        # WEB RESEARCH RESPONSE
        # ====================================================

        return ResearchResponse(

            success=True,

            agent="research_agent",

            status=state.status,

            question=state.question,

            plan=state.plan,

            search_results=state.search_results,

            evaluated_sources=(
                state.evaluated_sources
            ),

            ranked_sources=(
                state.ranked_sources
            ),

            analysis=state.analysis,

            final_answer=state.final_answer,

            errors=state.errors,

            metadata=metadata,
        )

    # ========================================================
    # HTTP ERROR
    # ========================================================

    except HTTPException:

        raise

    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as e:

        print(
            "[ERROR] Research execution failed:"
        )

        print(
            repr(e)
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Research execution failed: "
                f"{str(e)}"
            ),
        )

    # ========================================================
    # CLEAN TEMPORARY PDF
    # ========================================================

    finally:

        if (
            document_path
            and os.path.exists(document_path)
        ):

            try:

                os.remove(
                    document_path
                )

                print(
                    "[INFO] Temporary PDF removed."
                )

            except Exception as e:

                print(
                    "[WARNING] Could not remove "
                    f"temporary PDF: {repr(e)}"
                )


# ============================================================
# ROOT
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
def root():

    return {

        "project":
            "AI Research & Analysis Platform",

        "version":
            "0.1.0",

        "status":
            "running",

        "endpoints": {

            "health":
                "/health",

            "generate":
                "/generate",

            "research":
                "/research",

            "docs":
                "/docs",
        },
    }