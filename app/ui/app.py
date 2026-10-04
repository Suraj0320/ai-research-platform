import uuid
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

st.set_page_config(
    page_title="AI Research & Analysis Platform",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION / MEMORY
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background: #0d0f12;
        color: #f5f5f5;
    }

    .main {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* ======================================================
       MAIN CONTAINER
       ====================================================== */

    .block-container {
        max-width: 1080px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        margin: auto;
    }


    /* ======================================================
       HERO
       ====================================================== */

    .eyebrow {
        color: #9ca3af;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.9rem;
        font-weight: 750;
        letter-spacing: -1.5px;
        line-height: 1.1;
        margin-bottom: 0.7rem;
        color: #f7f7f7;
    }

    .hero-subtitle {
        color: #9ca3af;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 850px;
        margin-bottom: 3.3rem;
    }


    /* ======================================================
       SECTION
       ====================================================== */

    .section-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f5f5f5;
        margin-bottom: 0.7rem;
    }

    .section-description {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }


    /* ======================================================
       TEXTAREA
       ====================================================== */

    textarea {
        background-color: #25272f !important;
        color: #eeeeee !important;
        border: 1px solid #363943 !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }

    textarea::placeholder {
        color: #9ca0aa !important;
    }

    textarea:focus {
        border: 1px solid #555963 !important;
        box-shadow: none !important;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    div[data-testid="stFileUploader"] {
        background: #17191e;
        border: 1px solid #30333a;
        border-radius: 12px;
        padding: 0.5rem;
    }


    /* ======================================================
       START RESEARCH BUTTON
       ====================================================== */

    div.stButton > button {
        width: 100%;
        min-height: 48px;

        background: #ff4f52;
        color: white;

        border: none;
        border-radius: 9px;

        font-size: 0.95rem;
        font-weight: 600;

        margin-top: 0.25rem;

        transition:
            background 0.15s ease,
            transform 0.1s ease;
    }

    div.stButton > button:hover {
        background: #ff4145;
        color: white;
        border: none;
    }

    div.stButton > button:active {
        transform: scale(0.995);
    }


    /* ======================================================
       DIVIDER
       ====================================================== */

    .custom-divider {
        height: 1px;
        background: #30333a;
        margin-top: 3.1rem;
        margin-bottom: 2.5rem;
    }


    /* ======================================================
       ANSWER
       ====================================================== */

    .answer-card {
        background: #17191e;
        border: 1px solid #30333a;
        border-radius: 12px;
        padding: 1.4rem;
        margin-top: 1rem;
        line-height: 1.7;
    }

    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }


    /* ======================================================
       DOCUMENT STATUS
       ====================================================== */

    .document-status {
        background: #17191e;
        border: 1px solid #30333a;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        color: #c7cad1;
        font-size: 0.85rem;
    }


    /* ======================================================
       MEMORY STATUS
       ====================================================== */

    .memory-status {
        color: #9ca3af;
        font-size: 0.78rem;
        text-align: center;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
    }


    /* ======================================================
       MEMORY DETAILS CARD
       ====================================================== */

    .memory-card {
        background: #17191e;
        border: 1px solid #30333a;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.5rem;
    }

    .memory-label {
        color: #8f96a3;
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
    }

    .memory-value {
        color: #eeeeee;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .custom-footer {
        text-align: center;
        color: #6f7480;
        font-size: 0.78rem;
        margin-top: 1rem;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    div[data-testid="stExpander"] {
        background: #111317;
        border: 1px solid #30333a;
        border-radius: 8px;
    }


    /* ======================================================
       METRICS
       ====================================================== */

    div[data-testid="stMetric"] {
        background: #17191e;
        border-radius: 8px;
        padding: 0.8rem;
    }


    /* ======================================================
       GENERAL SPACING
       ====================================================== */

    [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="eyebrow">
        AI AGENT • RESEARCH SYSTEM
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero-title">
        🔎 AI Research & Analysis Platform
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero-subtitle">
        Ask a research question and let the agent plan the research,
        search for evidence, evaluate sources, rank them, analyze the
        findings, and generate a grounded answer.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RESEARCH QUESTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Research Question
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="section-description">
        Ask a question that requires research and evidence.
    </div>
    """,
    unsafe_allow_html=True,
)


question = st.text_area(
    label="Research Question",
    placeholder=(
        "Example: What is Retrieval-Augmented Generation, "
        "how does it work, and what are its major limitations?"
    ),
    height=135,
    label_visibility="collapsed",
)


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Document
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="section-description">
        Upload a PDF if you want the agent to answer using your
        document. Leave it empty for web-based research.
    </div>
    """,
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader(
    label="Upload PDF",
    type=["pdf"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)


# ============================================================
# DOCUMENT STATUS
# ============================================================

if uploaded_file is not None:

    file_size_mb = uploaded_file.size / (1024 * 1024)

    st.markdown(
        f"""
        <div class="document-status">
            📄 <b>{uploaded_file.name}</b>
            &nbsp;•&nbsp;
            {file_size_mb:.2f} MB
            &nbsp;•&nbsp;
            PDF ready for RAG
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# START RESEARCH
# ============================================================

start_research = st.button(
    "🔍  Start Research",
    use_container_width=True,
    type="primary",
)


# ============================================================
# RESEARCH EXECUTION
# ============================================================

if start_research:

    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    if not question.strip():

        st.warning(
            "Please enter a research question."
        )

        st.stop()


    # ========================================================
    # SESSION
    # ========================================================

    session_id = st.session_state.session_id


    # ========================================================
    # PREVIOUS CONVERSATION COUNT
    # ========================================================

    previous_turns = len(
        st.session_state.conversation_history
    )


    # ========================================================
    # PREPARE REQUEST
    # ========================================================

    form_data = {
        "question": question.strip(),
        "session_id": session_id,
    }

    files = None


    # ========================================================
    # PDF PRESENT
    # ========================================================

    if uploaded_file is not None:

        files = {
            "document": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        }


    # ========================================================
    # API REQUEST
    # ========================================================

    with st.spinner(
        (
            "Processing document, creating embeddings, "
            "retrieving relevant content, reranking, "
            "and generating answer..."
            if uploaded_file is not None
            else
            "Research agent is planning, searching, "
            "evaluating, and synthesizing..."
        )
    ):

        try:

            response = requests.post(
                f"{API_URL}/research",
                data=form_data,
                files=files,
                timeout=1000,
            )

            response.raise_for_status()

            data = response.json()


        # ====================================================
        # CONNECTION ERROR
        # ====================================================

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI backend. "
                "Make sure Uvicorn is running on port 8000."
            )

            st.stop()


        # ====================================================
        # TIMEOUT
        # ====================================================

        except requests.exceptions.Timeout:

            st.error(
                "The research request timed out."
            )

            st.stop()


        # ====================================================
        # HTTP ERROR
        # ====================================================

        except requests.exceptions.HTTPError as e:

            try:

                error_detail = response.json().get(
                    "detail",
                    str(e),
                )

            except Exception:

                error_detail = str(e)

            st.error(
                f"Research API error: {error_detail}"
            )

            st.stop()


        # ====================================================
        # OTHER REQUEST ERROR
        # ====================================================

        except requests.exceptions.RequestException as e:

            st.error(
                f"Research API request failed: {e}"
            )

            st.stop()


        # ====================================================
        # JSON ERROR
        # ====================================================

        except ValueError:

            st.error(
                "The backend returned an invalid response."
            )

            st.stop()


        # ====================================================
        # GENERAL ERROR
        # ====================================================

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )

            st.stop()


    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    st.session_state.conversation_history.append(
        {
            "question": question.strip(),

            "answer": data.get(
                "final_answer",
                "",
            ),

            "document": (
                uploaded_file.name
                if uploaded_file is not None
                else None
            ),
        }
    )


    # ========================================================
    # DIVIDER
    # ========================================================

    st.markdown(
        """
        <div class="custom-divider"></div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # STATUS
    # ========================================================

    if data.get("status") == "completed":

        st.success(
            "Research completed successfully."
        )

    else:

        st.warning(
            f"Research status: "
            f"{data.get('status', 'unknown')}"
        )


    # ========================================================
    # METADATA
    # ========================================================

    metadata = data.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        metadata = {}


    # ========================================================
    # GENERAL METADATA
    # ========================================================

    mode = metadata.get(
        "mode",
        "web_research",
    )

    document_uploaded = metadata.get(
        "document_uploaded",
        uploaded_file is not None,
    )

    document_name = metadata.get(
        "document_name",
        (
            uploaded_file.name
            if uploaded_file is not None
            else None
        ),
    )

    retrieval_method = metadata.get(
        "retrieval_method",
    )

    reranking_enabled = metadata.get(
        "reranking_enabled",
    )

    retrieved_documents = metadata.get(
        "retrieved_documents",
    )

    candidate_k = metadata.get(
        "candidate_k",
    )

    top_k = metadata.get(
        "top_k",
    )

    context_k = metadata.get(
        "context_k",
    )


    # ========================================================
    # MEMORY INFORMATION
    # ========================================================

    memory_loaded = metadata.get(
        "memory_loaded"
    )

    memory_count = metadata.get(
        "memory_count",
        0,
    )

    memory_context = metadata.get(
        "memory_context",
        "",
    )

    intent = metadata.get(
        "intent",
        "",
    )


    # ========================================================
    # MEMORY STATUS
    # ========================================================

    if memory_loaded is True:

        st.success(
            f"🧠 Conversation memory active • "
            f"{memory_count} previous conversation(s) loaded."
        )

    elif memory_loaded is False:

        st.info(
            "🧠 Conversation memory active • "
            "No previous conversation found."
        )


    # ========================================================
    # MEMORY DETAILS
    # ========================================================

    if memory_loaded is not None:

        with st.expander(
            "🧠 Memory Details",
            expanded=False,
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "**Memory Status**"
                )

                if memory_loaded:

                    st.success(
                        "Active"
                    )

                else:

                    st.info(
                        "No previous memory"
                    )


                st.markdown(
                    "**Previous Conversations**"
                )

                st.write(
                    memory_count
                )


            with col2:

                st.markdown(
                    "**Session ID**"
                )

                st.code(
                    session_id,
                    language=None,
                )


                if intent:

                    st.markdown(
                        "**Detected Intent**"
                    )

                    st.write(
                        intent
                    )


            # ------------------------------------------------
            # Memory context preview
            # ------------------------------------------------

            if memory_context:

                st.markdown(
                    "---"
                )

                st.markdown(
                    "**Memory Context Used**"
                )

                st.text(
                    memory_context
                )

            else:

                st.markdown(
                    "---"
                )

                st.caption(
                    "No previous conversation context was available."
                )


    # ========================================================
    # DOCUMENT RAG STATUS
    # ========================================================

    if document_uploaded:

        retrieval_text = (
            retrieval_method
            if retrieval_method
            else "N/A"
        )

        if reranking_enabled is True:

            reranking_text = "Enabled"

        elif reranking_enabled is False:

            reranking_text = "Disabled"

        else:

            reranking_text = "N/A"


        retrieved_text = (
            str(retrieved_documents)
            if retrieved_documents is not None
            else "N/A"
        )


        st.markdown(
            f"""
            <div class="document-status">
                📄 <b>Document RAG Active</b>
                &nbsp;•&nbsp;
                {document_name or "Uploaded PDF"}
                &nbsp;•&nbsp;
                Retrieval: {retrieval_text}
                &nbsp;•&nbsp;
                Retrieved: {retrieved_text}
                &nbsp;•&nbsp;
                Reranking: {reranking_text}
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    final_answer = data.get(
        "final_answer",
    )


    if final_answer:

        st.markdown(
            """
            <div class="result-title">
                📊 Research Answer
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.markdown(
            """
            <div class="answer-card">
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # Render LLM Markdown normally
        # ----------------------------------------------------

        st.markdown(
            final_answer
        )


        st.markdown(
            """
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.warning(
            "The research agent did not return a final answer."
        )


    # ========================================================
    # DOCUMENT RAG DETAILS
    # ========================================================

    if document_uploaded:

        with st.expander(
            "📄 Document RAG Details",
            expanded=False,
        ):

            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Retrieval Method",
                    retrieval_method or "N/A",
                )

                st.metric(
                    "Candidate K",
                    (
                        candidate_k
                        if candidate_k is not None
                        else "N/A"
                    ),
                )

                st.metric(
                    "Top K",
                    (
                        top_k
                        if top_k is not None
                        else "N/A"
                    ),
                )


            with col2:

                st.metric(
                    "Reranking",
                    (
                        "Enabled"
                        if reranking_enabled is True
                        else "Disabled"
                        if reranking_enabled is False
                        else "N/A"
                    ),
                )

                st.metric(
                    "Retrieved Documents",
                    (
                        retrieved_documents
                        if retrieved_documents is not None
                        else "N/A"
                    ),
                )

                st.metric(
                    "Context K",
                    (
                        context_k
                        if context_k is not None
                        else "N/A"
                    ),
                )


            st.markdown(
                f"**Document:** "
                f"{document_name or 'N/A'}"
            )


            st.markdown(
                "**Pipeline:** "
                "Load → Chunk → Embed → Retrieve → "
                "Rerank → Generate"
            )


    # ========================================================
    # RESEARCH PLAN
    # ========================================================

    plan = data.get(
        "plan",
        [],
    )


    if plan:

        with st.expander(
            "🧭 Research Plan",
            expanded=False,
        ):

            for index, step in enumerate(
                plan,
                start=1,
            ):

                st.markdown(
                    f"**{index}.** {step}"
                )


    # ========================================================
    # RANKED SOURCES
    # ========================================================

    ranked_sources = data.get(
        "ranked_sources",
        [],
    )


    if ranked_sources:

        with st.expander(
            "📚 Ranked Sources",
            expanded=False,
        ):

            for source in ranked_sources:

                rank = source.get(
                    "rank",
                    "-",
                )

                title = source.get(
                    "title",
                    "Untitled source",
                )

                url = source.get(
                    "url",
                    "",
                )

                score = source.get(
                    "score",
                    "N/A",
                )

                authority = source.get(
                    "authority",
                    "UNKNOWN",
                )

                relevance = source.get(
                    "relevance",
                    "UNKNOWN",
                )

                usefulness = source.get(
                    "usefulness",
                    "UNKNOWN",
                )

                reason = source.get(
                    "reason",
                    "",
                )


                st.markdown(
                    f"### #{rank} — {title}"
                )


                col1, col2, col3, col4 = st.columns(4)


                with col1:

                    st.metric(
                        "Score",
                        score,
                    )


                with col2:

                    st.metric(
                        "Authority",
                        authority,
                    )


                with col3:

                    st.metric(
                        "Relevance",
                        relevance,
                    )


                with col4:

                    st.metric(
                        "Usefulness",
                        usefulness,
                    )


                if reason:

                    st.markdown(
                        f"**Reason:** {reason}"
                    )


                if url:

                    st.markdown(
                        f"[🔗 Open source]({url})"
                    )


                st.divider()


    # ========================================================
    # RESEARCH ANALYSIS
    # ========================================================

    analysis = data.get(
        "analysis",
    )


    if analysis:

        with st.expander(
            "🧠 Research Analysis",
            expanded=False,
        ):

            st.markdown(
                analysis
            )


    # ========================================================
    # SEARCH EVIDENCE
    # ========================================================

    search_results = data.get(
        "search_results",
        [],
    )


    if search_results:

        with st.expander(
            "🔎 Search Evidence",
            expanded=False,
        ):

            for index, result in enumerate(
                search_results,
                start=1,
            ):

                st.markdown(
                    f"### Search Result {index}"
                )


                if result.get(
                    "success",
                    True,
                ):

                    query_text = result.get(
                        "query",
                        "",
                    )

                    answer_text = result.get(
                        "answer",
                        "",
                    )


                    if query_text:

                        st.markdown(
                            "**Research Query**"
                        )

                        st.code(
                            query_text
                        )


                    if answer_text:

                        st.markdown(
                            "**Evidence Summary**"
                        )

                        st.write(
                            answer_text
                        )


                    citations = result.get(
                        "citations",
                        [],
                    )


                    if citations:

                        st.markdown(
                            "**Citations**"
                        )


                        for citation in citations:

                            title = citation.get(
                                "title",
                                "Source",
                            )

                            url = citation.get(
                                "url",
                                "",
                            )


                            if url:

                                st.markdown(
                                    f"- [{title}]({url})"
                                )

                            else:

                                st.write(
                                    f"- {title}"
                                )

                else:

                    st.error(
                        result.get(
                            "error",
                            "Search failed",
                        )
                    )


# ============================================================
# NEW CONVERSATION
# ============================================================

st.markdown(
    """
    <div class="custom-divider"></div>
    """,
    unsafe_allow_html=True,
)


new_conversation = st.button(
    "↻  New Conversation"
)


if new_conversation:

    # --------------------------------------------------------
    # Generate a new memory session
    # --------------------------------------------------------

    st.session_state.session_id = str(
        uuid.uuid4()
    )


    # --------------------------------------------------------
    # Clear UI conversation history
    # --------------------------------------------------------

    st.session_state.conversation_history = []


    # --------------------------------------------------------
    # Restart Streamlit
    # --------------------------------------------------------

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">
        AI Research & Analysis Platform •
        Plan → Search → Evaluate → Analyze → Answer
    </div>
    """,
    unsafe_allow_html=True,
)