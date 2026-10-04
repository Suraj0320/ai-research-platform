import uuid
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Research & Analysis Platform",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION / MEMORY
# ============================================================
#
# IMPORTANT:
# This session_id remains the same across Streamlit reruns.
#
# Question 1 → session_id = ABC
# Question 2 → session_id = ABC
# Question 3 → session_id = ABC
#
# Your FastAPI backend passes this ID to ResearchAgent,
# which uses it to load/save conversation memory.
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# ============================================================
# OPTIONAL CONVERSATION HISTORY
# ============================================================

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
       HERO LABEL
       ====================================================== */

    .eyebrow {
        color: #9ca3af;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }


    /* ======================================================
       HERO TITLE
       ====================================================== */

    .hero-title {
        font-size: 2.9rem;
        font-weight: 750;
        letter-spacing: -1.5px;
        line-height: 1.1;
        margin-bottom: 0.7rem;
        color: #f7f7f7;
    }


    /* ======================================================
       HERO SUBTITLE
       ====================================================== */

    .hero-subtitle {
        color: #9ca3af;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 850px;
        margin-bottom: 3.3rem;
    }


    /* ======================================================
       SECTION TITLE
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
       START RESEARCH BUTTON
       ====================================================== */

    div.stButton > button {
        width: 100%;
        height: 48px;

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
       RESULT CARD
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
       MEMORY STATUS
       ====================================================== */

    .memory-status {
        color: #737985;
        font-size: 0.72rem;
        text-align: center;
        margin-top: 0.8rem;
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

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a research question."
        )

        st.stop()


    # --------------------------------------------------------
    # CURRENT SESSION ID
    # --------------------------------------------------------

    session_id = st.session_state.session_id


    # --------------------------------------------------------
    # SHOW MEMORY STATUS
    # --------------------------------------------------------

    previous_turns = len(
        st.session_state.conversation_history
    )


    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    with st.spinner(
        "Research agent is planning, searching, evaluating, "
        "and synthesizing..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/research",

                json={
                    "question": question.strip(),

                    # IMPORTANT:
                    # This is what connects multiple
                    # questions to the same memory.
                    "session_id": session_id,
                },

                timeout=300,
            )


            # Raise error for HTTP 4xx / 5xx

            response.raise_for_status()


            data = response.json()


        # ----------------------------------------------------
        # CONNECTION ERROR
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI backend. "
                "Make sure Uvicorn is running on port 8000."
            )

            st.stop()


        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        except requests.exceptions.Timeout:

            st.error(
                "The research request timed out."
            )

            st.stop()


        # ----------------------------------------------------
        # HTTP/API ERROR
        # ----------------------------------------------------

        except requests.exceptions.HTTPError as e:

            try:

                error_detail = response.json().get(
                    "detail",
                    str(e)
                )

            except Exception:

                error_detail = str(e)


            st.error(
                f"Research API error: {error_detail}"
            )

            st.stop()


        # ----------------------------------------------------
        # OTHER REQUEST ERROR
        # ----------------------------------------------------

        except requests.exceptions.RequestException as e:

            st.error(
                f"Research API request failed: {e}"
            )

            st.stop()


        # ----------------------------------------------------
        # GENERAL ERROR
        # ----------------------------------------------------

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )

            st.stop()


    # ========================================================
    # SAVE CONVERSATION LOCALLY FOR UI
    # ========================================================

    st.session_state.conversation_history.append(
        {
            "question": question.strip(),
            "answer": data.get(
                "final_answer",
                ""
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
    # FINAL ANSWER
    # ========================================================

    final_answer = data.get(
        "final_answer"
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

        # Use normal Streamlit markdown here so
        # Markdown formatting in the LLM response works.

        st.markdown(final_answer)

        st.markdown(
            """
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # MEMORY DEBUG INFORMATION
    # ========================================================
    #
    # Your ResearchAgent puts memory information into
    # state.metadata.
    #
    # This section is useful while developing.
    # ========================================================

    metadata = data.get(
        "metadata",
        {}
    )


    memory_loaded = metadata.get(
        "memory_loaded"
    )

    memory_count = metadata.get(
        "memory_count"
    )


    if memory_loaded is True:

        st.markdown(
            f"""
            <div class="memory-status">
                🧠 Memory active •
                {memory_count or 0} previous conversation(s) loaded
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif memory_loaded is False:

        st.markdown(
            """
            <div class="memory-status">
                🧠 Memory active •
                No previous conversation found
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # RESEARCH PLAN
    # ========================================================

    plan = data.get(
        "plan",
        []
    )


    if plan:

        with st.expander(
            "🧭 Research Plan",
            expanded=False,
        ):

            for index, step in enumerate(
                plan,
                start=1
            ):

                st.markdown(
                    f"**{index}.** {step}"
                )


    # ========================================================
    # RANKED SOURCES
    # ========================================================

    ranked_sources = data.get(
        "ranked_sources",
        []
    )


    if ranked_sources:

        with st.expander(
            "📚 Ranked Sources",
            expanded=False,
        ):

            for source in ranked_sources:

                rank = source.get(
                    "rank",
                    "-"
                )

                title = source.get(
                    "title",
                    "Untitled source"
                )

                url = source.get(
                    "url",
                    ""
                )

                score = source.get(
                    "score",
                    "N/A"
                )

                authority = source.get(
                    "authority",
                    "UNKNOWN"
                )

                relevance = source.get(
                    "relevance",
                    "UNKNOWN"
                )

                usefulness = source.get(
                    "usefulness",
                    "UNKNOWN"
                )

                reason = source.get(
                    "reason",
                    ""
                )


                st.markdown(
                    f"### #{rank} — {title}"
                )


                col1, col2, col3, col4 = st.columns(4)


                with col1:

                    st.metric(
                        "Score",
                        score
                    )


                with col2:

                    st.metric(
                        "Authority",
                        authority
                    )


                with col3:

                    st.metric(
                        "Relevance",
                        relevance
                    )


                with col4:

                    st.metric(
                        "Usefulness",
                        usefulness
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
        "analysis"
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
        []
    )


    if search_results:

        with st.expander(
            "🔎 Search Evidence",
            expanded=False,
        ):

            for index, result in enumerate(
                search_results,
                start=1
            ):

                st.markdown(
                    f"### Search Result {index}"
                )


                if result.get(
                    "success",
                    True
                ):

                    query_text = result.get(
                        "query",
                        ""
                    )

                    answer_text = result.get(
                        "answer",
                        ""
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
                        []
                    )


                    if citations:

                        st.markdown(
                            "**Citations**"
                        )


                        for citation in citations:

                            title = citation.get(
                                "title",
                                "Source"
                            )

                            url = citation.get(
                                "url",
                                ""
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
                            "Search failed"
                        )
                    )


# ============================================================
# NEW CONVERSATION
# ============================================================
#
# This creates a completely new memory session.
#
# Existing conversation remains stored in the backend,
# but the new session will not use it.
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

    # Generate completely new memory ID

    st.session_state.session_id = str(
        uuid.uuid4()
    )


    # Clear UI conversation history

    st.session_state.conversation_history = []


    # Clear previous question/result state

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