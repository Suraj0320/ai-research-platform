from app.agents.state import ResearchState

from app.agents.planner import ResearchPlanner
from app.agents.analyzer import ResearchAnalyzer
from app.agents.answer_generator import ResearchAnswerGenerator

from app.agents.source_evaluator import SourceEvaluator
from app.agents.source_ranker import SourceRanker

from app.tools.web_search import WebSearchTool

from app.memory.memory_manager import MemoryManager

from app.agents.intent_router import IntentRouter
from app.agents.memory_agent import MemoryAgent


class ResearchAgent:

    # ============================================================
    # CONFIGURATION
    # ============================================================

    MAX_SEARCH_STEPS = 3

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self):

        # --------------------------------------------------------
        # Research components
        # --------------------------------------------------------

        self.planner = ResearchPlanner()

        self.search_tool = WebSearchTool()

        self.source_evaluator = SourceEvaluator()

        self.source_ranker = SourceRanker()

        self.analyzer = ResearchAnalyzer()

        self.answer_generator = ResearchAnswerGenerator()

        # --------------------------------------------------------
        # Memory components
        # --------------------------------------------------------

        self.memory_manager = MemoryManager()

        self.memory_agent = MemoryAgent()

    # ============================================================
    # MAIN RESEARCH PIPELINE
    # ============================================================

    def run(
        self,
        question: str,
        session_id: str | None = None,
    ) -> ResearchState:

        # ========================================================
        # CREATE STATE
        # ========================================================

        state = ResearchState(
            question=question,
            session_id=session_id,
        )

        try:

            # ====================================================
            # 1. LOAD MEMORY
            # ====================================================

            state.status = "loading_memory"

            memory_context = ""

            # ----------------------------------------------------
            # Initialize memory defaults
            # ----------------------------------------------------

            state.memory = []

            state.metadata["memory_context"] = ""

            state.metadata["memory_loaded"] = False

            state.metadata["memory_count"] = 0

            # ----------------------------------------------------
            # Load memory only when session exists
            # ----------------------------------------------------

            if session_id:

                print(
                    f"[MEMORY] Loading memory for session: "
                    f"{session_id}"
                )

                # ------------------------------------------------
                # Load previous conversations
                # ------------------------------------------------

                state.memory = (
                    self.memory_manager.load_memory(
                        session_id=session_id
                    )
                    or []
                )

                # ------------------------------------------------
                # Format memory for agents
                # ------------------------------------------------

                memory_context = (
                    self.memory_manager.format_memory(
                        state.memory
                    )
                    or ""
                )

                # ------------------------------------------------
                # Store memory information in state
                # ------------------------------------------------

                state.metadata["memory_context"] = (
                    memory_context
                )

                state.metadata["memory_loaded"] = (
                    bool(state.memory)
                )

                state.metadata["memory_count"] = (
                    len(state.memory)
                )

            # ----------------------------------------------------
            # Debug information
            # ----------------------------------------------------

            print(
                f"[MEMORY] Session ID: {session_id}"
            )

            print(
                f"[MEMORY] Previous conversations: "
                f"{len(state.memory)}"
            )

            print(
                f"[MEMORY] Context available: "
                f"{bool(memory_context)}"
            )

            # ====================================================
            # 2. INTENT ROUTING
            # ====================================================

            state.status = "routing_intent"

            is_memory_query = (
                IntentRouter.is_memory_query(
                    question
                )
            )

            state.metadata["memory_query"] = (
                is_memory_query
            )

            # ----------------------------------------------------
            # Memory query
            # ----------------------------------------------------
            #
            # Examples:
            #
            # "What was my previous question?"
            # "What did I ask you before?"
            # "What did we discuss earlier?"
            #
            # These questions should NOT go through:
            #
            # Planner -> Search -> Evaluator -> Ranker -> Analyzer
            #
            # They should be answered directly from memory.
            # ----------------------------------------------------

            if is_memory_query:

                state.status = "answering_from_memory"

                state.metadata["intent"] = "memory"

                print(
                    "[ROUTER] Memory-related question detected."
                )

                print(
                    "[MEMORY] Answering directly from conversation memory."
                )

                # ------------------------------------------------
                # If there is no memory
                # ------------------------------------------------

                if not memory_context:

                    state.final_answer = (
                        "I don't have any previous "
                        "conversation available in this session."
                    )

                else:

                    # --------------------------------------------
                    # Ask MemoryAgent
                    # --------------------------------------------

                    state.final_answer = (
                        self.memory_agent.answer(
                            question=question,
                            memory_context=memory_context,
                        )
                    )

                # ------------------------------------------------
                # Metadata
                # ------------------------------------------------

                state.metadata["answer_available"] = (
                    bool(state.final_answer)
                )

                state.metadata["pipeline_completed"] = True

                state.metadata["pipeline_type"] = (
                    "memory_query"
                )

                # ------------------------------------------------
                # Completed
                # ------------------------------------------------

                state.status = "completed"

                print(
                    "[MEMORY] Memory answer generated successfully."
                )

                return state

            # ====================================================
            # NORMAL RESEARCH QUERY
            # ====================================================

            state.metadata["intent"] = "research"

            state.metadata["pipeline_type"] = (
                "research_query"
            )

            print(
                "[ROUTER] Research question detected."
            )

            # ====================================================
            # 3. PLANNING
            # ====================================================

            state.status = "planning"

            print(
                "[PLANNER] Creating research plan..."
            )

            state.plan = (
                self.planner.create_plan(
                    state
                )
            )

            # ----------------------------------------------------
            # Validate plan
            # ----------------------------------------------------

            if not state.plan:

                raise RuntimeError(
                    "Research planner returned an empty plan."
                )

            state.metadata["plan_count"] = (
                len(state.plan)
            )

            print(
                f"[PLANNER] Generated "
                f"{len(state.plan)} research objectives."
            )

            # ====================================================
            # 4. SEARCHING
            # ====================================================

            state.status = "searching"

            print(
                "[SEARCH] Starting web research..."
            )

            state.search_results = (
                self.search(state)
            )

            state.metadata["search_steps"] = (
                len(state.search_results)
            )

            print(
                f"[SEARCH] Completed "
                f"{len(state.search_results)} search steps."
            )

            # ----------------------------------------------------
            # Check whether search returned anything
            # ----------------------------------------------------

            if not state.search_results:

                print(
                    "[SEARCH] No search results were returned."
                )

            # ====================================================
            # 5. SOURCE EVALUATION
            # ====================================================

            state.status = "evaluating_sources"

            print(
                "[EVALUATOR] Evaluating sources..."
            )

            state.evaluated_sources = (
                self.source_evaluator.evaluate(
                    state.search_results
                )
                or []
            )

            state.metadata["evaluated_source_count"] = (
                len(state.evaluated_sources)
            )

            print(
                f"[EVALUATOR] Evaluated "
                f"{len(state.evaluated_sources)} sources."
            )

            # ====================================================
            # 6. SOURCE RANKING
            # ====================================================

            state.status = "ranking_sources"

            print(
                "[RANKER] Ranking sources..."
            )

            state.ranked_sources = (
                self.source_ranker.rank(
                    state.evaluated_sources
                )
                or []
            )

            state.metadata["ranked_source_count"] = (
                len(state.ranked_sources)
            )

            print(
                f"[RANKER] Ranked "
                f"{len(state.ranked_sources)} sources."
            )

            # ====================================================
            # 7. ANALYSIS
            # ====================================================

            state.status = "analyzing"

            print(
                "[ANALYZER] Analyzing research evidence..."
            )

            state.analysis = (
                self.analyzer.analyze(
                    state
                )
            )

            state.metadata["analysis_available"] = (
                bool(state.analysis)
            )

            print(
                "[ANALYZER] Analysis generated: "
                f"{bool(state.analysis)}"
            )

            # ====================================================
            # 8. FINAL ANSWER
            # ====================================================

            state.status = "generating"

            print(
                "[ANSWER] Generating final answer..."
            )

            state.final_answer = (
                self.answer_generator.generate(
                    state
                )
            )

            state.metadata["answer_available"] = (
                bool(state.final_answer)
            )

            print(
                "[ANSWER] Final answer generated: "
                f"{bool(state.final_answer)}"
            )

            # ====================================================
            # 9. SAVE MEMORY
            # ====================================================

            if (
                session_id
                and state.final_answer
            ):

                print(
                    "[MEMORY] Saving conversation..."
                )

                self.memory_manager.save_memory(
                    session_id=session_id,
                    question=question,
                    answer=state.final_answer,
                )

                print(
                    "[MEMORY] Conversation saved."
                )

            # ====================================================
            # 10. COMPLETED
            # ====================================================

            state.status = "completed"

            state.metadata["pipeline_completed"] = True

            print(
                "[RESEARCH] Pipeline completed successfully."
            )

        except Exception as e:

            # ====================================================
            # PIPELINE FAILURE
            # ====================================================

            state.status = "failed"

            state.metadata["pipeline_completed"] = False

            error_message = str(e)

            state.errors.append(
                error_message
            )

            print(
                "[RESEARCH] Pipeline failed: "
                f"{repr(e)}"
            )

        return state

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        state: ResearchState,
    ):

        results = []

        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------

        if not state.plan:

            print(
                "[SEARCH] No research objectives available."
            )

            return results

        # --------------------------------------------------------
        # Limit search steps
        # --------------------------------------------------------

        search_steps = state.plan[
            :self.MAX_SEARCH_STEPS
        ]

        # --------------------------------------------------------
        # Execute search for each objective
        # --------------------------------------------------------

        for index, step in enumerate(
            search_steps,
            start=1,
        ):

            # ----------------------------------------------------
            # Build search query
            # ----------------------------------------------------

            query = f"""
Research question:
{state.question}

Research objective:
{step}

Find relevant, reliable information for this research objective.

Prefer authoritative and recent sources when appropriate.
""".strip()

            print(
                f"\n[SEARCH] Step {index}/"
                f"{len(search_steps)}"
            )

            print(
                f"[SEARCH] Objective: {step}"
            )

            try:

                # ------------------------------------------------
                # Execute search
                # ------------------------------------------------

                result = self.search_tool.search(
                    query
                )

                # ------------------------------------------------
                # Store result
                # ------------------------------------------------

                if result is not None:

                    results.append(
                        result
                    )

                    print(
                        f"[SEARCH] Step {index} completed."
                    )

                else:

                    print(
                        f"[SEARCH] Step {index} "
                        f"returned no result."
                    )

            except Exception as e:

                print(
                    f"[SEARCH] Step {index} failed: "
                    f"{repr(e)}"
                )

                # ------------------------------------------------
                # Continue with remaining objectives
                # ------------------------------------------------

                continue

        # --------------------------------------------------------
        # Return search results
        # --------------------------------------------------------

        return results