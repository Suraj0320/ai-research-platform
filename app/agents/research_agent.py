from app.agents.state import ResearchState

from app.agents.planner import ResearchPlanner
from app.agents.analyzer import ResearchAnalyzer
from app.agents.answer_generator import ResearchAnswerGenerator

from app.agents.source_evaluator import SourceEvaluator
from app.agents.source_ranker import SourceRanker

from app.tools.web_search import WebSearchTool

from app.memory.memory_manager import MemoryManager


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
        # Memory
        # --------------------------------------------------------

        self.memory_manager = MemoryManager()

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

            if session_id:

                # ------------------------------------------------
                # Load previous conversation
                # ------------------------------------------------

                state.memory = (
                    self.memory_manager.load_memory(
                        session_id=session_id
                    )
                )

                # ------------------------------------------------
                # Format memory for downstream agents
                # ------------------------------------------------

                memory_context = (
                    self.memory_manager.format_memory(
                        state.memory
                    )
                )

                # ------------------------------------------------
                # Store memory context in state metadata
                #
                # Planner reads:
                #
                # state.metadata["memory_context"]
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

            else:

                # ------------------------------------------------
                # No session -> no memory
                # ------------------------------------------------

                state.memory = []

                state.metadata["memory_context"] = ""

                state.metadata["memory_loaded"] = False

                state.metadata["memory_count"] = 0

            # ----------------------------------------------------
            # Debug
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
            # 2. PLANNING
            # ====================================================

            state.status = "planning"

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
            # 3. SEARCHING
            # ====================================================

            state.status = "searching"

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

            # ====================================================
            # 4. SOURCE EVALUATION
            # ====================================================

            state.status = "evaluating_sources"

            state.evaluated_sources = (
                self.source_evaluator.evaluate(
                    state.search_results
                )
            )

            state.metadata["evaluated_source_count"] = (
                len(state.evaluated_sources)
            )

            print(
                f"[EVALUATOR] Evaluated "
                f"{len(state.evaluated_sources)} sources."
            )

            # ====================================================
            # 5. SOURCE RANKING
            # ====================================================

            state.status = "ranking_sources"

            state.ranked_sources = (
                self.source_ranker.rank(
                    state.evaluated_sources
                )
            )

            state.metadata["ranked_source_count"] = (
                len(state.ranked_sources)
            )

            print(
                f"[RANKER] Ranked "
                f"{len(state.ranked_sources)} sources."
            )

            # ====================================================
            # 6. ANALYSIS
            # ====================================================

            state.status = "analyzing"

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
            # 7. FINAL ANSWER
            # ====================================================

            state.status = "generating"

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
            # 8. SAVE MEMORY
            # ====================================================

            if (
                session_id
                and state.final_answer
            ):

                self.memory_manager.save_memory(
                    session_id=session_id,
                    question=question,
                    answer=state.final_answer,
                )

                print(
                    "[MEMORY] Conversation saved."
                )

            # ====================================================
            # 9. COMPLETED
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
                f"[RESEARCH] Pipeline failed: "
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

                # Continue with remaining objectives
                # instead of killing the entire research pipeline.

                continue

        # --------------------------------------------------------
        # Return search results
        # --------------------------------------------------------

        return results