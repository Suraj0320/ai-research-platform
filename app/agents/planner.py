from app.llm.gateway.gateway import LLMGateway
from app.agents.state import ResearchState


class ResearchPlanner:

    def __init__(self):
        self.llm = LLMGateway()

    def create_plan(
        self,
        state: ResearchState,
    ) -> list[str]:

        # ==================================================
        # CURRENT QUESTION
        # ==================================================

        question = state.question

        # ==================================================
        # MEMORY
        # ==================================================

        memory_context = ""

        if hasattr(state, "metadata"):

            metadata = state.metadata

            if isinstance(metadata, dict):

                memory_context = metadata.get(
                    "memory_context",
                    "",
                )

        # ==================================================
        # MEMORY SECTION
        # ==================================================

        if memory_context:

            memory_section = f"""
==================================================
PREVIOUS CONVERSATION MEMORY
==================================================

{memory_context}

Use this previous conversation only when it
helps resolve references in the current question,
such as:

- "it"
- "this"
- "that"
- "they"
- "the above"
- "previously discussed topic"

If the current question depends on the previous
conversation, use that context when creating the
research objectives.

Do not blindly assume that previous information
is relevant.

The current question always has priority over
previous conversation context.
"""

        else:

            memory_section = """
==================================================
PREVIOUS CONVERSATION MEMORY
==================================================

No previous conversation is available.
"""

        # ==================================================
        # PLANNING PROMPT
        # ==================================================

        prompt = f"""
You are a research planning agent.

==================================================
CURRENT RESEARCH QUESTION
==================================================

{question}

{memory_section}

==================================================
TASK
==================================================

Create a research plan containing exactly
3 to 5 independent research objectives.

The objectives must collectively answer the
CURRENT research question.

If the current question refers to something
from the previous conversation, resolve that
reference using the previous conversation.

Example:

Previous question:
"What is Retrieval-Augmented Generation?"

Current question:
"Why is it useful?"

Interpret "it" as Retrieval-Augmented Generation.

==================================================
RULES
==================================================

1. Focus only on what needs to be investigated.

2. Do not answer the question.

3. Do not write explanations.

4. Do not include a final-answer step.

5. Avoid duplicate objectives.

6. Each objective must be independently searchable.

7. Cover different aspects of the current question.

8. Preserve the exact intent of the current question.

9. Use previous conversation only when relevant.

10. Do not change the topic based on memory.

11. Prefer primary sources, academic papers,
    official documentation, and authoritative
    technical sources when appropriate.

12. Do not invent specific sources.

13. If the question is simple, do not create
    unnecessary objectives.

14. If the question asks "why", focus on causes,
    benefits, evidence, and relevant factors.

15. If the question asks "how", focus on workflow,
    mechanisms, architecture, and process.

16. If the question asks "what", focus on definition,
    components, characteristics, and purpose.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY a numbered list.

Example:

1. Define the core concept and its purpose.
2. Explain the architecture and workflow.
3. Examine the main components and techniques.
4. Analyze relevant applications or use cases.
5. Identify limitations and open challenges.
"""

        # ==================================================
        # LLM CALL
        # ==================================================

        try:

            response = self.llm.generate(
                prompt
            )

        except Exception as e:

            print(
                f"[PLANNER] LLM error: {e}"
            )

            return [
                "Identify the core concept and purpose.",
                "Explain the main mechanism or workflow.",
                "Examine the important factors relevant to the question.",
            ]

        # ==================================================
        # VALIDATE RESPONSE
        # ==================================================

        if not response:

            return [
                "Identify the core concept and purpose.",
                "Explain the main mechanism or workflow.",
                "Examine the important factors relevant to the question.",
            ]

        # ==================================================
        # PARSE PLAN
        # ==================================================

        plan = []

        for line in response.splitlines():

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Remove markdown formatting
            # ------------------------------------------------

            line = line.replace(
                "**",
                "",
            ).strip()

            # ------------------------------------------------
            # Remove common markdown list formatting
            # ------------------------------------------------

            if line.startswith("- "):

                line = line[2:].strip()

            # ------------------------------------------------
            # Only accept numbered lines
            # ------------------------------------------------

            if not line:

                continue

            if not line[0].isdigit():

                continue

            # ------------------------------------------------
            # Remove numbering
            #
            # Examples:
            #
            # 1. Define...
            # 2) Explain...
            # 3 - Analyze...
            # ------------------------------------------------

            line = line.lstrip(
                "0123456789.-) "
            ).strip()

            if not line:

                continue

            # ------------------------------------------------
            # Ignore accidental headings
            # ------------------------------------------------

            lowered = line.lower()

            if lowered in {
                "here is the research plan:",
                "here is the research plan",
                "research plan:",
                "research plan",
                "plan:",
                "plan",
            }:

                continue

            # ------------------------------------------------
            # Avoid duplicate objectives
            # ------------------------------------------------

            if line in plan:

                continue

            plan.append(line)

        # ==================================================
        # SAFETY LIMIT
        # ==================================================

        plan = plan[:5]

        # ==================================================
        # FALLBACK
        # ==================================================

        if not plan:

            print(
                "[PLANNER] Could not parse LLM plan. "
                "Using fallback plan."
            )

            plan = [
                "Identify the core concept and purpose.",
                "Explain the main mechanism or workflow.",
                "Examine the important components and factors.",
            ]

        # ==================================================
        # DEBUG OUTPUT
        # ==================================================

        print(
            f"[PLANNER] Generated "
            f"{len(plan)} research objectives."
        )

        return plan