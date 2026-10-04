from app.llm.gateway.gateway import LLMGateway


class SourceEvaluator:

    def __init__(self):
        self.llm = LLMGateway()

    # ============================================================
    # MAIN EVALUATION
    # ============================================================

    def evaluate(self, search_results):

        if not search_results:
            return []

        # --------------------------------------------------------
        # Extract sources from search results
        # --------------------------------------------------------

        sources = []

        for result in search_results:

            if not result.get("success", True):
                continue

            query = result.get("query", "")
            citations = result.get("citations", [])

            if not citations:
                continue

            for citation in citations:

                if not isinstance(citation, dict):
                    continue

                sources.append(
                    {
                        "query": query,
                        "title": citation.get("title", ""),
                        "url": citation.get("url", ""),
                    }
                )

        if not sources:
            return []

        # --------------------------------------------------------
        # Build source context
        # --------------------------------------------------------

        source_context_parts = []

        for index, source in enumerate(
            sources,
            start=1
        ):

            source_context_parts.append(
                f"""
SOURCE #{index}

Research query:
{source["query"]}

Title:
{source["title"]}

URL:
{source["url"]}

----------------------------------------
"""
            )

        source_context = "\n".join(
            source_context_parts
        )

        # --------------------------------------------------------
        # Evaluation prompt
        # --------------------------------------------------------

        prompt = f"""
You are a research source evaluation agent.

Your task is to evaluate each research source independently.

{source_context}

For EVERY source, evaluate:

1. Relevance
   - HIGH
   - MEDIUM
   - LOW

2. Authority
   - HIGH
   - MEDIUM
   - LOW

3. Usefulness
   - HIGH
   - MEDIUM
   - LOW

4. Reason
   - One concise sentence explaining the evaluation.

Evaluation guidelines:

- Relevance means how directly the source addresses the
  research objective.
- Authority means the credibility of the publisher/source.
- Usefulness means how useful the source is for answering
  the research question.
- Prefer official documentation, academic papers,
  government sources, standards, and primary sources
  when appropriate.
- Do not assume authority merely from the source title.
- Do not invent information about a source.
- Evaluate only the information provided above.
- Every source MUST receive an evaluation.
- Do not skip any source.
- Keep the exact source numbering.

IMPORTANT:
Return ONLY the evaluation blocks.

Do NOT use Markdown bold.
Do NOT add an introduction.
Do NOT add a conclusion.

Use EXACTLY this format:

SOURCE #1
Relevance: HIGH
Authority: HIGH
Usefulness: HIGH
Reason: One concise sentence.

SOURCE #2
Relevance: MEDIUM
Authority: HIGH
Usefulness: MEDIUM
Reason: One concise sentence.

Continue this format for EVERY source.
"""

        # --------------------------------------------------------
        # Call LLM
        # --------------------------------------------------------

        evaluation_text = self.llm.generate(
            prompt
        )

        # --------------------------------------------------------
        # Debug output
        # --------------------------------------------------------

        print(
            "\n================ SOURCE EVALUATOR RAW OUTPUT ================\n"
        )

        print(evaluation_text)

        print(
            "\n===============================================================\n"
        )

        # --------------------------------------------------------
        # Parse evaluation
        # --------------------------------------------------------

        evaluations = self._parse_evaluations(
            evaluation_text
        )

        print(
            "\nParsed evaluations:",
            len(evaluations)
        )

        print(
            "Expected sources:",
            len(sources)
        )

        # --------------------------------------------------------
        # Build evaluated source objects
        # --------------------------------------------------------

        evaluated_sources = []

        for index, source in enumerate(
            sources
        ):

            if index < len(evaluations):

                evaluation = evaluations[index]

            else:

                evaluation = {
                    "relevance": "UNKNOWN",
                    "authority": "UNKNOWN",
                    "usefulness": "UNKNOWN",
                    "reason": (
                        "Source evaluation was not returned."
                    ),
                }

            evaluated_sources.append(
                {
                    "title": source.get(
                        "title",
                        ""
                    ),

                    "url": source.get(
                        "url",
                        ""
                    ),

                    "query": source.get(
                        "query",
                        ""
                    ),

                    "relevance": evaluation.get(
                        "relevance",
                        "UNKNOWN"
                    ),

                    "authority": evaluation.get(
                        "authority",
                        "UNKNOWN"
                    ),

                    "usefulness": evaluation.get(
                        "usefulness",
                        "UNKNOWN"
                    ),

                    "reason": evaluation.get(
                        "reason",
                        ""
                    ),
                }
            )

        return evaluated_sources

    # ============================================================
    # PARSER
    # ============================================================

    def _parse_evaluations(
        self,
        text
    ):

        evaluations = []

        current = None

        if not text:
            return evaluations

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            # ----------------------------------------------------
            # Remove common Markdown formatting
            # ----------------------------------------------------

            cleaned = (
                line
                .replace("**", "")
                .replace("__", "")
                .strip()
            )

            upper = cleaned.upper()

            # ----------------------------------------------------
            # SOURCE HEADER
            #
            # Handles:
            #
            # SOURCE #1
            # **SOURCE #1**
            # SOURCE #1:
            # **SOURCE #1:**
            # ----------------------------------------------------

            if upper.startswith("SOURCE #"):

                # Save previous source
                if current is not None:

                    evaluations.append(
                        current
                    )

                current = {
                    "relevance": "UNKNOWN",
                    "authority": "UNKNOWN",
                    "usefulness": "UNKNOWN",
                    "reason": "",
                }

                continue

            # ----------------------------------------------------
            # Ignore content before first source
            # ----------------------------------------------------

            if current is None:
                continue

            # ----------------------------------------------------
            # RELEVANCE
            # ----------------------------------------------------

            if upper.startswith(
                "RELEVANCE:"
            ):

                value = self._extract_value(
                    cleaned
                )

                current["relevance"] = (
                    self._normalize_rating(
                        value
                    )
                )

            # ----------------------------------------------------
            # AUTHORITY
            # ----------------------------------------------------

            elif upper.startswith(
                "AUTHORITY:"
            ):

                value = self._extract_value(
                    cleaned
                )

                current["authority"] = (
                    self._normalize_rating(
                        value
                    )
                )

            # ----------------------------------------------------
            # USEFULNESS
            # ----------------------------------------------------

            elif upper.startswith(
                "USEFULNESS:"
            ):

                value = self._extract_value(
                    cleaned
                )

                current["usefulness"] = (
                    self._normalize_rating(
                        value
                    )
                )

            # ----------------------------------------------------
            # REASON
            # ----------------------------------------------------

            elif upper.startswith(
                "REASON:"
            ):

                value = self._extract_value(
                    cleaned
                )

                current["reason"] = value

        # --------------------------------------------------------
        # Save final source
        # --------------------------------------------------------

        if current is not None:

            evaluations.append(
                current
            )

        return evaluations

    # ============================================================
    # VALUE EXTRACTION
    # ============================================================

    @staticmethod
    def _extract_value(
        line
    ):

        if ":" not in line:
            return ""

        return (
            line.split(
                ":",
                1
            )[1]
            .strip()
        )

    # ============================================================
    # NORMALIZE RATING
    # ============================================================

    @staticmethod
    def _normalize_rating(
        value
    ):

        if not value:
            return "UNKNOWN"

        value = (
            value
            .strip()
            .upper()
        )

        # --------------------------------------------------------
        # Remove Markdown formatting
        # --------------------------------------------------------

        value = (
            value
            .replace("*", "")
            .replace("_", "")
            .strip()
        )

        # --------------------------------------------------------
        # Accept only valid ratings
        # --------------------------------------------------------

        if value in {
            "HIGH",
            "MEDIUM",
            "LOW",
        }:

            return value

        # --------------------------------------------------------
        # Sometimes LLM may return:
        #
        # HIGH - ...
        # HIGH (very relevant)
        # --------------------------------------------------------

        first_word = (
            value
            .split()[0]
            .strip(
                ".,:;()-"
            )
            if value.split()
            else ""
        )

        if first_word in {
            "HIGH",
            "MEDIUM",
            "LOW",
        }:

            return first_word

        return "UNKNOWN"