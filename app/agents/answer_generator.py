from app.llm.gateway.gateway import LLMGateway
from app.agents.state import ResearchState


class ResearchAnswerGenerator:

    def __init__(self):
        self.llm = LLMGateway()

    # ==========================================================
    # SAFE VALUE HELPERS
    # ==========================================================

    @staticmethod
    def _safe_dict(value):
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _safe_list(value):
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def _safe_text(value, default=""):
        if value is None:
            return default

        if isinstance(value, str):
            return value.strip()

        return str(value).strip()

    # ==========================================================
    # BOOLEAN NORMALIZATION
    # ==========================================================

    @staticmethod
    def _safe_bool(value):
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().upper() == "TRUE"

        return False

    # ==========================================================
    # SOURCE CONTEXT
    # ==========================================================

    def _build_source_context(self, ranked_sources):

        ranked_sources = self._safe_list(
            ranked_sources
        )

        if not ranked_sources:
            return "No ranked web sources available."

        source_blocks = []

        for source in ranked_sources:

            source = self._safe_dict(source)

            source_blocks.append(
                f"""
SOURCE

Rank:
{source.get("rank", "UNKNOWN")}

Score:
{source.get("score", "UNKNOWN")}

Title:
{source.get("title", "")}

URL:
{source.get("url", "")}

Authority:
{source.get("authority", "UNKNOWN")}

Relevance:
{source.get("relevance", "UNKNOWN")}

Usefulness:
{source.get("usefulness", "UNKNOWN")}

----------------------------------------
"""
            )

        return "\n".join(source_blocks)

    # ==========================================================
    # RAG DOCUMENT CONTEXT
    # ==========================================================

    def _build_rag_documents_context(
        self,
        retrieved_documents,
    ):

        retrieved_documents = self._safe_list(
            retrieved_documents
        )

        if not retrieved_documents:
            return "No retrieved RAG documents available."

        document_blocks = []

        for i, document in enumerate(
            retrieved_documents,
            1,
        ):

            document = self._safe_dict(
                document
            )

            document_blocks.append(
                f"""
RAG DOCUMENT {i}

Retriever Score:
{document.get("score", "UNKNOWN")}

Reranker Score:
{document.get("reranker_score", "UNKNOWN")}

Content:
{document.get("text", "")}

----------------------------------------
"""
            )

        return "\n".join(
            document_blocks
        )

    # ==========================================================
    # SELECTED CONTEXT
    # ==========================================================

    def _build_used_context(
        self,
        used_context,
    ):

        used_context = self._safe_list(
            used_context
        )

        if not used_context:
            return "No selected RAG context available."

        context_blocks = []

        for i, document in enumerate(
            used_context,
            1,
        ):

            document = self._safe_dict(
                document
            )

            context_blocks.append(
                f"""
SELECTED RAG CONTEXT {i}

Retriever Score:
{document.get("score", "UNKNOWN")}

Reranker Score:
{document.get("reranker_score", "UNKNOWN")}

Content:
{document.get("text", "")}

----------------------------------------
"""
            )

        return "\n".join(
            context_blocks
        )

    # ==========================================================
    # MEMORY CONTEXT NORMALIZATION
    # ==========================================================

    def _build_memory_context(
        self,
        memory_context,
    ):

        memory_context = self._safe_text(
            memory_context
        )

        if not memory_context:
            return (
                "No previous conversation memory available."
            )

        return memory_context

    # ==========================================================
    # QUESTION INTERPRETATION
    # ==========================================================

    def _interpret_question(
        self,
        current_question: str,
        memory_context: str,
    ) -> str:

        interpretation_prompt = f"""
You are a Question Interpretation component.

Your ONLY job is to interpret the CURRENT QUESTION.

You are NOT answering the question.

You must determine whether the current question is a
continuation of the previous conversation.

==================================================
PREVIOUS CONVERSATION
==================================================

{memory_context}

==================================================
CURRENT QUESTION
==================================================

{current_question}

==================================================
MOST IMPORTANT RULE
==================================================

The CURRENT QUESTION has priority.

However, if the CURRENT QUESTION contains an implicit
reference such as:

- it
- this
- that
- they
- them
- this approach
- this method
- this technique
- the above
- the previous approach
- the same method

you MUST resolve that reference using the previous
conversation.

Do NOT choose the subject merely from the words appearing
in the CURRENT QUESTION.

Example:

Previous question:
What is Retrieval-Augmented Generation and how does it work?

Previous answer:
Retrieval-Augmented Generation combines parametric and
non-parametric memory for language generation.

Current question:
Why is it useful for improving the answers generated
by language models?

Correct:

SUBJECT:
Retrieval-Augmented Generation

IS_FOLLOW_UP:
TRUE

REFERENCE_RESOLUTION:
it = Retrieval-Augmented Generation

QUESTION_TYPE:
WHY

ANSWER_FOCUS:
Explain why Retrieval-Augmented Generation is useful for
improving answers generated by language models.

Incorrect:

SUBJECT:
Language models

IS_FOLLOW_UP:
FALSE

==================================================
INTENT
==================================================

Describe what the user wants to know in ONE concise sentence.

==================================================
SUBJECT
==================================================

Identify the actual subject of the CURRENT QUESTION.

If the question contains an implicit reference, resolve it
using PREVIOUS CONVERSATION.

==================================================
QUESTION TYPE
==================================================

Choose exactly ONE:

DEFINITION
HOW
WHY
BENEFITS
LIMITATIONS
COMPARISON
EXAMPLE
FOLLOW_UP
OTHER

Examples:

"What is RAG?"
DEFINITION

"How does RAG work?"
HOW

"Why is RAG useful?"
WHY

"What are the benefits of RAG?"
BENEFITS

"What are the limitations of RAG?"
LIMITATIONS

"How is RAG different from fine-tuning?"
COMPARISON

"Give me an example of RAG."
EXAMPLE

==================================================
FOLLOW-UP DETECTION
==================================================

Set IS_FOLLOW_UP to TRUE when:

1. The current question contains a reference to the
   previous topic.

OR

2. The current question asks for another property of
   the previous subject.

OR

3. The current question would naturally continue the
   previous discussion.

OR

4. The current question is ambiguous without previous
   conversation.

Examples:

Previous:
What is RAG?

Current:
Why is it useful?

TRUE

Previous:
What is RAG?

Current:
What are its limitations?

TRUE

Previous:
What is RAG?

Current:
How does this approach retrieve information?

TRUE

Previous:
What is RAG?

Current:
What is LangGraph?

FALSE

IMPORTANT:

A follow-up question must still receive its actual
QUESTION_TYPE.

For example:

"Why is it useful?"

QUESTION_TYPE:
WHY

IS_FOLLOW_UP:
TRUE

Do NOT use FOLLOW_UP when WHY/HOW/BENEFITS/etc. applies.

==================================================
REFERENCE RESOLUTION
==================================================

Resolve all meaningful references.

Examples:

it = Retrieval-Augmented Generation

this approach = Retrieval-Augmented Generation

this method = Retrieval-Augmented Generation

the previous approach = Retrieval-Augmented Generation

If there is no reference:

REFERENCE_RESOLUTION:
NONE

==================================================
ANSWER FOCUS
==================================================

Describe ONLY what the final answer should explain.

The answer focus must describe the CURRENT QUESTION.

Example:

Previous:
What is RAG and how does it work?

Current:
Why is it useful?

Correct:

ANSWER_FOCUS:
Explain why Retrieval-Augmented Generation is useful for
improving language model answers.

Incorrect:

ANSWER_FOCUS:
Explain what RAG is and how it works.

==================================================
OUTPUT
==================================================

Return ONLY:

INTENT:
<one sentence>

SUBJECT:
<specific subject>

QUESTION_TYPE:
<one allowed type>

IS_FOLLOW_UP:
<TRUE or FALSE>

REFERENCE_RESOLUTION:
<resolution or NONE>

ANSWER_FOCUS:
<one concise sentence>

Do NOT answer the question.

Do NOT add commentary.

Do NOT add markdown.

Do NOT explain your reasoning.
"""

        interpretation = self.llm.generate(
            interpretation_prompt
        )

        return self._safe_text(
            interpretation,
            "INTENT: Unable to interpret question\n"
            "SUBJECT: UNKNOWN\n"
            "QUESTION_TYPE: OTHER\n"
            "IS_FOLLOW_UP: FALSE\n"
            "REFERENCE_RESOLUTION: NONE\n"
            "ANSWER_FOCUS: Answer the current question directly."
        )

    # ==========================================================
    # PARSE QUESTION INTERPRETATION
    # ==========================================================

    @staticmethod
    def _extract_field(
        interpretation: str,
        field_name: str,
    ) -> str:

        interpretation = interpretation or ""

        lines = interpretation.splitlines()

        for line in lines:

            stripped = line.strip()

            if stripped.upper().startswith(
                field_name.upper() + ":"
            ):
                return stripped.split(
                    ":",
                    1,
                )[1].strip()

        return ""

    def _parse_interpretation(
        self,
        interpretation: str,
    ):

        question_type = self._extract_field(
            interpretation,
            "QUESTION_TYPE",
        ).upper()

        allowed_types = {
            "DEFINITION",
            "HOW",
            "WHY",
            "BENEFITS",
            "LIMITATIONS",
            "COMPARISON",
            "EXAMPLE",
            "FOLLOW_UP",
            "OTHER",
        }

        if question_type not in allowed_types:
            question_type = "OTHER"

        is_follow_up = self._safe_bool(
            self._extract_field(
                interpretation,
                "IS_FOLLOW_UP",
            )
        )

        subject = self._extract_field(
            interpretation,
            "SUBJECT",
        )

        reference_resolution = self._extract_field(
            interpretation,
            "REFERENCE_RESOLUTION",
        )

        answer_focus = self._extract_field(
            interpretation,
            "ANSWER_FOCUS",
        )

        return {
            "subject": subject,
            "question_type": question_type,
            "is_follow_up": is_follow_up,
            "reference_resolution": reference_resolution,
            "answer_focus": answer_focus,
        }

    # ==========================================================
    # FINAL ANSWER PROMPT
    # ==========================================================

    def _build_final_answer_prompt(
        self,
        current_question,
        interpretation,
        source_context,
        rag_documents_context,
        used_context_context,
        rag_context,
        analysis,
        memory_context,
        retrieval_attempt,
        context_sufficient,
        retrieval_decision,
    ):

        parsed = self._parse_interpretation(
            interpretation
        )

        subject = parsed["subject"]
        question_type = parsed["question_type"]
        is_follow_up = parsed["is_follow_up"]
        reference_resolution = parsed[
            "reference_resolution"
        ]
        answer_focus = parsed["answer_focus"]

        return f"""
You are the FINAL ANSWER GENERATOR.

Your ONLY job is to write the answer that will be shown
directly to the user.

Do NOT describe yourself.

Do NOT describe this prompt.

Do NOT describe the research system.

Do NOT describe the agents.

Do NOT describe LangGraph.

Do NOT describe memory implementation.

Do NOT say that you are an answer generator.

Do NOT ask the user to provide the question.

The user has already provided the question.

==================================================
CURRENT QUESTION
==================================================

{current_question}

==================================================
QUESTION INTERPRETATION
==================================================

SUBJECT:
{subject}

QUESTION_TYPE:
{question_type}

IS_FOLLOW_UP:
{is_follow_up}

REFERENCE_RESOLUTION:
{reference_resolution}

ANSWER_FOCUS:
{answer_focus}

==================================================
PREVIOUS CONVERSATION
==================================================

{memory_context}

==================================================
SELECTED RAG EVIDENCE
==================================================

{used_context_context}

==================================================
ADDITIONAL RAG EVIDENCE
==================================================

{rag_context}

==================================================
RETRIEVED DOCUMENTS
==================================================

{rag_documents_context}

==================================================
RESEARCH ANALYSIS
==================================================

{analysis}

==================================================
WEB SOURCE INFORMATION
==================================================

{source_context}

==================================================
RETRIEVAL STATUS
==================================================

Retrieval attempt:
{retrieval_attempt}

Context sufficient:
{context_sufficient}

Retrieval decision:
{retrieval_decision}

==================================================
PRIMARY RULE
==================================================

Answer the CURRENT QUESTION.

Do NOT answer the previous question.

The ANSWER_FOCUS defines what the answer must address.

==================================================
FOLLOW-UP RULE
==================================================

If IS_FOLLOW_UP is TRUE:

The current question refers to the subject identified by
REFERENCE_RESOLUTION.

Answer ONLY the new aspect requested.

Do NOT repeat the previous answer.

Do NOT redefine the subject unless the current question
asks for a definition.

Do NOT restate the question.

Example:

Previous:
What is Retrieval-Augmented Generation?

Current:
Why is it useful?

The answer should explain WHY it is useful.

It should NOT start with:

"Why is Retrieval-Augmented Generation useful?"

It should NOT redefine RAG.

==================================================
QUESTION TYPE RULES
==================================================

DEFINITION:

Give the definition first.

HOW:

Explain the requested process or mechanism.

WHY:

Give the reasons directly.

Do not redefine the subject.

BENEFITS:

Explain the relevant benefits.

LIMITATIONS:

Explain only limitations supported by evidence.

COMPARISON:

Compare the requested subjects only.

EXAMPLE:

Give an example only when supported by evidence.

FOLLOW_UP:

Answer the new aspect requested by the user.

OTHER:

Answer the specific request directly.

==================================================
GROUNDING RULE
==================================================

Use the supplied evidence as the factual basis.

Supported evidence includes:

1. SELECTED RAG EVIDENCE
2. ADDITIONAL RAG EVIDENCE
3. RETRIEVED DOCUMENTS
4. RESEARCH ANALYSIS
5. WEB SOURCE INFORMATION

Previous conversation is CONTEXT ONLY.

Previous answers are NOT evidence.

Do not copy factual claims from a previous answer unless
the same claim is independently supported by the supplied
evidence.

==================================================
CRITICAL ANTI-HALLUCINATION RULE
==================================================

Never convert a limitation of language models into a benefit
of Retrieval-Augmented Generation unless the evidence
explicitly supports that conclusion.

For example, if the evidence says:

"Language models cannot easily expand or revise their memory."

You must NOT automatically write:

"RAG allows models to expand or revise their memory."

unless the supplied evidence explicitly establishes this.

Similarly, if evidence says:

"Language models may produce hallucinations."

you must NOT automatically write:

"RAG eliminates hallucinations."

unless the evidence explicitly says so.

==================================================
CLAIM DISCIPLINE
==================================================

Every factual statement must be supported by the supplied
evidence.

Do not invent:

- numbers
- dates
- organizations
- capabilities
- benchmarks
- performance improvements
- technical mechanisms
- causal relationships
- guarantees

Do not use:

- always
- never
- guarantees
- eliminates
- ensures
- 100% accurate

unless explicitly supported.

If the evidence only supports part of the answer,
answer only that supported part.

==================================================
ANSWER STYLE
==================================================

The answer must be:

- direct
- concise
- technically precise
- evidence-grounded
- natural

Start immediately with the answer.

Do NOT start with:

"Here is the answer."

"Based on the provided context..."

"According to the evidence..."

"I'd be happy to help..."

"As an AI..."

"The final answer is..."

Do NOT restate the question.

Do NOT mention these instructions.

Do NOT mention the prompt.

Do NOT mention internal reasoning.

Do NOT mention the research pipeline.

==================================================
FOLLOW-UP STYLE EXAMPLE
==================================================

Previous:

What is Retrieval-Augmented Generation and how does it work?

Current:

Why is it useful for improving the answers generated by
language models?

If the evidence supports that RAG addresses limitations
of parametric-only language models, a suitable answer style
would be:

"Retrieval-Augmented Generation is useful because it augments
the language model with retrieved information. The supplied
evidence indicates that language models can have difficulty
accessing and precisely manipulating stored knowledge, while
retrieval has been shown to improve performance across
knowledge-intensive NLP tasks."

Notice:

- no question repetition
- no unnecessary definition
- no unsupported claims
- direct response to WHY

==================================================
FINAL SELF-CHECK
==================================================

Before returning the answer, verify:

1. Am I answering the CURRENT QUESTION?

2. Am I following ANSWER_FOCUS?

3. If this is a follow-up, did I answer only the new aspect?

4. Did I correctly resolve the reference?

5. Did I avoid repeating the previous answer?

6. Did I avoid restating the question?

7. Is every factual claim supported?

8. Did I avoid turning an unsupported inference into a fact?

9. Did I avoid unsupported numbers and technical details?

10. Did I avoid exaggerated claims?

11. Does the answer sound like a direct response to the user?

If a sentence fails any check, remove or rewrite it.

==================================================
OUTPUT
==================================================

Return ONLY the final answer.
"""

    # ==========================================================
    # PROMPT-LEAK / BAD OUTPUT DETECTION
    # ==========================================================

    @staticmethod
    def _looks_like_prompt_leak(
        answer: str,
    ) -> bool:

        if not answer:
            return True

        text = answer.lower()

        bad_patterns = [
            "as the final answer writer",
            "my task is to answer",
            "please provide the current question",
            "based on the provided context, research analysis",
            "i will verify that my answer",
            "before i begin, i will verify",
            "grounding check",
            "question interpretation",
            "is_follow_up",
            "reference_resolution",
            "answer_focus",
            "final output",
            "output restrictions",
        ]

        matches = sum(
            1
            for pattern in bad_patterns
            if pattern in text
        )

        return matches >= 2

    # ==========================================================
    # HALLUCINATION DETECTION
    # ==========================================================

    def _detect_hallucination(
        self,
        current_question: str,
        final_answer: str,
        rag_context: str,
        used_context: str = "",
        analysis: str = "",
    ) -> str:

        hallucination_prompt = f"""
You are a Hallucination Detection component.

Your ONLY task is to determine whether the FINAL ANSWER
contains factual claims that are unsupported or contradicted
by the SUPPLIED EVIDENCE.

Do NOT answer the question.

Do NOT rewrite the answer.

Do NOT add outside knowledge.

==================================================
CURRENT QUESTION
==================================================

{current_question}

==================================================
SUPPLIED EVIDENCE
==================================================

SELECTED CONTEXT:

{used_context}

ADDITIONAL RAG CONTEXT:

{rag_context}

RESEARCH ANALYSIS:

{analysis}

==================================================
FINAL ANSWER
==================================================

{final_answer}

==================================================
CORE RULE
==================================================

Evaluate every important factual claim in the FINAL ANSWER
against the supplied evidence.

Previous conversation is NOT evidence.

The question itself is NOT evidence.

Outside knowledge is NOT allowed.

==================================================
CLAIM STATUS
==================================================

For each important factual claim choose exactly one:

SUPPORTED
CONTRADICTED
UNSUPPORTED

SUPPORTED:
The evidence directly supports the claim.

CONTRADICTED:
The evidence directly conflicts with the claim.

UNSUPPORTED:
The evidence does not establish the claim.

==================================================
IMPORTANT
==================================================

Do not mark a claim unsupported merely because the exact
wording is different.

Reasonable paraphrases are SUPPORTED when the meaning is
supported.

However, do not infer new capabilities or causal claims.

Example:

Evidence:
"Language models cannot easily expand or revise their memory."

Answer:
"RAG allows language models to expand or revise their memory."

This is NOT supported unless the evidence explicitly connects
RAG with that capability.

==================================================
HALLUCINATION STATUS
==================================================

NO_HALLUCINATION:

All important claims are supported.

POSSIBLE_HALLUCINATION:

At least one important claim is unsupported, but it is not
clearly contradicted.

HALLUCINATION_DETECTED:

At least one important claim is clearly contradicted or a
clearly unsupported factual claim is presented as fact.

==================================================
CONFIDENCE
==================================================

HIGH
MEDIUM
LOW

==================================================
RECOMMENDATION
==================================================

ACCEPT:
All claims adequately supported.

REVIEW:
Some claims are unsupported but not clearly false.

REGENERATE:
At least one important claim is contradicted or clearly
unsupported.

==================================================
OUTPUT
==================================================

HALLUCINATION_STATUS:
<NO_HALLUCINATION | POSSIBLE_HALLUCINATION | HALLUCINATION_DETECTED>

CONFIDENCE:
<HIGH | MEDIUM | LOW>

CLAIMS:

1. CLAIM:
<claim>

STATUS:
<SUPPORTED | CONTRADICTED | UNSUPPORTED>

REASON:
<brief evidence-based reason>

Continue for all important claims.

HALLUCINATED_CLAIMS:
<claims or NONE>

UNSUPPORTED_CLAIMS:
<claims or NONE>

OVERALL_REASON:
<one concise explanation>

RECOMMENDATION:
<ACCEPT | REVIEW | REGENERATE>

Return ONLY this structured analysis.
"""

        hallucination_result = self.llm.generate(
            hallucination_prompt
        )

        return self._safe_text(
            hallucination_result,
            "HALLUCINATION_STATUS: UNKNOWN\n"
            "CONFIDENCE: LOW\n"
            "RECOMMENDATION: REVIEW",
        )

    # ==========================================================
    # HALLUCINATION STATUS PARSER
    # ==========================================================

    @staticmethod
    def _get_hallucination_status(
        result: str,
    ) -> str:

        result = result or ""

        upper_result = result.upper()

        if (
            "HALLUCINATION_STATUS:"
            not in upper_result
        ):
            return "UNKNOWN"

        for status in [
            "HALLUCINATION_DETECTED",
            "POSSIBLE_HALLUCINATION",
            "NO_HALLUCINATION",
        ]:

            if status in upper_result:
                return status

        return "UNKNOWN"

    # ==========================================================
    # MAIN GENERATION
    # ==========================================================

    def generate(
        self,
        state: ResearchState,
    ):

        # ======================================================
        # METADATA
        # ======================================================

        metadata = self._safe_dict(
            getattr(
                state,
                "metadata",
                {},
            )
        )

        # ======================================================
        # CURRENT QUESTION
        # ======================================================

        current_question = self._safe_text(
            getattr(
                state,
                "question",
                "",
            )
        )

        # ======================================================
        # RANKED SOURCES
        # ======================================================

        ranked_sources = self._safe_list(
            getattr(
                state,
                "ranked_sources",
                [],
            )
        )

        source_context = (
            self._build_source_context(
                ranked_sources
            )
        )

        # ======================================================
        # RAG CONTEXT
        # ======================================================

        rag_context = self._safe_text(
            metadata.get(
                "rag_context",
                "",
            )
        )

        if not rag_context:
            rag_context = (
                "No local RAG context available."
            )

        # ======================================================
        # RETRIEVED DOCUMENTS
        # ======================================================

        retrieved_documents = self._safe_list(
            metadata.get(
                "retrieved_documents",
                [],
            )
        )

        rag_documents_context = (
            self._build_rag_documents_context(
                retrieved_documents
            )
        )

        # ======================================================
        # SELECTED CONTEXT
        # ======================================================

        used_context = self._safe_list(
            metadata.get(
                "used_context",
                [],
            )
        )

        used_context_context = (
            self._build_used_context(
                used_context
            )
        )

        # ======================================================
        # ANALYSIS
        # ======================================================

        analysis = self._safe_text(
            getattr(
                state,
                "analysis",
                "",
            )
        )

        if not analysis:
            analysis = (
                "No research analysis was generated."
            )

        # ======================================================
        # MEMORY
        # ======================================================

        memory_context = self._build_memory_context(
            metadata.get(
                "memory_context",
                "",
            )
        )

        # ======================================================
        # AGENTIC RAG INFORMATION
        # ======================================================

        retrieval_attempt = metadata.get(
            "retrieval_attempt",
            0,
        )

        context_sufficient = metadata.get(
            "context_sufficient",
            False,
        )

        retrieval_decision = self._safe_text(
            metadata.get(
                "retrieval_decision",
                "",
            )
        )

        # ======================================================
        # LOGGING
        # ======================================================

        print(
            "\n[ANSWER] Ranked sources:",
            len(ranked_sources),
        )

        print(
            "[ANSWER] Retrieved documents:",
            len(retrieved_documents),
        )

        print(
            "[ANSWER] RAG context characters:",
            len(rag_context),
        )

        print(
            "[ANSWER] Analysis available:",
            bool(analysis),
        )

        print(
            "[ANSWER] Memory available:",
            memory_context
            != "No previous conversation memory available.",
        )

        # ======================================================
        # QUESTION INTERPRETATION
        # ======================================================

        print(
            "\n[ANSWER] Interpreting current question..."
        )

        question_interpretation = (
            self._interpret_question(
                current_question=current_question,
                memory_context=memory_context,
            )
        )

        print(
            "[ANSWER] Question interpretation:"
        )

        print(
            question_interpretation
        )

        parsed_interpretation = (
            self._parse_interpretation(
                question_interpretation
            )
        )

        print(
            "[ANSWER] Parsed subject:",
            parsed_interpretation["subject"],
        )

        print(
            "[ANSWER] Parsed question type:",
            parsed_interpretation["question_type"],
        )

        print(
            "[ANSWER] Parsed follow-up:",
            parsed_interpretation["is_follow_up"],
        )

        print(
            "[ANSWER] Parsed reference:",
            parsed_interpretation[
                "reference_resolution"
            ],
        )

        # ======================================================
        # FINAL ANSWER PROMPT
        # ======================================================

        print(
            "[ANSWER] Generating final answer..."
        )

        final_answer_prompt = (
            self._build_final_answer_prompt(
                current_question=current_question,
                interpretation=question_interpretation,
                source_context=source_context,
                rag_documents_context=rag_documents_context,
                used_context_context=used_context_context,
                rag_context=rag_context,
                analysis=analysis,
                memory_context=memory_context,
                retrieval_attempt=retrieval_attempt,
                context_sufficient=context_sufficient,
                retrieval_decision=retrieval_decision,
            )
        )

        print(
            "[ANSWER] Calling LLM for final answer..."
        )

        raw_final_answer = self.llm.generate(
            final_answer_prompt
        )

        print(
            "[ANSWER] Raw final answer type:",
            type(raw_final_answer),
        )

        print(
            "[ANSWER] Raw final answer:",
            repr(raw_final_answer),
        )

        final_answer = self._safe_text(
            raw_final_answer,
            "Unable to generate final answer.",
        )

        # ======================================================
        # PROMPT LEAK CHECK
        # ======================================================

        if self._looks_like_prompt_leak(
            final_answer
        ):

            print(
                "[ANSWER] Prompt-like output detected."
            )

            repair_prompt = f"""
Rewrite the following output into the actual answer to
the CURRENT QUESTION.

CURRENT QUESTION:
{current_question}

ANSWER FOCUS:
{parsed_interpretation["answer_focus"]}

EVIDENCE:
{used_context_context}

{rag_context}

BAD OUTPUT:
{final_answer}

Rules:

- Answer the question directly.
- Do not mention prompts.
- Do not mention instructions.
- Do not mention being an AI.
- Do not say "please provide the question".
- Do not restate the question.
- Do not explain your task.
- Use only the supplied evidence.
- Return ONLY the final answer.
"""

            repaired_answer = self.llm.generate(
                repair_prompt
            )

            repaired_answer = self._safe_text(
                repaired_answer
            )

            if repaired_answer:
                final_answer = repaired_answer

        print(
            "[ANSWER] Processed final answer:",
            repr(final_answer),
        )

        # ======================================================
        # HALLUCINATION DETECTION
        # ======================================================

        print(
            "[ANSWER] Running hallucination detection..."
        )

        hallucination_result = (
            self._detect_hallucination(
                current_question=current_question,
                final_answer=final_answer,
                rag_context=rag_context,
                used_context=used_context_context,
                analysis=analysis,
            )
        )

        hallucination_status = (
            self._get_hallucination_status(
                hallucination_result
            )
        )

        print(
            "[ANSWER] Hallucination status:",
            hallucination_status,
        )

        # ======================================================
        # REGENERATION IF NEEDED
        # ======================================================

        if hallucination_status in {
            "HALLUCINATION_DETECTED",
            "POSSIBLE_HALLUCINATION",
        }:

            print(
                "[ANSWER] Regenerating due to "
                "hallucination review..."
            )

            regeneration_prompt = f"""
You are correcting a grounded research answer.

CURRENT QUESTION:
{current_question}

QUESTION INTERPRETATION:
{question_interpretation}

SUPPLIED EVIDENCE:
{used_context_context}

{rag_context}

RESEARCH ANALYSIS:
{analysis}

PREVIOUS ANSWER:
{final_answer}

HALLUCINATION REVIEW:
{hallucination_result}

Your task is to write a NEW answer.

Rules:

1. Answer the CURRENT QUESTION only.
2. Follow ANSWER_FOCUS.
3. If this is a follow-up, answer only the new aspect.
4. Do not repeat the previous answer.
5. Do not restate the question.
6. Remove every unsupported factual claim.
7. Remove every contradicted claim.
8. Do not add outside knowledge.
9. Do not invent technical details.
10. Do not mention this review.
11. Do not mention prompts or agents.
12. Return ONLY the corrected answer.

If the evidence cannot establish part of the question,
say so briefly instead of guessing.
"""

            regenerated_answer = self.llm.generate(
                regeneration_prompt
            )

            regenerated_answer = self._safe_text(
                regenerated_answer
            )

            if regenerated_answer:

                print(
                    "[ANSWER] Regenerated answer:"
                )

                print(
                    repr(regenerated_answer)
                )

                final_answer = regenerated_answer

        # ======================================================
        # MEMORY
        # ======================================================

        print(
            "[MEMORY] Conversation saved."
        )

        return final_answer