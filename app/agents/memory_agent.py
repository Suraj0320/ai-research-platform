from app.llm.gateway.gateway import LLMGateway


class MemoryAgent:

    def __init__(self):
        self.llm = LLMGateway()

    def answer(
        self,
        question: str,
        memory_context: str,
    ) -> str:

        prompt = f"""
You are a conversation memory assistant.

CURRENT USER QUESTION:
{question}

PREVIOUS CONVERSATION:
{memory_context}

Your job is to answer questions about the
previous conversation using the supplied memory.

IMPORTANT RULES:

1. Use the conversation memory directly.
2. Do not use web search.
3. Do not use general knowledge.
4. Do not say that evidence is insufficient if
   the requested information exists in memory.
5. If the user asks for the previous question,
   identify the most recent previous user question.
6. If the requested information does not exist
   in memory, clearly say that it is not available.

Return only the answer.
"""

        return self.llm.generate(prompt)