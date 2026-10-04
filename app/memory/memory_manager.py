from typing import Dict, List

from app.memory.memory_store import MemoryStore


class MemoryManager:

    def __init__(self):

        self.store = MemoryStore()

    # ======================================================
    # LOAD MEMORY
    # ======================================================

    def load_memory(
        self,
        session_id: str,
    ) -> List[Dict]:

        return self.store.get(
            session_id
        )

    # ======================================================
    # FORMAT MEMORY
    # ======================================================

    def format_memory(
        self,
        history: List[Dict],
    ) -> str:

        if not history:
            return ""

        formatted = []

        for i, item in enumerate(
            history,
            start=1,
        ):

            formatted.append(
                f"""
Previous Conversation {i}

User:
{item.get("question", "")}

Assistant:
{item.get("answer", "")}
""".strip()
            )

        return "\n\n----------------------------------------\n\n".join(
            formatted
        )

    # ======================================================
    # SAVE MEMORY
    # ======================================================

    def save_memory(
        self,
        session_id: str,
        question: str,
        answer: str,
    ):

        self.store.add(
            session_id=session_id,
            question=question,
            answer=answer,
        )

    # ======================================================
    # CLEAR MEMORY
    # ======================================================

    def clear_memory(
        self,
        session_id: str,
    ):

        self.store.clear(
            session_id
        )