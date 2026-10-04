from typing import Dict, List


class MemoryStore:

    def __init__(self):
        # session_id -> conversation history
        self._sessions: Dict[str, List[Dict]] = {}

    # ======================================================
    # CREATE SESSION
    # ======================================================

    def create_session(self, session_id: str):

        if session_id not in self._sessions:
            self._sessions[session_id] = []

    # ======================================================
    # ADD MEMORY
    # ======================================================

    def add(
        self,
        session_id: str,
        question: str,
        answer: str,
    ):

        self.create_session(session_id)

        self._sessions[session_id].append(
            {
                "question": question,
                "answer": answer,
            }
        )

    # ======================================================
    # GET MEMORY
    # ======================================================

    def get(
        self,
        session_id: str,
    ) -> List[Dict]:

        return self._sessions.get(
            session_id,
            [],
        )

    # ======================================================
    # CLEAR MEMORY
    # ======================================================

    def clear(
        self,
        session_id: str,
    ):

        self._sessions.pop(
            session_id,
            None,
        )