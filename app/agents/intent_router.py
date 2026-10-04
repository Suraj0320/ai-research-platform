class IntentRouter:

    MEMORY_KEYWORDS = [
        "previous question",
        "previous questions",
        "what did i ask",
        "what was my question",
        "earlier question",
        "last question",
        "previous conversation",
        "earlier conversation",
        "what did we discuss",
        "what have i asked",
    ]

    @classmethod
    def is_memory_query(cls, question: str) -> bool:

        normalized = question.lower().strip()

        return any(
            keyword in normalized
            for keyword in cls.MEMORY_KEYWORDS
        )