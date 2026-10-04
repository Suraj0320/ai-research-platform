from app.agents.state import ResearchState
from app.agents.answer_generator import ResearchAnswerGenerator


def main():

    state = ResearchState(
        question="What is Retrieval-Augmented Generation, how does it work, and what are its major limitations?"
    )

    state.analysis = """
RAG combines a language model with external information retrieval.
The retrieved information is provided to the language model as context.
Important limitations include retrieval quality, outdated or irrelevant
information, latency, and the possibility of incorrect generated answers.
"""

    state.search_results = [
        {
            "success": True,
            "query": "What is Retrieval-Augmented Generation?",
            "answer": (
                "RAG combines information retrieval with language generation "
                "to provide responses using external information."
            ),
            "citations": [
                {
                    "title": "What Is Retrieval-Augmented Generation",
                    "url": "https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation"
                }
            ],
        }
    ]

    generator = ResearchAnswerGenerator()

    answer = generator.generate(state)

    print("\n")
    print("=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60)

    print("\nTYPE:", type(answer))


if __name__ == "__main__":
    main()