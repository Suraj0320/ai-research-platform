from app.agents.state import ResearchState
from app.agents.analyzer import ResearchAnalyzer


def main():

    state = ResearchState(
        question="What is Retrieval-Augmented Generation?"
    )

    # -------------------------------------------------
    # Search results
    # -------------------------------------------------

    state.search_results = [
        {
            "success": True,
            "query": "What is Retrieval-Augmented Generation?",
            "answer": (
                "Retrieval-Augmented Generation combines "
                "information retrieval with language generation."
            ),
            "citations": [
                {
                    "title": "What Is Retrieval-Augmented Generation",
                    "url": (
                        "https://blogs.nvidia.com/blog/"
                        "what-is-retrieval-augmented-generation"
                    ),
                }
            ],
        }
    ]

    # -------------------------------------------------
    # Ranked sources
    # -------------------------------------------------

    state.ranked_sources = [
        {
            "title": "What Is Retrieval-Augmented Generation",
            "url": (
                "https://blogs.nvidia.com/blog/"
                "what-is-retrieval-augmented-generation"
            ),
            "authority": "HIGH",
            "relevance": "HIGH",
            "usefulness": "HIGH",
            "score": 1.0,
            "rank": 1,
            "reason": (
                "Highly relevant and authoritative source "
                "for understanding RAG."
            ),
        }
    ]

    # -------------------------------------------------
    # Run analyzer
    # -------------------------------------------------

    analyzer = ResearchAnalyzer()

    analysis = analyzer.analyze(state)

    # -------------------------------------------------
    # Display result
    # -------------------------------------------------

    print("\n")
    print("=" * 60)
    print("RESEARCH ANALYSIS")
    print("=" * 60)

    print(analysis)

    print("=" * 60)


if __name__ == "__main__":
    main()