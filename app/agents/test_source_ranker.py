from app.agents.source_ranker import SourceRanker


def main():

    sources = [
        {
            "title": "NVIDIA RAG Guide",
            "url": "https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation",
            "relevance": "HIGH",
            "authority": "HIGH",
            "usefulness": "HIGH",
        },
        {
            "title": "Wikipedia RAG",
            "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            "relevance": "HIGH",
            "authority": "MEDIUM",
            "usefulness": "HIGH",
        },
        {
            "title": "K2View RAG Guide",
            "url": "https://www.k2view.com/what-is-retrieval-augmented-generation",
            "relevance": "HIGH",
            "authority": "MEDIUM",
            "usefulness": "MEDIUM",
        },
        {
            "title": "Google Cloud RAG",
            "url": "https://cloud.google.com/use-cases/retrieval-augmented-generation",
            "relevance": "HIGH",
            "authority": "HIGH",
            "usefulness": "HIGH",
        },
        {
            "title": "Example Low Quality Source",
            "url": "https://example.com/rag",
            "relevance": "LOW",
            "authority": "LOW",
            "usefulness": "LOW",
        },
    ]

    ranker = SourceRanker()

    ranked_sources = ranker.rank(sources)

    print()
    print("=" * 60)
    print("SOURCE RANKING")
    print("=" * 60)

    for source in ranked_sources:

        print()
        print(f"RANK: {source['rank']}")

        print(f"TITLE: {source['title']}")

        print(f"AUTHORITY: {source['authority']}")

        print(f"RELEVANCE: {source['relevance']}")

        print(f"USEFULNESS: {source['usefulness']}")

        print(f"SCORE: {source['score']}")

        print("-" * 60)


if __name__ == "__main__":
    main()