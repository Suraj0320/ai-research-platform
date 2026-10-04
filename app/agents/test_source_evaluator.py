from app.tools.web_search import WebSearchTool
from app.agents.source_evaluator import SourceEvaluator


def main():

    search_tool = WebSearchTool()
    evaluator = SourceEvaluator()

    result = search_tool.search(
        "What is Retrieval-Augmented Generation?"
    )

    if not result.get("success", True):

        print("Search failed:")
        print(result.get("error"))

        return

    evaluated = evaluator.evaluate([result])

    print("\n")
    print("=" * 60)
    print("SOURCE EVALUATION")
    print("=" * 60)

    print(f"\nTotal sources: {len(evaluated)}")

    for index, source in enumerate(evaluated, start=1):

        print("\n" + "-" * 60)
        print(f"SOURCE #{index}")

        print("\nTITLE:")
        print(source["title"])

        print("\nURL:")
        print(source["url"])

        print("\nRELEVANCE:")
        print(source["relevance"])

        print("\nAUTHORITY:")
        print(source["authority"])

        print("\nUSEFULNESS:")
        print(source["usefulness"])

        print("\nREASON:")
        print(source["reason"])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()