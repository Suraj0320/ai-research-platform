from app.agents.research_agent import ResearchAgent


def main():

    agent = ResearchAgent()

    question = """
What is Retrieval-Augmented Generation,
how does it work, and what are its major limitations?
"""

    state = agent.run(question)

    print("\n" + "=" * 60)
    print("RESEARCH AGENT TEST")
    print("=" * 60)

    print("\nSTATUS:")
    print(state.status)

    print("\nQUESTION:")
    print(state.question)

    print("\nRESEARCH PLAN:")

    for i, step in enumerate(state.plan, 1):
        print(f"{i}. {step}")

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    for i, result in enumerate(state.search_results, 1):

        print(f"\n--- Search Result {i} ---")

        print("Success:")
        print(result.get("success"))

        print("\nQuery:")
        print(result.get("query"))

        if result.get("success"):

            print("\nAnswer:")
            print(result.get("answer"))

            print("\nCitations:")

            for citation in result.get("citations", []):
                print(
                    f"- {citation.get('title')}: "
                    f"{citation.get('url')}"
                )

        else:

            print("\nError:")
            print(result.get("error"))

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    print(state.analysis)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(state.final_answer)

    print("\n" + "=" * 60)
    print("FINAL STATUS:")
    print(state.status)
    print("=" * 60)


if __name__ == "__main__":
    main()