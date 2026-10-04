from app.tools.web_search import WebSearchTool


def main():

    tool = WebSearchTool()

    query = (
        "What are the latest developments "
        "in Retrieval-Augmented Generation?"
    )

    result = tool.search(query)

    print("\n" + "=" * 60)
    print("WEB SEARCH TEST")
    print("=" * 60)

    print("\nSUCCESS:")
    print(result["success"])

    print("\nQUERY:")
    print(result["query"])

    if result["success"]:

        print("\nSEARCH ANSWER:")
        print(result["answer"])

        print("\nCITATIONS:")

        for citation in result["citations"]:
            print(
                f"- {citation['title']}: "
                f"{citation['url']}"
            )

        print("\nRESULT COUNT:")
        print(len(result["results"]))

    else:

        print("\nWEB SEARCH FAILED:")
        print(result["error"])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()