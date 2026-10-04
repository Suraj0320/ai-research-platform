from app.llm.gemini import GeminiService


def main():
    llm = GeminiService()

    response = llm.generate(
        "Explain what an AI agent is in 3 sentences."
    )

    print("\nGemini response:\n")
    print(response)


if __name__ == "__main__":
    main()