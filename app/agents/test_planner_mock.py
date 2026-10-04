from app.agents.research_agent import ResearchAgent
from app.agents.state import ResearchState


def main():

    agent = ResearchAgent()

    # Mock the LLM planner
    agent.llm.generate = lambda prompt: """
1. Define Retrieval-Augmented Generation and its purpose.
2. Explain how the retrieval process works.
3. Explain how retrieved context is provided to the language model.
4. Identify common applications of RAG.
5. Discuss important limitations of RAG.
"""

    state = ResearchState(
        question="What is Retrieval-Augmented Generation and how does it work?"
    )

    plan = agent.create_plan(state)

    print("\nGenerated Research Plan:\n")

    for i, step in enumerate(plan, start=1):
        print(f"{i}. {step}")


if __name__ == "__main__":
    main()