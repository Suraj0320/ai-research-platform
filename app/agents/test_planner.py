from app.agents.planner import ResearchPlanner
from app.agents.state import ResearchState


def main():

    planner = ResearchPlanner()

    state = ResearchState(
        question="What is Retrieval-Augmented Generation and how does it work?"
    )

    plan = planner.create_plan(state)

    print("\nGenerated Research Plan:\n")

    for i, step in enumerate(plan, start=1):
        print(f"{i}. {step}")


if __name__ == "__main__":
    main()