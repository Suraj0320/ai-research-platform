# from app.llm.gateway.gateway import LLMGateway


# def main():
#     gateway = LLMGateway()

#     response = gateway.generate(
#         "Explain what Retrieval-Augmented Generation is in 3 sentences."
#     )

#     print("\nGateway response:\n")
#     print(response)


# if __name__ == "__main__":
#     main()


from app.llm.gateway.gateway import LLMGateway


def main():

    gateway = LLMGateway()

    response = gateway.generate(
        "Explain Retrieval-Augmented Generation (RAG) in exactly 3 sentences."
    )

    print("\nLLM Response:\n")
    print(response)


if __name__ == "__main__":
    main()