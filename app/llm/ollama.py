import ollama


class OllamaLLM:

    def __init__(
        self,
        model: str = "llama3:latest",
    ):

        self.model = model

    def generate(
        self,
        prompt: str,
    ) -> str:

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]