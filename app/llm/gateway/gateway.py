from app.config.settings import LLM_PROVIDER

from app.llm.gemini import GeminiService
from app.llm.ollama import OllamaLLM


class LLMGateway:
    """
    Central entry point for all LLM calls.

    Agents should never directly interact with
    Gemini, Ollama, or any other model provider.
    """

    def __init__(self):

        if LLM_PROVIDER == "gemini":

            self.llm = GeminiService()

        elif LLM_PROVIDER == "ollama":

            self.llm = OllamaLLM()

        else:

            raise ValueError(
                f"Unsupported LLM provider: {LLM_PROVIDER}"
            )

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the configured LLM.
        """

        return self.llm.generate(prompt)