from google import genai

from app.config.settings import GEMINI_API_KEY


class GeminiService:

    def __init__(self, model: str = "gemini-3.8-flash"):
        self.model = model
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text