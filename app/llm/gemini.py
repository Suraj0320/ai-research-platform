from google import genai

from app.config.settings import GEMINI_API_KEY


class GeminiService:

    def __init__(self):

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    def generate(self, prompt: str):

        interaction = self.client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt,
        )

        return interaction.output_text