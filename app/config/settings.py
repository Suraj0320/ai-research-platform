import os

from dotenv import load_dotenv

load_dotenv()


# ==========================================================
# LLM CONFIGURATION
# ==========================================================

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "mock",
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3:latest",
)


# ==========================================================
# VALIDATION
# ==========================================================

if LLM_PROVIDER == "gemini":

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )


if LLM_PROVIDER == "ollama":

    if not OLLAMA_MODEL:
        raise RuntimeError(
            "OLLAMA_MODEL is not set."
        )


if LLM_PROVIDER != "mock":

    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set."
        )