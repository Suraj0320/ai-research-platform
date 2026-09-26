from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.llm.gemini import GeminiService


app = FastAPI(
    title="AI Research & Analysis Platform",
    description="Production-oriented agentic AI platform",
    version="0.1.0",
)


class GenerateRequest(BaseModel):
    prompt: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": "AI Research & Analysis Platform",
        "version": "0.1.0",
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        gemini = GeminiService()

        response = gemini.generate(request.prompt)

        return {
            "success": True,
            "response": response,
        }

    except Exception as e:
        print(f"Gemini generation error: {repr(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )