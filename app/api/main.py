from fastapi import FastAPI

app = FastAPI(
    title="AI Research & Analysis Platform",
    description="Production-grade agentic AI platform",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": "AI Research & Analysis Platform",
        "version": "0.1.0",
    }