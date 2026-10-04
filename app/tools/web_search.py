from typing import Dict, List

from tavily import TavilyClient

from app.config.settings import TAVILY_API_KEY


class WebSearchTool:

    def __init__(self):

        self.client = TavilyClient(
            api_key=TAVILY_API_KEY
        )

    def search(self, query: str) -> Dict:

        try:

            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=5,
                include_answer=True,
            )

            citations: List[Dict] = []

            for result in response.get("results", []):

                citations.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                    }
                )

            return {
                "success": True,
                "query": query,
                "answer": response.get("answer", ""),
                "citations": citations,
                "results": response.get("results", []),
                "error": None,
            }

        except Exception as exc:

            print(f"Web search error: {exc!r}")

            return {
                "success": False,
                "query": query,
                "answer": "",
                "citations": [],
                "results": [],
                "error": str(exc),
            }