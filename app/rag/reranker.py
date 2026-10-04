from typing import List, Dict

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder reranker.

    Takes retrieved candidate chunks and scores them
    directly against the user's query.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name

        self.model = CrossEncoder(
            model_name,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 3,
    ) -> List[Dict]:

        if not query or not query.strip():
            return []

        if not results:
            return []

        # ---------------------------------------------
        # Create query-document pairs
        # ---------------------------------------------

        pairs = [
            (
                query,
                result["text"],
            )
            for result in results
        ]

        # ---------------------------------------------
        # Cross-encoder scoring
        # ---------------------------------------------

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        # ---------------------------------------------
        # Add reranker score
        # ---------------------------------------------

        reranked_results = []

        for result, score in zip(
            results,
            scores,
        ):

            updated_result = result.copy()

            updated_result[
                "reranker_score"
            ] = float(score)

            reranked_results.append(
                updated_result
            )

        # ---------------------------------------------
        # Sort by reranker score
        # ---------------------------------------------

        reranked_results.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        # ---------------------------------------------
        # Add final rank
        # ---------------------------------------------

        for rank, result in enumerate(
            reranked_results[:top_k],
            start=1,
        ):

            result["reranker_rank"] = rank

        return reranked_results[:top_k]