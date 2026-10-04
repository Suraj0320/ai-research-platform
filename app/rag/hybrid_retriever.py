from typing import List, Dict
import re

from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    Hybrid retriever combining:

    1. FAISS semantic retrieval
    2. BM25 keyword retrieval
    3. Reciprocal Rank Fusion (RRF)

    Retrieval flow:

        Query
          |
          +----> FAISS ----> candidate_k
          |
          +----> BM25 -----> candidate_k
                          |
                          v
                     RRF Fusion
                          |
                          v
                       top_k

    Raw FAISS and BM25 scores are NOT directly combined
    because they exist on different scales.
    """

    def __init__(
        self,
        chunks,
        embedding_model,
        vector_store,
        semantic_weight=0.5,
        keyword_weight=0.5,
        rrf_k=60,
    ):

        self.chunks = chunks
        self.embedding_model = embedding_model
        self.vector_store = vector_store

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

        # ==================================================
        # BUILD BM25 INDEX
        # ==================================================

        tokenized_chunks = [
            self._tokenize(chunk)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_chunks
        )

    # ======================================================
    # TOKENIZATION
    # ======================================================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenizer for BM25.
        """

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    # ======================================================
    # BM25 SEARCH
    # ======================================================

    def keyword_search(
        self,
        query: str,
        top_k: int = 30,
    ) -> List[Dict]:
        """
        Retrieve documents using BM25.

        top_k:
            Number of keyword candidates to retrieve.
        """

        query_tokens = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:

            results.append(
                {
                    "index": index,
                    "text": self.chunks[index],
                    "keyword_score": float(
                        scores[index]
                    ),
                }
            )

        return results

    # ======================================================
    # SEMANTIC / FAISS SEARCH
    # ======================================================

    def semantic_search(
        self,
        query: str,
        top_k: int = 30,
    ) -> List[Dict]:
        """
        Retrieve documents using semantic similarity.
        """

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        return results

    # ======================================================
    # GET ORIGINAL CHUNK INDEX
    # ======================================================

    def _get_chunk_index(
        self,
        result,
    ):
        """
        Identify the original chunk index.

        Prefer the index returned by FAISS/BM25.

        If unavailable, fall back to matching
        the chunk text.
        """

        if "index" in result:

            return result["index"]

        text = result.get(
            "text",
            "",
        )

        for i, chunk in enumerate(
            self.chunks
        ):

            if chunk == text:

                return i

        return None

    # ======================================================
    # RECIPROCAL RANK FUSION
    # ======================================================

    def _rrf_score(
        self,
        semantic_rank,
        keyword_rank,
    ):
        """
        Reciprocal Rank Fusion.

        RRF contribution:

            weight / (rrf_k + rank)

        Rank starts at 1.
        """

        score = 0.0

        # --------------------------------------------------
        # Semantic contribution
        # --------------------------------------------------

        if semantic_rank is not None:

            score += (
                self.semantic_weight
                / (
                    self.rrf_k
                    + semantic_rank
                )
            )

        # --------------------------------------------------
        # Keyword contribution
        # --------------------------------------------------

        if keyword_rank is not None:

            score += (
                self.keyword_weight
                / (
                    self.rrf_k
                    + keyword_rank
                )
            )

        return score

    # ======================================================
    # HYBRID RETRIEVAL
    # ======================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        candidate_k: int = 30,
    ) -> List[Dict]:
        """
        Hybrid retrieval pipeline.

        Stage 1:
            FAISS retrieves candidate_k documents.

        Stage 2:
            BM25 retrieves candidate_k documents.

        Stage 3:
            Results are merged.

        Stage 4:
            Reciprocal Rank Fusion calculates
            the hybrid ranking.

        Stage 5:
            top_k documents are returned.

        Example:

            candidate_k = 30
            top_k = 30

        means:

            FAISS -> 30
            BM25  -> 30
            RRF   -> best 30
        """

        # ==================================================
        # SEMANTIC RETRIEVAL
        # ==================================================

        semantic_results = self.semantic_search(
            query,
            top_k=candidate_k,
        )

        # ==================================================
        # KEYWORD RETRIEVAL
        # ==================================================

        keyword_results = self.keyword_search(
            query,
            top_k=candidate_k,
        )

        # ==================================================
        # CREATE CANDIDATE DICTIONARY
        # ==================================================

        candidates = {}

        # ==================================================
        # ADD SEMANTIC RESULTS
        # ==================================================

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):

            index = self._get_chunk_index(
                result
            )

            if index is None:
                continue

            if index not in candidates:

                candidates[index] = {
                    "index": index,
                    "text": self.chunks[index],
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                }

            candidates[index][
                "semantic_rank"
            ] = rank

            candidates[index][
                "semantic_score"
            ] = float(
                result.get(
                    "score",
                    result.get(
                        "semantic_score",
                        0.0,
                    ),
                )
            )

        # ==================================================
        # ADD BM25 RESULTS
        # ==================================================

        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):

            index = self._get_chunk_index(
                result
            )

            if index is None:
                continue

            if index not in candidates:

                candidates[index] = {
                    "index": index,
                    "text": self.chunks[index],
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                }

            candidates[index][
                "keyword_rank"
            ] = rank

            candidates[index][
                "keyword_score"
            ] = float(
                result.get(
                    "keyword_score",
                    0.0,
                )
            )

        # ==================================================
        # CALCULATE RRF SCORES
        # ==================================================

        for candidate in candidates.values():

            candidate[
                "rrf_score"
            ] = self._rrf_score(
                candidate[
                    "semantic_rank"
                ],
                candidate[
                    "keyword_rank"
                ],
            )

        # ==================================================
        # SORT BY RRF SCORE
        # ==================================================

        ranked_candidates = sorted(
            candidates.values(),
            key=lambda x: x[
                "rrf_score"
            ],
            reverse=True,
        )

        # ==================================================
        # RETURN TOP HYBRID RESULTS
        # ==================================================

        results = []

        for candidate in ranked_candidates[
            :top_k
        ]:

            results.append(
                {
                    "index": candidate[
                        "index"
                    ],

                    "text": candidate[
                        "text"
                    ],

                    "semantic_rank": candidate[
                        "semantic_rank"
                    ],

                    "keyword_rank": candidate[
                        "keyword_rank"
                    ],

                    "semantic_score": candidate[
                        "semantic_score"
                    ],

                    "keyword_score": candidate[
                        "keyword_score"
                    ],

                    "rrf_score": candidate[
                        "rrf_score"
                    ],

                    "hybrid_score": candidate[
                        "rrf_score"
                    ],
                }
            )

        return results