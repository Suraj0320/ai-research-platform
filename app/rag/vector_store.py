import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dimension: int):

        self.dimension = dimension

        # Inner Product + normalized embeddings
        # approximates cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)

        self.documents = []

    def add(self, embeddings, documents):

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match "
                "number of documents."
            )

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(
        self,
        query_embedding,
        top_k: int = 5,
    ):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(
                1,
                -1,
            )

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query embedding dimension "
                f"{self.dimension}, "
                f"got {query_embedding.shape[1]}"
            )

        # Never request more results than indexed documents.
        actual_k = min(
            top_k,
            len(self.documents),
        )

        if actual_k == 0:
            return []

        scores, indices = self.index.search(
            query_embedding,
            actual_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index == -1:
                continue

            results.append(
                {
                    "score": float(score),
                    "text": self.documents[index],
                }
            )

        return results