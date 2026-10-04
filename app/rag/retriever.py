from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import FAISSVectorStore


class RAGRetriever:

    def __init__(
        self,
        chunks,
        embedding_model=None,
        vector_store=None,
    ):

        if not chunks:
            raise ValueError(
                "No chunks provided to retriever."
            )

        self.chunks = chunks

        # --------------------------------------------------
        # Embedding model
        # --------------------------------------------------

        if embedding_model is None:
            embedding_model = EmbeddingModel()

        self.embedding_model = embedding_model

        # --------------------------------------------------
        # FAISS vector store
        # --------------------------------------------------

        if vector_store is None:

            embeddings = self.embedding_model.encode(
                chunks
            )

            vector_store = FAISSVectorStore(
                dimension=embeddings.shape[1]
            )

            vector_store.add(
                embeddings,
                chunks,
            )

        self.vector_store = vector_store

    # ======================================================
    # SEMANTIC SEARCH
    # ======================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        query_embedding = self.embedding_model.encode(
            [query]
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        # --------------------------------------------------
        # Similarity filtering
        # --------------------------------------------------

        filtered_results = [
            result
            for result in results
            if result["score"] >= score_threshold
        ]

        return filtered_results