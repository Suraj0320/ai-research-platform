from app.rag.loader import load_document
from app.rag.chunker import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import FAISSVectorStore
from app.rag.retriever import RAGRetriever
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.llm.gateway.gateway import LLMGateway


class RAGPipeline:

    def __init__(self):

        # ======================================================
        # LLM
        # ======================================================

        self.llm = LLMGateway()

        # ======================================================
        # SHARED EMBEDDING MODEL
        # ======================================================

        self.embedding_model = EmbeddingModel()

        # ======================================================
        # CROSS-ENCODER RERANKER
        # ======================================================

        self.reranker = Reranker()

    # ==========================================================
    # BUILD RETRIEVER
    # ==========================================================

    def build_retriever(
        self,
        document_path: str,
        retrieval_method: str = "hybrid",
    ):

        # ------------------------------------------------------
        # Load document
        # ------------------------------------------------------

        text = load_document(
            document_path
        )

        # ------------------------------------------------------
        # Chunk document
        # ------------------------------------------------------

        chunks = chunk_text(
            text,
            chunk_size=500,
            chunk_overlap=100,
        )

        if not chunks:
            raise ValueError(
                "No chunks were generated from the document."
            )

        # ------------------------------------------------------
        # Generate embeddings
        # ------------------------------------------------------

        embeddings = self.embedding_model.encode(
            chunks
        )

        # ------------------------------------------------------
        # Build FAISS vector store
        # ------------------------------------------------------

        vector_store = FAISSVectorStore(
            dimension=embeddings.shape[1]
        )

        vector_store.add(
            embeddings,
            chunks,
        )

        # ------------------------------------------------------
        # Select retrieval method
        # ------------------------------------------------------

        if retrieval_method == "semantic":

            retriever = RAGRetriever(
                chunks=chunks,
                embedding_model=self.embedding_model,
                vector_store=vector_store,
            )

        elif retrieval_method in (
            "bm25",
            "hybrid",
        ):

            retriever = HybridRetriever(
                chunks=chunks,
                embedding_model=self.embedding_model,
                vector_store=vector_store,
            )

        else:

            raise ValueError(
                f"Unknown retrieval method: "
                f"{retrieval_method}. "
                f"Use 'semantic', 'bm25', or 'hybrid'."
            )

        return retriever

    # ==========================================================
    # RETRIEVE
    # ==========================================================

    def retrieve(
        self,
        question: str,
        retriever,
        retrieval_method: str,
        candidate_k: int = 30,
        top_k: int = 20,
        rerank: bool = True,
    ):
        """
        Retrieval pipeline.

        Stage 1:
            Retrieve candidate documents using
            semantic / BM25 / hybrid retrieval.

        candidate_k:
            Number of initial candidate documents.

        Stage 2:
            Cross-encoder reranking.

        top_k:
            Number of documents retained after reranking.

        Example:

            candidate_k = 30
            top_k = 20

        means:

            30 candidates
                ↓
            Cross-encoder
                ↓
            20 reranked documents
        """

        # ======================================================
        # INITIAL RETRIEVAL
        # ======================================================

        if retrieval_method == "bm25":

            results = retriever.keyword_search(
                question,
                top_k=candidate_k,
            )

        elif retrieval_method == "hybrid":

            results = retriever.retrieve(
                question,
                top_k=candidate_k,
                candidate_k=candidate_k,
            )

        elif retrieval_method == "semantic":

            results = retriever.retrieve(
                question,
                top_k=candidate_k,
            )

        else:

            raise ValueError(
                f"Unknown retrieval method: "
                f"{retrieval_method}"
            )

        # ======================================================
        # CROSS-ENCODER RERANKING
        # ======================================================

        if rerank and results:

            results = self.reranker.rerank(
                query=question,
                results=results,
                top_k=top_k,
            )

        else:

            results = results[:top_k]

        return results

    # ==========================================================
    # GENERATE
    # ==========================================================

    def generate(
        self,
        question: str,
        document_path: str,
        candidate_k: int = 30,
        top_k: int = 20,
        context_k: int = 5,
        retrieval_method: str = "hybrid",
        rerank: bool = True,
    ):
        """
        Complete RAG pipeline.

        Flow:

            Document
                ↓
            Chunking
                ↓
            Embeddings
                ↓
            FAISS / BM25 / Hybrid
                ↓
            candidate_k candidates
                ↓
            Cross-encoder reranker
                ↓
            top_k documents
                ↓
            context_k documents
                ↓
            LLM
                ↓
            Final answer

        Default configuration:

            candidate_k = 30
            top_k       = 20
            context_k   = 5
        """

        # ======================================================
        # BUILD RETRIEVER
        # ======================================================

        retriever = self.build_retriever(
            document_path=document_path,
            retrieval_method=retrieval_method,
        )

        # ======================================================
        # RETRIEVE + RERANK
        # ======================================================

        retrieved_documents = self.retrieve(
            question=question,
            retriever=retriever,
            retrieval_method=retrieval_method,
            candidate_k=candidate_k,
            top_k=top_k,
            rerank=rerank,
        )

        # ======================================================
        # FINAL CONTEXT
        # ======================================================

        context_results = retrieved_documents[
            :context_k
        ]

        # ------------------------------------------------------
        # Build context string
        # ------------------------------------------------------

        context_parts = []

        for i, result in enumerate(
            context_results,
            start=1,
        ):

            context_parts.append(
                f"""
[Context {i}]

{result["text"]}
""".strip()
            )

        context = "\n\n".join(
            context_parts
        )

        # ======================================================
        # PROMPT
        # ======================================================

        prompt = f"""
You are a research assistant using
Retrieval-Augmented Generation.

Answer the user's question using ONLY
the provided context.

Rules:

1. Use only information present in the context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the context is insufficient, say:
   "I don't have enough information in the
   provided context."
5. Give a concise but complete answer.
6. Prefer information directly relevant
   to the user's question.
7. Do not mention relevance scores.
8. If the user explicitly asks about the
   retrieval process, explain the retrieval
   information that is present in the context.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

        # ======================================================
        # GENERATE ANSWER
        # ======================================================

        answer = self.llm.generate(
            prompt
        )

        # ======================================================
        # RETURN RESULT
        # ======================================================

        return {
            "question": question,
            "answer": answer,

            # Retrieval configuration
            "retrieval_method": retrieval_method,
            "reranking_enabled": rerank,

            "candidate_k": candidate_k,
            "top_k": top_k,
            "context_k": context_k,

            # Documents
            "retrieved_documents": retrieved_documents,
            "used_context": context_results,

            # Final context
            "context": context,
        }