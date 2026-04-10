import torch
import os
from transformers import AutoModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils.log import get_logger

logger = get_logger(__name__)



class Retriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        persist_dir: str = None,
        collection_name: str = "test",
        device: str = "cuda",
        reranker_name: str = "jinaai/jina-reranker-v3",
    ):
        self.model_name = model_name

        if persist_dir is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            persist_dir = os.path.join(BASE_DIR, "data", "vector_db")

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.device = device
        self.reranker_name = reranker_name

        self.embedding = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
        )

        self.vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding,
            collection_name=self.collection_name,
        )

        self.reranker_model = self._load_reranker()

    def _load_reranker(self):
        device = "cuda" if torch.cuda.is_available() and self.device == "cuda" else "cpu"
        logger.info("Loading reranker model '%s' on device '%s'", self.reranker_name, device)
        model = AutoModel.from_pretrained(
            self.reranker_name,
            dtype="auto",
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        logger.info("Reranker model loaded successfully")
        return model

    def similarity_search(self, query: str, k: int = 5):
        logger.info("Running similarity search for query='%s' with k=%d", query, k)
        results = self.vector_db.similarity_search(query, k=k)
        logger.info("Retrieved %d candidate documents", len(results))
        return results

    def rerank(self, query: str, documents: list[str]):
        """Rerank candidate documents using the Jina reranker model."""
        if not documents:
            logger.warning("Empty documents → skip rerank")
            return []

        logger.info("Reranking %d documents for query='%s'", len(documents), query)
        results = self.reranker_model.rerank(query, documents)
        logger.info("Reranking completed")
        return results

    def search_with_rerank(self, query: str, k: int = 5):
        logger.info("Starting search_with_rerank for query='%s' with k=%d", query, k)
        docs = self.similarity_search(query, k=k)
        documents = [doc.page_content for doc in docs]
        reranked_results = self.rerank(query, documents)
        logger.info("search_with_rerank completed with %d reranked results", len(reranked_results))
        return reranked_results


if __name__ == "__main__":
    retriever = Retriever()
    query_text = "Công thức hàm đối ngẫu Lagrange là gì"

    logger.info("Searching and reranking documents for query")
    reranked_results = retriever.search_with_rerank(query_text, k=5)

    for index, result in enumerate(reranked_results, start=1):
        logger.info("Reranked [%d] score=%.4f document=%s", index, result["relevance_score"], result["document"])
        print(f"[{index}] Score: {result['relevance_score']:.4f}")
        print(result["document"])
        print()

