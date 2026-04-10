from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModel
from dotenv import load_dotenv

import os
from src.utils.log import get_logger

load_dotenv()



class Retriever:
    def __init__(
        self, 
        model_name="BAAI/bge-m3", 
        persist_dir=None, 
        collection_name="test", 
        reranker_model="jinaai/jina-reranker-v3", 
        device="cuda"
    ):

        if persist_dir is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            persist_dir = os.path.join(BASE_DIR, "data", "vector_db")

        self.logger = get_logger(__name__)
        self.logger.info("Initializing Retriever")

        # Initialize embedding
        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )

        # Initialize vector database
        self.vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding,
            collection_name=collection_name
        )

        # Create retriever
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})

        # Initialize reranker
        self.reranker = AutoModel.from_pretrained(
            reranker_model,
            dtype="auto",
            trust_remote_code=True,
        )
        self.reranker.to(device)
        self.reranker.eval()

        self.logger.info("Retriever initialized successfully")

    def query_and_rerank(self, query):
        self.logger.info(f"Querying for: {query}")

        # Retrieve documents
        docs = self.retriever.invoke(query)
        self.logger.info(f"Retrieved {len(docs)} documents")

        # Rerank documents
        rerank_docs = self.reranker.rerank(
            query,
            [doc.page_content for doc in docs]
        )

        self.logger.info("Reranking completed")
        return rerank_docs

    def combine_docs(self, docs):
        """Convert reranked documents into a single text string."""
        if docs is None:
            return ""

        if isinstance(docs, str):
            return docs

        items = []
        for doc in docs:
            if isinstance(doc, dict) and "document" in doc:
                items.append(str(doc["document"]))
            elif hasattr(doc, "page_content"):
                items.append(str(doc.page_content))
            else:
                items.append(str(doc))

        return "\n\n".join(items)


if __name__ == "__main__":

    retriever = Retriever()
    query = "Công thức hàm đối ngẫu Lagrange là gì"
    results = retriever.query_and_rerank(query)
    combined_text = retriever.combine_docs(results)
    
    print("Combined documents:\n")
    print(combined_text)
    print("\n---\n")
