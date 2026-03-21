from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.utils.log import get_logger

logger = get_logger(__name__)

class VectorEmbedder:
    def __init__(self, model_name='BAAI/bge-m3', device='cuda'):
        logger.info(f"Initializing VectorEmbedder with model {model_name} on device {device}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        logger.info("VectorEmbedder initialized successfully")

    def store_documents(self, documents, persist_dir):
        logger.info(f"Starting to create vector database at {persist_dir} with {len(documents)} documents")
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=persist_dir
        )
        vector_db.persist()
        logger.info(f"Successfully saved vector database to {persist_dir}")
        return vector_db