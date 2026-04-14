from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.utils.log import get_logger
import os
from .chunker import MarkdownChunker
import hashlib
import glob


def generate_id(doc):
    content = doc.page_content
    source = doc.metadata.get("source", "")
    raw = source + content
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

logger = get_logger(__name__)

class VectorEmbedder:
    def __init__(self, model_name='BAAI/bge-m3', device='cuda'):
        logger.info(f"Initializing VectorEmbedder with model {model_name} on device {device}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        logger.info("VectorEmbedder initialized successfully")

    def load_or_create_db(self, persist_dir, collection_name):
        """
        Always safe:
        - nếu có DB → load
        - chưa có → auto create
        """
        logger.info(f"Connecting to DB at {persist_dir}")

        vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_model,
            collection_name=collection_name
        )

        return vector_db

    def store_documents(self, vector_db, documents, batch_size=10):
        """
        Incremental ingestion (KHÔNG duplicate)
        """
        logger.info(f"Appending {len(documents)} documents")

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]

            # đảm bảo metadata tồn tại
            for d in batch_docs:
                if not hasattr(d, "metadata") or d.metadata is None:
                    d.metadata = {}

            batch_ids = [generate_id(doc) for doc in batch_docs]

            vector_db.add_documents(batch_docs, ids=batch_ids)

            logger.info(f"Inserted batch {i // batch_size + 1}")

        vector_db.persist()
        logger.info("Append completed")


if __name__ == "__main__":

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[3]

    persist_dir = BASE_DIR / "data/vector_db"
    input_dir = BASE_DIR / "data/processed/fix_header"
    
    collection_name = "test"

    embedder = VectorEmbedder()
    chunker = MarkdownChunker()

    vector_db = embedder.load_or_create_db(
        persist_dir=persist_dir,
        collection_name=collection_name
    )

    documents = []
    md_files = glob.glob(os.path.join(input_dir, "*.md"))

    for file in md_files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        docs = chunker.process_text(
            file_id=file,
            file_title=os.path.splitext(os.path.basename(file))[0],
            file_content=text,
        )

        for d in docs:
            if not hasattr(d, "metadata") or d.metadata is None:
                d.metadata = {}

            d.metadata["source"] = file

        documents.extend(docs)

    if not documents:
        logger.error("No documents found!")
    else:
        embedder.store_documents(vector_db, documents)

        logger.info("Done.")



