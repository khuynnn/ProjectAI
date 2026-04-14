import os
from .extractor import PDFExtractor
from .processor import HeaderProcessor
from .chunker import MarkdownChunker
from .embedder import VectorEmbedder
from src.utils.log import get_logger

logger = get_logger(__name__)

class IngestionPipeline:
    def __init__(self, input_path, collection_name = "test"):
        logger.info("Initializing IngestionPipeline")
        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))  # thêm 1 cái nữa
                )
            )
        )
        self.chroma_dir = os.path.join(BASE_DIR, "data", "vector_db")
        self.input_path = input_path
        self.collection_name = collection_name
        
        # Khởi tạo các module
        self.extractor = PDFExtractor()
        self.processor = HeaderProcessor()
        self.chunker = MarkdownChunker(merge_length=1024)
        self.embedder = VectorEmbedder()
        logger.info("IngestionPipeline initialized successfully")

        self.vector_db = self.embedder.load_or_create_db(
            persist_dir=self.chroma_dir,
            collection_name=self.collection_name
        )

    def run(self):
        logger.info("Starting ingestion pipeline")
        # BƯỚC 1: PDF -> Markdown (Unfix)
        logger.info("=== STAGE 1: PDF EXTRACTION ===")
        unfix_md = self.extractor.extract_file(self.input_path)

        # BƯỚC 2: Unfix Markdown -> Fix Markdown (Normalize)
        logger.info("=== STAGE 2: HEADER NORMALIZATION ===")
        logger.info(f"Normalizing headers for: {self.input_path}")
        if unfix_md is None:
            logger.error("❌ Extraction returned None → STOP")
            return
        fix_md = self.processor.normalize_headers_single(unfix_md)

        # BƯỚC 3: Fix Markdown -> Chunks -> Embeddings
        logger.info("=== STAGE 3: CHUNKING & EMBEDDING ===")
        
        logger.info(f"Chunking file: {self.input_path}")
        docs = self.chunker.process_text(
            file_id=self.input_path,
            file_title=os.path.splitext(os.path.basename(self.input_path))[0],
            file_content=fix_md,
        )

        # BƯỚC 4: Lưu vào cơ sở dữ liệu vector
        if docs:
            logger.info("=== STAGE 4: VECTOR DATABASE APPENDING ===")
            self.embedder.store_documents(self.vector_db, docs)
            logger.info("Ingestion pipeline completed successfully!")
        else:
            logger.warning("No documents to process - pipeline completed with no output")
        
        logger.info("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    pipeline = IngestionPipeline("data/raw/chapter7_ML.pdf", "test")
    pipeline.run()