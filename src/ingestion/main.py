import os
from extractor import PDFExtractor
from processor import HeaderProcessor
from chunker import MarkdownChunker
from embedder import VectorEmbedder
from src.utils.log import get_logger

logger = get_logger(__name__)

class IngestionPipeline:
    def __init__(self):
        logger.info("Initializing IngestionPipeline")
        # Định nghĩa đường dẫn dựa trên ảnh explorer của bạn
        self.base_data = "data"
        self.raw_dir = os.path.join(self.base_data, "raw")
        self.unfix_dir = os.path.join(self.base_data, "processed", "unfix_header")
        self.fix_dir = os.path.join(self.base_data, "processed", "fix_header")
        self.chroma_dir = os.path.join(self.base_data, "chroma_db")

        # Khởi tạo các module
        self.extractor = PDFExtractor()
        self.processor = HeaderProcessor()
        self.chunker = MarkdownChunker(merge_length=1024)
        self.embedder = VectorEmbedder()
        logger.info("IngestionPipeline initialized successfully")

    def run(self):
        logger.info("Starting ingestion pipeline")
        # BƯỚC 1: PDF -> Markdown (Unfix)
        logger.info("=== STAGE 1: PDF EXTRACTION ===")
        self.extractor.extract_all(self.raw_dir, self.unfix_dir)

        # BƯỚC 2: Unfix Markdown -> Fix Markdown (Normalize)
        logger.info("=== STAGE 2: HEADER NORMALIZATION ===")
        unfix_files = [f for f in os.listdir(self.unfix_dir) if f.endswith(".md")]
        logger.info(f"Found {len(unfix_files)} markdown files to process")
        for f_name in unfix_files:
            logger.info(f"Normalizing headers for: {f_name}")
            self.processor.normalize_headers(
                os.path.join(self.unfix_dir, f_name),
                os.path.join(self.fix_dir, f_name)
            )

        # BƯỚC 3: Fix Markdown -> Chunks -> Embeddings
        logger.info("=== STAGE 3: CHUNKING & EMBEDDING ===")
        all_final_docs = []
        fix_files = [f for f in os.listdir(self.fix_dir) if f.endswith(".md")]
        logger.info(f"Found {len(fix_files)} processed markdown files for chunking")
        
        for f_name in fix_files:
            logger.info(f"Chunking file: {f_name}")
            docs = self.chunker.process_and_merge(os.path.join(self.fix_dir, f_name))
            all_final_docs.extend(docs)
            logger.info(f"Generated {len(docs)} chunks from {f_name}")

        logger.info(f"Total documents collected: {len(all_final_docs)}")

        # BƯỚC 4: Lưu vào cơ sở dữ liệu vector
        if all_final_docs:
            logger.info("=== STAGE 4: VECTOR DATABASE CREATION ===")
            self.embedder.store_documents(all_final_docs, self.chroma_dir)
            logger.info("Ingestion pipeline completed successfully!")
        else:
            logger.warning("No documents to process - pipeline completed with no output")
        
        logger.info("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()