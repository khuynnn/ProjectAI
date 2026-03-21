import os
import time
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

from src.utils.log import get_logger

logger = get_logger(__name__)

class PDFExtractor:
    def __init__(self):
        try:
            logger.info("Initialize Marker Model...")
            start = time.time()
            self.converter = PdfConverter(artifact_dict=create_model_dict())
            logger.info(f"Model is ready (Init time: {time.time() - start:.2f}s)")
        except Exception as e:
            logger.error(f"Initialization PDFConverter Error: {e}")
            raise

    def extract_all(self, input_dir, output_dir):

        if not os.path.exists(input_dir):
            logger.warning(f"Input dir is not existing: {input_dir}")
            return

        pdf_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pdf")])
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Find {len(pdf_files)} files PDF in {input_dir}")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            file_name = os.path.splitext(pdf_file)[0] + ".md"
            output_file = os.path.join(output_dir, file_name)

            if os.path.exists(output_file):
                logger.info(f"Skip (existed): {file_name}")
                continue

            try:
                logger.info(f"Starting extraction: {pdf_file}")
                start_time = time.time()

                result = self.converter(pdf_path)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result.markdown)
                
                elapsed = time.time() - start_time
                logger.info(f"Finished: {pdf_file} in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"Extraction Error while handling file {pdf_file}: {str(e)}", exc_info=True)
            
        logger.info("--- Finished extraction PDF file ---")