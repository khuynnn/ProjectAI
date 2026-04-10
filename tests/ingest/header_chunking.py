from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain.schema import Document
from typing_extensions import List
import uuid

import os
from chunking_service.utils.load_config import load_config

# Define which markdown headers to split on
HEADERS_TO_SPLIT_ON = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
    ("####", "Header_4"),
    ("#####", "Header_5"),
    ("######", "Header_6"),
]

# Load model configurations
config = load_config()
EMBEDDING_MODEL = (
    config["embedding_model"]["model_name"]
    if config["embedding_model"]["local_path"] is None
    or not os.path.exists(config["embedding_model"]["local_path"])
    else config["embedding_model"]["local_path"]
)
MODEL_LENGTH = config["embedding_model"]["model_length"]
SUB_CHUNK_LENGTH = MODEL_LENGTH // 2
OVERLAP_LENGTH = SUB_CHUNK_LENGTH // 10


class HeaderChunker:
    def __init__(
        self,
        model_path=EMBEDDING_MODEL,
        model_length=MODEL_LENGTH,
        overlap_length=OVERLAP_LENGTH,
        sub_chunk_length=SUB_CHUNK_LENGTH,
    ):
        self.model_length = model_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=sub_chunk_length, chunk_overlap=overlap_length
        )

    def split_markdown_text(
        self, file_id: str, file_title: str, file_content: str
    ) -> List[Document]:
        chunks = []
        sections = self.markdown_splitter.split_text(file_content)
        for section in sections:
            meta = section.metadata.copy()
            # Metadata + context
            meta["file_id"] = file_id
            meta["file_title"] = file_title
            context = f"File title: {file_title}\n"
            # Add ancestor headers to content
            cur_header = section.page_content.split()[0]
            if not cur_header.startswith("#"):  # Get the current header level in text
                cur_header = None
            for head in HEADERS_TO_SPLIT_ON:
                if head[0] != cur_header:
                    if head[1] in meta:  # Ancestors of the current header
                        context += f"{head[0]} {meta[head[1]]}\n"  # Add ancestor header
                    else:
                        meta[head[1]] = ""  # Resolve Milvus missing field issue
            # Handle large chunks
            n_tokens = len(self.tokenizer(context + section.page_content)["input_ids"])
            if n_tokens > self.model_length:
                sub_chunk_contents = self.recursive_splitter.split_text(section.page_content)
                sub_chunks = [
                    Document(
                        page_content=context + sub_content,
                        metadata=meta,
                        id=str(uuid.uuid4()),
                    )
                    for sub_content in sub_chunk_contents
                ]
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    Document(
                        page_content=context + section.page_content,
                        metadata=meta,
                        id=str(uuid.uuid4()),
                    )
                )
        return chunks
