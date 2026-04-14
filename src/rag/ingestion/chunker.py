from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Optional
from typing_extensions import List
from src.utils.log import get_logger

import re
import uuid
import os


logger = get_logger(__name__)

HEADERS_TO_SPLIT_ON = [
    ("|#", "Header 1"),
    ("|##", "Header 2"),
    ("|###", "Header 3"),
    ("|####", "Header 4"),
    ("|#####", "Header 5"),
    ("|######", "Header 6"),
]

# Load model configurations
# config = load_config()
# EMBEDDING_MODEL = (
#     config["embedding_model"]["model_name"]
#     if config["embedding_model"]["local_path"] is None
#     or not os.path.exists(config["embedding_model"]["local_path"])
#     else config["embedding_model"]["local_path"]
# )
SPLIT_LENGTH = 4096 # In tokens
TOKEN_TO_CHAR_RATIO = 4    # 1 token ~ 4 chars
CHARACTER_OVERLAP = SPLIT_LENGTH // 10
MERGE_LENGTH = 1024 # In chars
SPECIAL_CHAR = "|"

class MarkdownChunker:
    def __init__(
        self,
        # model_path=EMBEDDING_MODEL,
        # model_length=MODEL_LENGTH,
        overlap_length=CHARACTER_OVERLAP,
        character_chunk_size=SPLIT_LENGTH*TOKEN_TO_CHAR_RATIO,
        merge_length=MERGE_LENGTH,
        special_char=SPECIAL_CHAR,
    ):
        # self.model_length = model_length
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=character_chunk_size, chunk_overlap=overlap_length
        )
        self.character_chunk_size = character_chunk_size
        self.merge_length = merge_length
        self.special_char = special_char

    # ========== UTILS ==========
    def _normalize_header(self, header):
        if not header:
            return ""
        return header.lstrip(self.special_char).lstrip("#").strip()

    def _get_depth(self, header):
        content = header[1:] if header.startswith(self.special_char) else header
        match = re.match(r'#+', content)
        return len(match.group()) if match else 0

    def _flatten_to_str(self, v):
        """Làm phẳng metadata header thành chuỗi duy nhất."""
        if isinstance(v, list):
            flat = []
            def _rec(l):
                for item in l:
                    if isinstance(item, list): _rec(item)
                    else: flat.append(str(item).strip())
            _rec(v)
            return " & ".join(list(dict.fromkeys(flat)))
        return str(v).strip()

    def _merge_metadata(self, meta1, meta2):
        merged = meta1.copy()
        for k, v2 in meta2.items():
            v1 = merged.get(k)
            if v1 and v1 != v2:
                merged[k] = [v1, v2] if not isinstance(v1, list) else v1 + [v2]
            else:
                merged[k] = v2
        return merged

    def _finalize(self, docs):
        logger.info(f"Finalizing {len(docs)} documents")
        final_results = []
        for doc in docs:
            file_title = doc.metadata.get("file_title", "Unknown")

            # Sắp xếp header keys
            h_keys = sorted([k for k in doc.metadata if "Header" in k], 
                            key=lambda x: int(re.search(r'\d+', x).group()))
            
            clean_meta = {}
            for k in h_keys:
                flattened = self._flatten_to_str(doc.metadata[k])
                if flattened:
                    clean_meta[k] = flattened
            hierarchy = " > ".join(clean_meta.values())
            
            # Regex xóa dấu đặc biệt ở đầu dòng
            clean_text = re.sub(r'^[|#\s]+', '', doc.page_content).strip()
            
            context_header = f"NGỮ CẢNH: {file_title}"
            if hierarchy:
                context_header += f" > {hierarchy}"
            
            doc.page_content = f"{context_header}\n{'-'*30}\n{clean_text}"
                
            doc.metadata.update(clean_meta)
            final_results.append(doc)
            
        logger.info(f"Completed finalization, returning {len(final_results)} processed documents")
        return final_results

    # ========== MAIN FUNCTION ==========
    def process_text(self, file_id: str, file_title: str, file_content: str) -> List[Document]:
        """Xử lý text trực tiếp thay vì từ file."""
        logger.info("Starting chunking process for text input")
        
        lines = file_content.split("\n")
        headers = []
        ignore_code_block = False

        # 1. Trích xuất và đánh dấu Header
        for line in lines:
            if line.strip() == "```":
                ignore_code_block = not ignore_code_block
            if line.startswith("#") and not ignore_code_block:
                headers.append(line)

        logger.info(f"Extracted {len(headers)} headers from {file_title}")

        marked_headers = [self.special_char + h for h in headers]
        header_ptr = 0
        for i, line in enumerate(lines):
            if header_ptr < len(headers) and line == headers[header_ptr]:
                lines[i] = marked_headers[header_ptr]
                header_ptr += 1

        # 2. Xây dựng mối quan hệ Sibling
        max_depth = max((self._get_depth(h) for h in headers), default=0)
        clean_siblings = {}
        prev_h = None

        for h in marked_headers:
            norm_h = self._normalize_header(h)
            if norm_h not in clean_siblings:
                clean_siblings[norm_h] = {}
            
            if self._get_depth(h) == max_depth and prev_h and self._get_depth(prev_h) == max_depth:
                norm_prev = self._normalize_header(prev_h)
                clean_siblings[norm_prev]["subsequent"] = norm_h
                clean_siblings[norm_h]["previous"] = norm_prev
            prev_h = h

        logger.info(f"Built sibling relationships for {len(clean_siblings)} headers")

        # 3. Chunking by headers
        headers_to_split_on = HEADERS_TO_SPLIT_ON
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text("\n".join(lines))

        valid_splits = []
        for doc in md_header_splits:
            meta_vals = list(doc.metadata.values())
            
            clean_content = re.sub(r'^[|#\s]+', '', doc.page_content).strip()
            
            is_empty_chunk = False
            
            if not clean_content:
                is_empty_chunk = True
            elif meta_vals:
                normalized_content = clean_content.strip().lower()
                flattened_meta = [self._flatten_to_str(v).lower() for v in meta_vals if self._flatten_to_str(v)]
                if normalized_content in flattened_meta:
                    is_empty_chunk = True
            
            if not is_empty_chunk:
                valid_splits.append(doc)
            else:
                logger.debug(f"Đã xóa chunk metadata-only: {doc.metadata}")

        md_header_splits = valid_splits

        header_to_idx = {}
        for idx, doc in enumerate(md_header_splits):
            meta_vals = list(doc.metadata.values())
            if meta_vals:
                deepest = self._normalize_header(str(meta_vals[-1]))
                header_to_idx[deepest] = idx

        logger.info(f"Created {len(md_header_splits)} chunks using LangChain")

        # 4. Merge các section nhỏ
        merged_docs = []
        used_indices = set()

        for i, doc in enumerate(md_header_splits):
            if i in used_indices:
                continue
            
            curr_content = doc.page_content
            curr_meta = doc.metadata.copy()
            meta_vals = list(curr_meta.values())
            curr_h_norm = self._normalize_header(str(meta_vals[-1])) if meta_vals else ""

            if len(curr_content) >= self.merge_length:
                merged_docs.append(doc)
                used_indices.add(i)
                continue

            target_idx = None
            sib = clean_siblings.get(curr_h_norm, {})
            
            # Ưu tiên tìm chunk cùng header hoặc sibling
            for j, other in enumerate(md_header_splits):
                if i != j and j not in used_indices:
                    other_h = self._normalize_header(str(list(other.metadata.values())[-1])) if other.metadata else ""
                    if other_h == curr_h_norm:
                        target_idx = j
                        break
            
            if target_idx is None:
                for relate in ["previous", "subsequent"]:
                    neighbor_h = sib.get(relate)
                    if neighbor_h in header_to_idx:
                        idx = header_to_idx[neighbor_h]
                        if idx not in used_indices:
                            target_idx = idx
                            break

            if target_idx is not None:
                target_doc = md_header_splits[target_idx]
                if target_idx < i:
                    doc.page_content = target_doc.page_content + "\n\n" + curr_content
                else:
                    doc.page_content = curr_content + "\n\n" + target_doc.page_content
                doc.metadata = self._merge_metadata(curr_meta, target_doc.metadata)
                used_indices.add(target_idx)
            
            merged_docs.append(doc)
            used_indices.add(i)

        logger.info(f"Merged chunks into {len(merged_docs)} final documents")

        # 5. Chunking by texts
        chunks = []
        for section in merged_docs:
            meta = {
                k: v for k, v in section.metadata.items()
                if not (k.startswith("Header") and not self._flatten_to_str(v))
            }
            # Metadata + context
            meta["file_id"] = file_id
            meta["file_title"] = file_title
            
            # Handle large chunks by character length
            if len(section.page_content) > self.character_chunk_size:
                sub_chunk_contents = self.recursive_splitter.split_text(section.page_content)
                sub_chunks = [
                    Document(
                        page_content=sub_content,
                        metadata=meta,
                        id=str(uuid.uuid4()),
                    )
                    for sub_content in sub_chunk_contents
                ]
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    Document(
                        page_content=section.page_content,
                        metadata=meta,
                        id=str(uuid.uuid4()),
                    )
                )

        # 6. Làm sạch lần cuối và tạo Context
        return self._finalize(chunks)

    def process_folder(self, folder_path: str, encoding: str = "utf-8", extensions: Optional[List[str]] = None) -> List[Document]:
        """Process all files in a single folder and return the combined chunk documents."""
        logger.info(f"Starting folder processing for: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"The provided path is not a directory: {folder_path}")

        documents: List[Document] = []
        file_names = sorted(os.listdir(folder_path))

        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.isfile(file_path):
                continue

            if extensions is not None:
                if not any(file_name.lower().endswith(ext.lower()) for ext in extensions):
                    logger.debug(f"Skipping file due to extension filter: {file_name}")
                    continue

            try:
                with open(file_path, "r", encoding=encoding) as file:
                    content = file.read()
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                continue

            logger.info(f"Processing file: {file_name}")
            file_title = os.path.splitext(file_name)[0]
            try:
                file_docs = self.process_text(
                    file_id=file_path,
                    file_title=file_title,
                    file_content=content,
                )
                documents.extend(file_docs)
            except Exception as e:
                logger.warning(f"Failed to process file {file_path}: {e}")

        logger.info(f"Completed folder processing: {len(documents)} documents generated")
        return documents


if __name__ == "__main__":
    input_dir = "data/processed/fix_header"

    chunker = MarkdownChunker()
    docs = chunker.process_folder(input_dir)

    