import re
from langchain_text_splitters import MarkdownHeaderTextSplitter
from src.utils.log import get_logger

logger = get_logger(__name__)


class MarkdownChunker:
    def __init__(self, merge_length=1024, special_char="|"):
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

    # ========== CÁC BƯỚC XỬ LÝ CHÍNH ==========
    def process_and_merge(self, file_path):
        logger.info(f"Starting chunking process for {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        lines = text.split("\n")
        headers = []
        ignore_code_block = False

        # 1. Trích xuất và đánh dấu Header
        for line in lines:
            if line.strip() == "```":
                ignore_code_block = not ignore_code_block
            if line.startswith("#") and not ignore_code_block:
                headers.append(line)

        logger.info(f"Extracted {len(headers)} headers from {file_path}")

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

        # 3. Chunking bằng LangChain
        headers_to_split_on = [(f"{self.special_char}{'#'*i}", f"Header {i}") for i in range(1, 5)]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text("\n".join(lines))

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

        # 5. Làm sạch lần cuối và tạo Context
        return self._finalize(merged_docs)

    def _finalize(self, docs):
        logger.info(f"Finalizing {len(docs)} documents")
        final_results = []
        for doc in docs:
            # Sắp xếp header keys
            h_keys = sorted([k for k in doc.metadata if "Header" in k], 
                            key=lambda x: int(re.search(r'\d+', x).group()))
            
            clean_meta = {k: self._flatten_to_str(doc.metadata[k]) for k in h_keys}
            hierarchy = " > ".join(clean_meta.values())
            
            # Regex xóa dấu đặc biệt ở đầu dòng
            clean_text = re.sub(r'^[|#\s]+', '', doc.page_content, flags=re.MULTILINE).strip()
            
            if hierarchy:
                doc.page_content = f"NGỮ CẢNH: {hierarchy}\n{'-'*30}\n{clean_text}"
            else:
                doc.page_content = clean_text
                
            doc.metadata = clean_meta
            final_results.append(doc)
            
        logger.info(f"Completed finalization, returning {len(final_results)} processed documents")
        return final_results