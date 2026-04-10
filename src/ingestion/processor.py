import re
import os
from src.utils.log import get_logger

logger = get_logger(__name__)

class HeaderProcessor:
    def __init__(self):
        self.header_pattern = re.compile(r'^(#+)\s+(.*)')

    @staticmethod
    def detect_level(text):
        """Xác định cấp độ header dựa trên format số/chữ cái."""
        # 7 / 7.2 / 7.2.1
        if m := re.match(r'^(\d+(?:\.\d+)*)', text):
            return m.group(1).count('.') + 1
        # Roman numeral: I. II.
        if re.match(r'^[IVXLC]+\.', text):
            return 1
        # Alphabet: A. B.
        if re.match(r'^[A-Za-z]\.', text):
            return 2
        return None

    @staticmethod
    def _is_short_title(text):
        return len(text.split()) <= 10

    def _is_fake_header(self, text):
        """Kiểm tra xem có phải header 'giả' (do scan lỗi) không."""
        return (
            self.detect_level(text) is None
            and self._is_short_title(text)
            and not re.search(r'\d', text)
        )

    def normalize_headers(self, input_path, output_path):
        """Hàm chính xử lý file Markdown."""
        logger.info(f"Starting to normalize headers for {input_path}")
        new_lines = []
        in_code_block = False
        seen_main_title = False
        prev_level = None

        if not os.path.exists(input_path):
            logger.error(f"File {input_path} not found.")
            return

        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            # ===== 1. XỬ LÝ CODE BLOCK =====
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                new_lines.append(line)
                continue

            if in_code_block:
                new_lines.append(line)
                continue

            # ===== 2. PHÁT HIỆN HEADER =====
            m = self.header_pattern.match(line)
            if m:
                text = m.group(2).strip()
                level = self.detect_level(text)

                # CASE 1: Header có số (7.1, A., I.)
                if level:
                    prev_level = level
                    line = "#" * level + " " + text + "\n"

                else:
                    # CASE 2: Tiêu đề chính đầu tiên (Main Title)
                    if not seen_main_title and self._is_short_title(text):
                        level = 1
                        prev_level = level
                        seen_main_title = True
                        line = "# " + text + "\n"

                    # CASE 3: Header giả (Text ngắn không số) -> Chuyển thành text thường
                    elif self._is_fake_header(text):
                        new_lines.append(text + "\n")
                        continue

                    # CASE 4: Header ngữ cảnh (Không số nhưng là header thật)
                    else:
                        level = min(prev_level + 1, 4) if prev_level else 2
                        prev_level = level
                        line = "#" * level + " " + text + "\n"

            new_lines.append(line)

        # Ghi file đầu ra
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        logger.info(f"Finished normalizing headers, output written to {output_path}")

    def normalize_headers_single(self, text):
        """Xử lý một text duy nhất và trả về nội dung đã xử lý dưới dạng string."""
        logger.info("Starting header normalization (in-memory)")
        new_lines = []
        in_code_block = False
        seen_main_title = False
        prev_level = None

        lines = text.splitlines(keepends=True)

        for line in lines:
            # ===== 1. XỬ LÝ CODE BLOCK =====
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                new_lines.append(line)
                continue

            if in_code_block:
                new_lines.append(line)
                continue

            # ===== 2. PHÁT HIỆN HEADER =====
            m = self.header_pattern.match(line)
            if m:
                text = m.group(2).strip()
                level = self.detect_level(text)

                # CASE 1: Header có số (7.1, A., I.)
                if level:
                    prev_level = level
                    line = "#" * level + " " + text + "\n"

                else:
                    # CASE 2: Tiêu đề chính đầu tiên (Main Title)
                    if not seen_main_title and self._is_short_title(text):
                        level = 1
                        prev_level = level
                        seen_main_title = True
                        line = "# " + text + "\n"

                    # CASE 3: Header giả (Text ngắn không số) -> Chuyển thành text thường
                    elif self._is_fake_header(text):
                        new_lines.append(text + "\n")
                        continue

                    # CASE 4: Header ngữ cảnh (Không số nhưng là header thật)
                    else:
                        level = min(prev_level + 1, 4) if prev_level else 2
                        prev_level = level
                        line = "#" * level + " " + text + "\n"

            new_lines.append(line)

        return "".join(new_lines)


if __name__ == '__main__':
    import glob
    
    input_dir = "data/processed/unfix_header"
    output_dir = "data/processed/fix_header"
    
    processor = HeaderProcessor()
    
    # Process all markdown files in the input directory
    md_files = glob.glob(os.path.join(input_dir, "*.md"))
    
    for input_file in md_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        processor.normalize_headers(input_file, output_file)