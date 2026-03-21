import re

def detect_level(text):
    # 7 / 7.2 / 7.2.1
    m = re.match(r'^(\d+(?:\.\d+)*)', text)
    if m:
        return m.group(1).count('.') + 1

    # Roman numeral: I. II.
    if re.match(r'^[IVXLC]+\.', text):
        return 1

    # Alphabet: A. B.
    if re.match(r'^[A-Za-z]\.', text):
        return 2

    return None


def is_short_title(text):
    return len(text.split()) <= 10


def is_fake_header(text):
    """
    Heuristic:
    - không có số
    - ngắn
    - không phải dạng section thật
    """
    return (
        detect_level(text) is None
        and is_short_title(text)
        and not re.search(r'\d', text)
    )


def normalize_headers(input_md, output_md):

    header_pattern = re.compile(r'^(#+)\s+(.*)')
    new_lines = []

    in_code_block = False
    seen_main_title = False
    prev_level = None

    with open(input_md, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):

        # ===== CODE BLOCK =====
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue

        if in_code_block:
            new_lines.append(line)
            continue

        # ===== HEADER DETECT =====
        m = header_pattern.match(line)

        if m:
            text = m.group(2).strip()
            level = detect_level(text)

            # ===== CASE 1: NUMERIC HEADER =====
            if level:
                prev_level = level
                line = "#" * level + " " + text + "\n"

            else:
                # ===== CASE 2: MAIN TITLE =====
                if not seen_main_title and is_short_title(text):
                    level = 1
                    prev_level = level
                    seen_main_title = True
                    line = "# " + text + "\n"

                # ===== CASE 3: FAKE HEADER → convert thành text =====
                elif is_fake_header(text):
                    new_lines.append(text + "\n")
                    continue

                # ===== CASE 4: CONTEXT-AWARE HEADER =====
                else:
                    if prev_level:
                        level = min(prev_level + 1, 4)  # không vượt quá H4
                    else:
                        level = 2

                    prev_level = level
                    line = "#" * level + " " + text + "\n"

        new_lines.append(line)

    with open(output_md, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


input_md = "../data/processed/unfix_header/chapter7.md"
output_md = "../data/processed/fix_header/chapter7.md"

normalize_headers(input_md, output_md)