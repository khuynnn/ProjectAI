import re

def detect_level(text):
    # numeric section: 7 hoặc 7.2 hoặc 7.2.1
    m = re.match(r'^(\d+(?:\.\d+)*)', text)
    if m:
        return m.group(1).count('.') + 1

    # Roman numeral: I. II. III.
    if re.match(r'^[IVXLC]+\.', text):
        return 1

    # Alphabet: A. B. C. hoặc a. b. c.
    if re.match(r'^[A-Za-z]\.', text):
        return 2

    return None


def normalize_headers(input_md, output_md):

    header_pattern = re.compile(r'^(#+)\s+(.*)')

    new_lines = []
    in_code_block = False

    with open(input_md, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:

        # detect code block
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue

        # nếu đang trong code block thì bỏ qua
        if in_code_block:
            new_lines.append(line)
            continue

        m = header_pattern.match(line)

        if m:
            text = m.group(2).strip()

            level = detect_level(text)

            if level:
                line = "#" * level + " " + text + "\n"
            else:
                # header không có số
                line = "#### " + text + "\n"

        new_lines.append(line)

    with open(output_md, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


input_md = "../data/processed/unfix_header/chapter7.md"
output_md = "../data/processed/fix_header/chapter7.md"

normalize_headers(input_md, output_md)