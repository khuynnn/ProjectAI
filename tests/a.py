# from docling.document_converter import DocumentConverter

# source = "../data/raw/chapter7.pdf"
# converter = DocumentConverter()
# doc = converter.convert(source).document
# markdown = doc.export_to_markdown()

# markdown = markdown.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

# with open("../data/processed/chapter7.md", "w", encoding="utf-8") as f:
#     f.write(markdown)

import pypdfium2 as pdfium
from pathlib import Path

input_path = Path("../data/raw/chapter7.pdf")
output_path = Path("../data/processed/chapter7.md")

pdf = pdfium.PdfDocument(input_path)

text = ""

for i in range(len(pdf)):
    page = pdf[i]
    textpage = page.get_textpage()
    page_text = textpage.get_text_range()
    
    # Simple extraction of code blocks and tables
    # This is a naive approach: look for lines that start with typical code block/table markers
    lines = page_text.splitlines()
    in_code_block = False
    in_table = False
    code_block = []
    table_block = []
    for line in lines:
        # Detect code blocks (e.g., lines starting with '    ' or '```')
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            if not in_code_block:
                text += "```python\n" + "\n".join(code_block) + "\n```\n\n"
                code_block = []
            continue
        elif in_code_block or line.startswith("    "):
            code_block.append(line)
            continue

        # Detect tables (e.g., lines containing '|' or tab-separated values)
        if "|" in line or "\t" in line:
            table_block.append(line)
            in_table = True
        elif in_table:
            text += "\n".join(table_block) + "\n\n"
            table_block = []
            in_table = False

        # Add normal text
        if not in_code_block and not in_table:
            text += line + "\n"
    # Add any remaining table at end of page
    if table_block:
        text += "\n".join(table_block) + "\n\n"
    # Add any remaining code block at end of page
    if code_block:
        text += "```python\n" + "\n".join(code_block) + "\n```\n\n"
    text += "\n\n"

output_path.write_text(text, encoding="utf-8")