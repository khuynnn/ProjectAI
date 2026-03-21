import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


def extract_with_marker(pdf_path, output_dir):

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )

    result = converter(pdf_path)
    markdown = result.markdown

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "chapter1_ML.md")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown)

    print("Saved to:", output_file)

if __name__ == "__main__":

    input_pdf = "../data/raw/chapter1_ML.pdf"
    output_folder = "../data/processed/unfix_header"

    extract_with_marker(input_pdf, output_folder)

