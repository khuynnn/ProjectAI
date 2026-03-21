import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


def extract_with_marker(pdf_path, output_dir, converter):
    file_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
    output_file = os.path.join(output_dir, file_name)

    # exist -> delete
    if os.path.exists(output_file):
        print(f"Skip (already exists): {file_name}")
        return

    try:
        print(f"\nProcessing: {os.path.basename(pdf_path)}")

        result = converter(pdf_path)
        markdown = result.markdown

        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"Saved to: {output_file}")

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")


def process_all_pdfs(input_dir, output_dir):
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )

    pdf_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pdf")])

    print(f"Found {len(pdf_files)} PDF files\n")

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n===== [{idx}/{len(pdf_files)}] =====")

        pdf_path = os.path.join(input_dir, pdf_file)
        extract_with_marker(pdf_path, output_dir, converter)


if __name__ == "__main__":
    input_folder = "../data/raw"
    output_folder = "../data/processed/unfix_header"

    process_all_pdfs(input_folder, output_folder)