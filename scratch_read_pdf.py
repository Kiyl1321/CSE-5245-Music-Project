import pypdf
import sys

def read_pdf(file_path):
    try:
        reader = pypdf.PdfReader(file_path)
        with open("scratch_pdf_output.txt", "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                f.write(f"--- Page {i+1} ---\n")
                f.write(page.extract_text() + "\n")
        print("Successfully wrote to scratch_pdf_output.txt")
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        read_pdf(sys.argv[1])
