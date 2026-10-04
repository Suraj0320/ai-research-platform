from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
}


def load_document(file_path: str) -> str:
    """
    Load a supported document.

    Supported:
    - .txt
    - .md
    - .pdf
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    # --------------------------------------------------
    # TXT / MARKDOWN
    # --------------------------------------------------

    if extension in {".txt", ".md"}:
        return path.read_text(
            encoding="utf-8"
        )

    # --------------------------------------------------
    # PDF
    # --------------------------------------------------

    if extension == ".pdf":

        reader = PdfReader(
            str(path)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n\n".join(pages).strip()

    return ""