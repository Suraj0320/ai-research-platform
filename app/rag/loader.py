from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
}


def load_document(file_path: str) -> str:
    """
    Load a supported text-based document.

    Args:
        file_path: Path to the document.

    Returns:
        Document contents as a string.
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

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    return path.read_text(encoding="utf-8")