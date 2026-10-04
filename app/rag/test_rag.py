from app.rag.loader import load_document
from app.rag.chunker import chunk_text


def main():

    file_path = "data/documents/documents.txt"

    # -----------------------------------------
    # Load document
    # -----------------------------------------

    text = load_document(file_path)

    print("=" * 60)
    print("DOCUMENT LOADER TEST")
    print("=" * 60)

    print(f"Characters: {len(text)}")

    # -----------------------------------------
    # Chunk document
    # -----------------------------------------

    chunks = chunk_text(
        text,
        chunk_size=500,
        chunk_overlap=75,
    )

    print("\n" + "=" * 60)
    print("CHUNKING TEST")
    print("=" * 60)

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):

        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)


if __name__ == "__main__":
    main()