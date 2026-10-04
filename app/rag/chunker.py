import re


def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[str]:

    if not text or not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    # --------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # --------------------------------------------------
    # Split document into paragraphs
    # --------------------------------------------------

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text
        )
        if paragraph.strip()
    ]

    chunks = []
    current_sentences = []
    current_length = 0

    # --------------------------------------------------
    # Build chunks using sentences
    # --------------------------------------------------

    for paragraph in paragraphs:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_length = len(sentence)

            # ------------------------------------------
            # Sentence fits into current chunk
            # ------------------------------------------

            if (
                current_length == 0
                or current_length + sentence_length + 1
                <= chunk_size
            ):

                current_sentences.append(sentence)

                current_length += (
                    sentence_length + 1
                )

                continue

            # ------------------------------------------
            # Current chunk is full
            # ------------------------------------------

            chunk = " ".join(
                current_sentences
            ).strip()

            if chunk:
                chunks.append(chunk)

            # ------------------------------------------
            # Create semantic overlap
            # ------------------------------------------

            overlap_sentences = []

            overlap_length = 0

            for previous_sentence in reversed(
                current_sentences
            ):

                if (
                    overlap_length
                    + len(previous_sentence)
                    + 1
                    > chunk_overlap
                ):
                    break

                overlap_sentences.insert(
                    0,
                    previous_sentence
                )

                overlap_length += (
                    len(previous_sentence) + 1
                )

            # ------------------------------------------
            # Start next chunk with overlap
            # ------------------------------------------

            current_sentences = (
                overlap_sentences
                + [sentence]
            )

            current_length = sum(
                len(s) + 1
                for s in current_sentences
            )

    # --------------------------------------------------
    # Add final chunk
    # --------------------------------------------------

    if current_sentences:

        final_chunk = " ".join(
            current_sentences
        ).strip()

        if final_chunk:
            chunks.append(final_chunk)

    return chunks