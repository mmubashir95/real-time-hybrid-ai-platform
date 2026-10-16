"""Demonstrate fixed-size and overlapping word chunking with plain Python.

This standalone learning exercise uses a small synthetic document. It does not
import from, modify, or feed data into the production RAG ingestion pipeline.
Chunks are measured in whitespace-separated words rather than model tokens.
"""

from __future__ import annotations


document = """\
Refund Policy

Customers can request a refund within 30 days of purchase.
Refund requests must include the original order number.
Approved refunds are processed within 5 business days.

Subscription Policy

Subscriptions renew automatically every month.
Customers can cancel their subscriptions at any time.
Cancellation prevents future charges but does not refund previous payments.

Shipping Policy

Standard shipping takes 5 to 7 business days.
Customers receive a tracking number when the order is shipped.
"""


EXPERIMENTS = (
    ("A", 10, 0),
    ("B", 20, 0),
    ("C", 20, 5),
    ("D", 40, 10),
)


def chunk_text(text: str, chunk_size: int = 20) -> list[str]:
    """Split text into ordered chunks containing at most ``chunk_size`` words.

    ``range(0, len(words), chunk_size)`` produces starting indexes separated by
    one complete chunk. ``words[start:end]`` returns words from ``start`` up to,
    but not including, ``end``. Python slicing stops safely at the list boundary,
    which is why the final chunk can contain fewer than ``chunk_size`` words.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    words = text.split()
    chunks: list[str] = []

    # Each start value points to the first word of the next non-overlapping chunk.
    for start in range(0, len(words), chunk_size):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

    return chunks


def chunk_text_with_overlap(
    text: str,
    chunk_size: int = 20,
    overlap: int = 5,
) -> list[str]:
    """Split text into word chunks that repeat words at adjacent boundaries.

    The step is ``chunk_size - overlap`` because each new chunk moves past only
    its non-overlapping words. For example, with five-word chunks and an overlap
    of two, each new chunk begins three words after the previous one.

    Given ``A B C D E F G H I J K L``, a chunk size of 5 and overlap of 2
    produces ``A B C D E``, ``D E F G H``, ``G H I J K``, and ``J K L``.
    Adjacent chunks share two words because their starting indexes differ by 3.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    words = text.split()
    chunks: list[str] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))

        # Once a chunk reaches the final source word, another step would contain
        # only already-seen overlap words or no words at all.
        if start + chunk_size >= len(words):
            break

    return chunks


def print_chunks(chunks: list[str]) -> None:
    """Print every chunk in full with its actual word count."""
    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(f"Word count: {len(chunk.split())}")
        print(chunk)


def show_default_chunking() -> None:
    """Display the requested introductory 20-word chunking result."""
    chunks = chunk_text(document, chunk_size=20)
    print("FIXED-SIZE CHUNKING DEMO")
    print(f"Total chunks: {len(chunks)}")
    print_chunks(chunks)


def show_chunk_size_comparison() -> None:
    """Display full non-overlapping results for 10, 20, and 40 words."""
    print("\n\nCHUNK SIZE COMPARISON")
    for chunk_size in (10, 20, 40):
        chunks = chunk_text(document, chunk_size=chunk_size)
        print(f"\nConfigured chunk size: {chunk_size}")
        print(f"Total chunks: {len(chunks)}")
        print_chunks(chunks)


def calculate_duplicate_words(chunks: list[str], source_text: str) -> int:
    """Count additional word occurrences introduced by overlapping chunks."""
    return sum(len(chunk.split()) for chunk in chunks) - len(source_text.split())


def run_configured_experiments() -> None:
    """Run the four requested configurations and print their comparison table."""
    total_document_words = len(document.split())
    comparison_rows: list[tuple[str, int, int, int, int]] = []

    print("\n\nOVERLAP EXPERIMENTS")
    for experiment_name, chunk_size, overlap in EXPERIMENTS:
        chunks = chunk_text_with_overlap(document, chunk_size, overlap)
        duplicate_words = calculate_duplicate_words(chunks, document)
        comparison_rows.append(
            (
                experiment_name,
                chunk_size,
                overlap,
                len(chunks),
                duplicate_words,
            )
        )

        print(f"\nExperiment: {experiment_name}")
        print(f"Chunk size: {chunk_size}")
        print(f"Overlap: {overlap}")
        print(f"Total document words: {total_document_words}")
        print(f"Total chunks: {len(chunks)}")
        print_chunks(chunks)

    print("\n\nEXPERIMENT COMPARISON")
    print("Experiment | Chunk Size | Overlap | Total Chunks | Duplicate Words")
    print("-" * 66)
    for name, chunk_size, overlap, total_chunks, duplicate_words in comparison_rows:
        print(
            f"{name:<10} | {chunk_size:<10} | {overlap:<7} | "
            f"{total_chunks:<12} | {duplicate_words}"
        )


def main() -> None:
    """Run all educational demonstrations when the file is executed directly."""
    show_default_chunking()
    show_chunk_size_comparison()
    run_configured_experiments()


if __name__ == "__main__":
    main()


# Learning questions — intentionally unanswered:
# Q1. Why does increasing chunk size generally reduce the number of chunks?
# Q2. Why might overlap improve retrieval even though it duplicates information?
# Q3. If a document contains 100 words, chunk_size=20, and overlap=5, how many
#     chunks will this implementation create?
# Q4. Why can fixed-size chunking split sentences in the middle?
# Q5. What happens when the document contains fewer words than the configured
#     chunk size?
# Q6. What is the difference between word-based chunking and token-based chunking?
