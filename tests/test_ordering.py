from collections.abc import Generator
import tiktoken
from tikchunk import Chunker, Interval


def test_chunk_ordering(
    gutenberg_texts: list[str],
    encoding: tiktoken.Encoding,
    max_chunk_size: int
) -> None:
    """Validate all chunks return in expected monotonic ordering"""

    for text in gutenberg_texts:
        chunker: Chunker = Chunker(
            encoding=encoding,
            text=text,
            max_tokens=max_chunk_size,
            as_text=False
        )
        chunks: Generator[str] = chunker.chunk()

        last_chunk: Interval = next(chunks)
        for chunk in chunks:
            assert last_chunk.end <= chunk.start

