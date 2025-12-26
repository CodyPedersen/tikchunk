from collections.abc import Generator
import tiktoken
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from tikchunk import Chunker, Interval

from .conftest import semantically_chunked_text


def test_chunk_ordering_gutenberg(
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
            last_chunk = chunk


@given(text=semantically_chunked_text())
@settings(deadline=None, max_examples=50)
def test_chunk_ordering_invariant(
    text: str,
    encoding: tiktoken.Encoding,
    max_chunk_size: int
) -> None:
    """Validate ordering property holds for arbitrarily chunked text"""

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
        last_chunk = chunk
