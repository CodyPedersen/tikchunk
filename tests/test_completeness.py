from collections.abc import Generator

import tiktoken
import pytest

from hypothesis import given, settings, strategies as st

from tikchunk import Chunker

from .conftest import semantically_chunked_text


def test_chunk_completeness(
    gutenberg_texts: list[str],
    encoding: tiktoken.Encoding,
    max_chunk_size: int
) -> None:
    """Test that all chunks are within the specified max token size."""
    for text in gutenberg_texts:
        chunker: Chunker = Chunker(
            encoding=encoding,
            text=text,
            max_tokens=max_chunk_size,
        )
        chunks: Generator[str] = chunker.chunk()

        reconstructed_text: str = "".join(chunks)

        assert text == reconstructed_text



@given(text=semantically_chunked_text())
@settings(deadline=None, max_examples=50)
def test_completeness_invariant(
    text: str,
    encoding: tiktoken.Encoding,
    max_chunk_size: int
) -> None:
    """Chunk fuzzed text and assert post-chunk reconstruction is valid"""
    chunker: Chunker = Chunker(
        encoding=encoding,
        text=text,
        max_tokens=max_chunk_size,
    )
    chunks: Generator[str] = chunker.chunk()

    reconstructed_text: str = "".join(chunks)

    assert text == reconstructed_text


# TODO: Move me
@given(text=st.text(min_size=1000))
@settings(deadline=None, max_examples=5)
def test_raises_descriptive_error(text: str) -> None:
    """Ensure descriptive failure on unchunkable text"""
    encoding = tiktoken.get_encoding("cl100k_base")
    chunker: Chunker = Chunker(
        encoding=encoding,
        text=text,
        max_tokens=32,
    )
    with pytest.raises(
        ValueError,
        match=r"Unable to split at specified token token chunk size.*"
    ):
        _ = chunker.chunk()
