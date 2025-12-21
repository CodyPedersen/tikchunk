from collections.abc import Generator
import tiktoken
from tikchunk import Chunker




def test_chunk_sizes_within_bounds(
    gutenberg_texts: list[str],
    encoding: tiktoken.Encoding,
    max_chunk_size: int
) -> None:
    """Test that all chunks are within the specified max token size."""
    all_chunks: list[str] = []

    for text in gutenberg_texts:
        chunker: Chunker = Chunker(
            encoding=encoding,
            text=text,
            max_tokens=max_chunk_size
        )
        chunks: Generator[str] = chunker.chunk()
        all_chunks.extend(chunks)

    # Validate all chunks are within bounds
    for chunk in all_chunks:
        assert 0 < len(encoding.encode(chunk)) <= max_chunk_size, (
            f"chunk {chunk} failed. Size: {len(encoding.encode(chunk))} vs {max_chunk_size})"
        )
