"""Validate chunk sizes"""
from typing import Generator
import pytest
import tiktoken
import nltk

from hypothesis.strategies import (
    composite,
    DrawFn,
    integers,
    sampled_from,
    booleans
)
from wonderwords import RandomWord

from tikchunk.chunk import DELIMITER_PRIORITY


@pytest.fixture(scope="session", autouse=True)
def download_nltk_data():
    """Download NLTK data before running tests"""
    try:
        nltk.data.find('corpora/gutenberg')
    except LookupError:  # pragma: no cover
        nltk.download('gutenberg', quiet=True)


@pytest.fixture(scope="module")
def gutenberg_texts() -> Generator[str]:
    """Load Gutenberg corpus texts once for all tests."""
    return (
        nltk.corpus.gutenberg.raw(fileid)
        for fileid in nltk.corpus.gutenberg.fileids()
    )


@pytest.fixture(
    scope="module",
    params=["o200k_base", "cl100k_base", "p50k_base", "r50k_base", "gpt2"]
)
def encoding(request: pytest.FixtureRequest) -> tiktoken.Encoding:
    """Load different encodings for comprehensive testing."""
    return tiktoken.get_encoding(request.param)


@pytest.fixture(
    scope="module",
    params=[64, 128, 256, 512, 1024, 2048]
)
def max_chunk_size(request: pytest.FixtureRequest) -> int:
    return request.param


@composite
def semantically_chunked_text(draw: DrawFn) -> str:
    """Generate text that looks like actual prose with varied delimiters."""
    # Calculate total words needed upfront
    num_paragraphs: int = draw(integers(min_value=1, max_value=10))

    # Pre-calculate structure and total word count
    structure: list[list[list[int]]] = []  # paragraphs -> sentences -> chunks (word counts)
    total_words: int = 0

    for _ in range(num_paragraphs):
        num_sentences: int = draw(integers(min_value=1, max_value=8))
        paragraph_structure: list[list[int]] = []

        for _ in range(num_sentences):
            num_chunks: int = draw(integers(min_value=1, max_value=4))
            sentence_structure: list[int] = []

            for _ in range(num_chunks):
                num_words: int = draw(integers(min_value=3, max_value=12))
                sentence_structure.append(num_words)
                total_words += num_words

            paragraph_structure.append(sentence_structure)

        structure.append(paragraph_structure)

    # Generate ALL words in one call
    word_generator = RandomWord()
    all_words: list[str] = word_generator.random_words(
        amount=total_words,
        word_min_length=1,
        word_max_length=12
    )

    # Track position in word list
    word_idx: int = 0
    paragraphs: list[str] = []
    para_sep: str = draw(sampled_from(DELIMITER_PRIORITY[0]))

    for paragraph_structure in structure:
        sentences: list[str] = []

        for sentence_structure in paragraph_structure:
            chunks: list[str] = []

            for num_words in sentence_structure:
                # Extract the next batch of words
                words: list[str] = all_words[word_idx:word_idx + num_words]
                word_idx += num_words

                # Capitalize first word of each chunk
                words[0] = words[0].capitalize()
                chunks.append(" ".join(words))

            # Join chunks with clause separators (priority 3-4)
            if len(chunks) > 1:
                internal_seps: list[str] = []
                for _ in range(len(chunks) - 1):
                    sep_priority: int = draw(sampled_from([3, 4]))
                    sep: str = draw(sampled_from(DELIMITER_PRIORITY[sep_priority]))
                    if not sep.endswith(" "):
                        sep += " "
                    internal_seps.append(sep)

                sentence_parts: list[str] = []
                for i, chunk in enumerate(chunks):
                    sentence_parts.append(chunk)
                    if i < len(internal_seps):
                        sentence_parts.append(internal_seps[i])
                sentence: str = "".join(sentence_parts)
            else:
                sentence = chunks[0]

            # End sentence with priority 2 delimiter
            sentence_end: str = draw(sampled_from(DELIMITER_PRIORITY[2]))
            if not sentence_end.endswith(" "):
                sentence += sentence_end
            else:
                sentence += sentence_end.rstrip()

            sentences.append(sentence)

        # Optionally add line breaks within paragraph (priority 1)
        if draw(booleans()) and len(sentences) > 2:
            split_point: int = draw(integers(min_value=1, max_value=len(sentences) - 1))
            line_sep: str = draw(sampled_from(DELIMITER_PRIORITY[1]))
            paragraph = " ".join(
                sentences[:split_point]) + line_sep + " ".join(sentences[split_point:]
            )
        else:
            paragraph = " ".join(sentences)

        paragraphs.append(paragraph)

    return para_sep.join(paragraphs)
