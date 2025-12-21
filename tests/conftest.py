"""Validate chunk sizes"""
from typing import Generator
import pytest
import tiktoken
import nltk

from hypothesis import strategies as st
from hypothesis.strategies import composite, DrawFn




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
    """Generate text that looks like actual prose."""
    num_paragraphs: int = draw(st.integers(min_value=1, max_value=10))
    paragraphs: list[str] = []

    for _ in range(num_paragraphs):
        num_sentences: int = draw(st.integers(min_value=1, max_value=8))
        sentences: list[str] = []

        for _ in range(num_sentences):
            num_words: int = draw(st.integers(min_value=6, max_value=20))

            words: list[str] = []
            for _ in range(num_words):
                word: str = draw(st.text(
                    alphabet=st.characters(
                        whitelist_categories=('Lu', 'Ll')  # Uppercase & Lowercase
                    ),
                    min_size=1,
                    max_size=12
                ))
                words.append(word)

            sentence_end: str = draw(st.sampled_from([".", "!", "?", ",", ":", ";"]))
            sentence: str = " ".join(words) + sentence_end
            sentences.append(sentence)

        paragraph: str = " ".join(sentences)
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)

