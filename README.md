# TikChunk: Semantic Text Chunker

A Python library for quickly chunking text documents at semantic boundaries while respecting token limits. Designed for RAG (Retrieval-Augmented Generation) applications that need to split documents into meaningful, token-constrained segments.

**Performance**: Extremely performant python-based semantic chunker. Chunks the entire NLTK Gutenberg corpus in ~2.8 seconds on an M1 Mac with max_tokens=512.


## Features

- **Semantic boundary detection**: Splits text at natural breakpoints (paragraphs, sentences, clauses) rather than arbitrary character counts
- **Token-aware chunking**: Uses tiktoken to ensure chunks stay within specified token limits
- **Hierarchical splitting**: Progressively splits at different semantic levels (paragraphs → sentences → clauses → words)
- **Delimiter preservation**: Maintains delimiters in the output for natural text flow
- **Intelligent merging**: Combines smaller chunks to maximize token usage without exceeding limits
- **Flexible output**: Returns either text chunks or interval boundaries


## Quick Start

`pip install tikchunk`

```python
import tiktoken
from tikchunk import Chunker

# Initialize with your text and encoding
encoding = tiktoken.get_encoding("cl100k_base")
text = "Your long document text here..."

# Create chunker with desired max tokens per chunk
chunker = Chunker(
    encoding=encoding,
    text=text,
    max_tokens=512
)

# Generate chunks
for chunk in chunker.chunk():
    print(chunk)
    print("---")
```

## How It Works

The chunker uses a priority-based splitting strategy:

1. **Priority 0**: Paragraph breaks (`\n\n\n`, `\r\n\r\n\r\n`, `\n\n`, `\r\n\r\n`)
2. **Priority 1**: Line breaks and dividers (`\n---\n`, `\n===\n`, `\n***\n`, `\r\n`, `\n`, `\r`)
3. **Priority 2**: Sentence endings (`. `, `! `, `? `, `.`, `!`, `?`)
4. **Priority 3**: Clause separators (`; `, `: `, `;`, `:`, ` -- `, ` — `, ` – `, `--`, `—`, `–`)
5. **Priority 4**: Phrase separators (`, `, `,`, `...`, `…`)
6. **Priority 5**: Word boundaries (` `)

When a chunk exceeds `max_tokens`, the algorithm:
1. Splits at the current priority level
2. Merges adjacent segments up to the token limit
3. Recursively processes any remaining oversized chunks at the next priority level

This ensures text is split at the most semantically meaningful boundaries possible while staying within token constraints.

## API Reference

### `Chunker`

**Parameters:**
- `encoding` (tiktoken.Encoding): The tokenizer encoding to use
- `text` (str): The text to chunk
- `max_tokens` (int, optional): Maximum tokens per chunk. Default: 512
- `as_text` (bool, optional): Return text chunks (True) or Interval objects (False). Default: True

**Methods:**
- `chunk()`: Returns a generator yielding text chunks or Intervals

### `chunk(text, tok_prefix_sum, max_tokens)`

Low-level function for custom chunking workflows.

**Parameters:**
- `text` (str): Text to chunk
- `tok_prefix_sum` (np.ndarray): Prefix sum array of token positions
- `max_tokens` (int): Maximum tokens per chunk

**Returns:**
- `list[Interval]`: List of text intervals representing chunks

## Use Cases

- **RAG pipelines**: Split documents for vector database ingestion
- **Long-context processing**: Break documents into manageable segments for LLM processing
- **Document analysis**: Create semantically coherent text segments for analysis
- **Context window management**: Ensure text fits within model token limits

## Advanced Usage

### Getting Interval Boundaries

```python
chunker = Chunker(
    encoding=encoding,
    text=text,
    max_tokens=512,
    as_text=False  # Return Interval objects
)

for interval in chunker.chunk():
    print(f"Chunk from {interval.start} to {interval.end}")
    print(text[interval.start:interval.end])
```

## Implementation Details

- Uses regex-based pattern matching for efficient delimiter detection
- Employs numpy for fast token prefix sum calculations
- Implements a stack-based iterative approach to avoid recursion performance costs 
- Preserves delimiters to maintain natural text readability

## User Notes
- There is a very rare edge case which is intentionally NOT handled by design in order to ensure optimally fast processing. This occurs for select encodings when tokens both start at the same index and overlap or *would* extend over the input text bounds. This is is *highly unusual* in English text (not found in NLTK corpus or any other English test cases; only found in deliberate fuzzing). This will lead to a +1 - +2 token size over expectation and will have non-critical impact in nearly all usecases.

## License

MIT License

## Contributing

Contributions welcome! 
Please ensure any changes maintain semantic splitting behavior and include appropriate tests. 
As this project is open source, ensure minimum coverage constraints are met, and include relevant property tests created via hypothesis