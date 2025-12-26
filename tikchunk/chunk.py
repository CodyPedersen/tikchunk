"""Chunk text on semantic boundaries"""
from __future__ import annotations
from collections import namedtuple, deque
from collections.abc import Generator
from dataclasses import dataclass

import re

import tiktoken
import numpy as np


Interval = namedtuple('Interval', ['start', 'end'])
IntervalPrio = namedtuple('IntervalPrio', ['interval', 'delimiter_prio'])

DELIMITER_PRIORITY: dict[int, list[str]] = {
    0: ["\n\n\n", "\r\n\r\n\r\n", "\n\n", "\r\n\r\n"],  # Paragraph and section breaks
    1: ["\n---\n", "\n===\n", "\n***\n", "\r\n", "\n", "\r"], # Line breaks and major dividers
    2: [". ", "! ", "? ", ".", "!", "?"],  # Sentence endings
    3: ["; ", ": ", ";", ":", " -- ", " — ", " – ", "--", "—", "–"],  # Clause separators
    4: [", ", ",", "...", "…"], # Phrase separators
    5: [" "], # Word boundaries
}

DELIMITER_RE = {
    i: re.compile('|'.join(re.escape(d) for d in DELIMITER_PRIORITY[i]))
    for i in range(len(DELIMITER_PRIORITY))
}



def build_tok_prefix_sum(encoding_text: str, token_pos: list[int]) -> np.ndarray:
    """Build a prefix sum array of tokens at a given text index"""
    text_len = len(encoding_text)

    positions = np.array(token_pos, dtype=np.int32)

    # Edge case: Uses bincount as more than than one token max start at the same index
    token_starts = np.bincount(positions, minlength=text_len + 1)

    return np.cumsum(token_starts)


def chunk(
    text: str,
    tok_prefix_sum: np.ndarray,
    max_tokens: int = 512
) -> list[Interval]:
    """
    Chunks a given text block at semantic boundaries, such that all chunk tokens are < max_tokens
    """

    def _calculate_tokens(start: int, end: int) -> int:
        """
        Calculates tokens per text string with a 1 token buffer,
        as the prefix sum calcualtes token start positions, not end
        """
        # Note: There are edge-cases, where tokens overlap
        return tok_prefix_sum[end] - tok_prefix_sum[start] + 1

    def _chunk_section_at_prio(interval: Interval, delim_prio: int) -> list[Interval]:
        """
        Further chunk an active Interval at the level associated with delimeter priority
        Preserves delimeters
        """
        if delim_prio >= len(DELIMITER_PRIORITY):
            err = (
                "Unable to split at specified token token chunk size. "
                "Consider increasing max_tokens."
            )
            raise ValueError(err)

        pattern: re.Pattern = DELIMITER_RE[delim_prio]
        spans: list[Interval] = []
        last: int = interval.start

        # Find all splits for priority -> append if chunk is further in text
        for m in pattern.finditer(text, interval.start, interval.end):
            if last <= m.start():
                spans.append(Interval(last, m.end()))
            last: int = m.end()  # consume delimiter

        if last < interval.end:
            spans.append(Interval(last, interval.end))

        return spans


    def _merge_chunks(intervals: list[Interval], max_tokens: int) -> list[Interval]:
        """
        Re-merge ORDERED chunks split on semantic boundaries, up to max tokens.
        """

        merged_intervals: list[Interval] = [intervals.pop(0)]  # Not the most efficient

        for interval in intervals:

            # Test token count with merge
            tentative_tokens: int = _calculate_tokens(
                start=merged_intervals[-1].start, end=interval.end
            )

            # Within token range - swap with merged interval
            if tentative_tokens < max_tokens:
                merged_intervals[-1] = Interval(merged_intervals[-1].start, interval.end)
                continue

            # Outside of max token range
            merged_intervals.append(interval)

        return merged_intervals


    def _chunk_and_merge(interval: Interval, delim_prio: int) -> list[Interval]:
        naive_intervals = _chunk_section_at_prio(interval, delim_prio)
        return _merge_chunks(naive_intervals, max_tokens)


    _root_interval = Interval(start=0, end=len(text))

    token_intervals = deque([IntervalPrio(interval=_root_interval, delimiter_prio=0)])

    final_intervals: list[Interval] = []

    while token_intervals:
        cur_interval, cur_prio = token_intervals.pop()

        cur_interval_tok: int = _calculate_tokens(cur_interval.start, cur_interval.end)

        # if > max tok -> decompose further
        if cur_interval_tok > max_tokens:
            subintervals: list[Interval] = _chunk_and_merge(cur_interval, cur_prio)
            for subinterval in reversed(subintervals):
                token_intervals.append(
                    IntervalPrio(subinterval, cur_prio + 1)
                )
            continue

        # valid subinterval, add to final
        final_intervals.append(cur_interval)

    return final_intervals


@dataclass
class Chunker:
    """Core text chunker class"""
    encoding: tiktoken.Encoding
    text: str
    max_tokens: int = 512
    as_text: bool = True  # False: Case where you strictly need the interval boundaries

    def chunk(self) -> Generator[str | Interval, None, None]:
        """Chunk text at semantic boundaries with max_tokens"""
        toks: list[int] = self.encoding.encode(self.text)
        processed_text, token_pos = self.encoding.decode_with_offsets(toks)
        tok_prefix_sum: np.ndarray = build_tok_prefix_sum(processed_text, token_pos)
        intervals = chunk(processed_text, tok_prefix_sum, self.max_tokens)

        if not self.as_text:
            return iter(intervals)
        return (
            self.text[interval.start:interval.end]
            for interval in intervals
        )
