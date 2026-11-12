"""Tokenize fact text for decoder vocabulary lookup."""

import re


def tokenize_fact(text: str) -> list[str]:
    """Split hyphens/punctuation and remove straight or curly apostrophes."""
    normalized = text.lower().replace("-", " ").replace("'", "").replace("’", "")
    # Preserve special-token strings so the vocabulary filter can reject them.
    return re.findall(r"<[^>\s]+>|\w+", normalized)
