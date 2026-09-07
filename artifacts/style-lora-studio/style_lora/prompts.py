"""Prompt pairing and token alignment helpers."""

from __future__ import annotations


def insert_style(content: str, phrase: str) -> str:
    content = content.strip()
    phrase = phrase.strip()
    if content.lower().startswith("a "):
        return f"a {phrase} {content[2:].lstrip()}"
    return f"{phrase} {content}"


def pair_prompts(contents: list[str], phrases: list[str]) -> list[dict]:
    pairs = []
    for i, content in enumerate(contents):
        phrase = phrases[i % len(phrases)] if phrases else ""
        styled = insert_style(content, phrase) if phrase else content
        pairs.append(
            {
                "id": i,
                "content": content,
                "phrase": phrase,
                "styled": styled,
            }
        )
    return pairs


def all_phrase_pairs(contents: list[str], phrases: list[str]) -> list[dict]:
    """Cartesian pairing used when the user wants every phrase on every content."""
    pairs = []
    k = 0
    for content in contents:
        for phrase in phrases:
            pairs.append(
                {
                    "id": k,
                    "content": content,
                    "phrase": phrase,
                    "styled": insert_style(content, phrase),
                }
            )
            k += 1
    return pairs


def tokenize(text: str) -> list[str]:
    out = []
    buf = []
    for ch in text.lower():
        if ch.isalnum() or ch in ("'", "-"):
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def aligned_content_span(content: str, styled: str) -> tuple[list[str], list[str]]:
    """Return (shared_tokens, extra_style_tokens) using LCS on whitespace tokens."""
    a = tokenize(content)
    b = tokenize(styled)
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    shared = []
    i, j = n, m
    used_b = set()
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            shared.append(a[i - 1])
            used_b.add(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    shared.reverse()
    extra = [tok for idx, tok in enumerate(b) if idx not in used_b]
    return shared, extra
