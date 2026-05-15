"""Hashtag generation: LLM response parsing + TF-IDF fallback."""
from __future__ import annotations

import re
from collections import Counter

from loguru import logger

_HASHTAG_RE = re.compile(r"#\w+")

# Common English stopwords (subset — no NLTK dependency)
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those",
    "it", "its", "we", "our", "they", "their", "not", "no", "as", "if",
    "so", "than", "more", "also", "very", "just", "about", "into", "over",
    "such", "only", "other", "new", "some", "each", "which", "when",
    "where", "how", "all", "both", "through", "between", "after", "before",
    "during", "up", "out", "then", "here", "there", "what", "who", "whom",
})

_ACRONYMS = {"ai", "nlp", "ml", "llm", "cpu", "gpu", "tpu", "api"}


def format_pascal_case(text: str) -> str:
    """Format text into #PascalCase.
    
    Examples:
    - "machine learning" -> "#MachineLearning"
    - "ai" -> "#AI"
    - "web3" -> "#Web3"
    """
    if not text:
        return ""
    
    # Strip non-alphanumeric (except underscores which we treat as word boundaries)
    clean = re.sub(r"[^a-zA-Z0-9_]", " ", text)
    parts = clean.replace("_", " ").split()
    
    formatted_parts = []
    for part in parts:
        lower = part.lower()
        if lower in _ACRONYMS:
            formatted_parts.append(lower.upper())
        else:
            # Capitalize first letter, keep rest as-is if it has existing caps
            # or lowercase if it was all lower
            if part.islower():
                formatted_parts.append(part.capitalize())
            else:
                formatted_parts.append(part[0].upper() + part[1:])
                
    return "#" + "".join(formatted_parts)


def build_hashtag_instruction(n: int, style: str = "lowercase") -> str:
    """Build the prompt instruction for LLM hashtag generation.

    Returns empty string when n == 0.
    """
    if n <= 0:
        return ""
    
    if style == "pascal":
        format_ex = "#MachineLearning #ArtificialIntelligence"
        case_instr = "PascalCase (e.g., #MachineLearning)"
    else:
        format_ex = "#machine_learning #climate_change"
        case_instr = "lowercase (e.g., #machine_learning)"
        
    return (
        f"After the summary, on a new line, generate exactly {n} hashtags. "
        f"Each hashtag should be 1-2 words, {case_instr}, commonly used on academic "
        "social media, and descriptive of the text's key topics. "
        f"Format: {format_ex} ..."
    )


def parse_hashtags_from_response(response: str) -> tuple[str, list[str]]:
    """Split LLM response into (summary_text, hashtags_list).

    Finds the last line containing hashtags (matching #word pattern).
    Returns (full_response, []) if no hashtag line is found.
    """
    if not response:
        return "", []

    lines = response.strip().split("\n")

    # Search from the end for a line that looks like hashtags
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        tags = _HASHTAG_RE.findall(line)
        if tags and len(tags) >= 2:  # at least 2 hashtags on one line
            summary = "\n".join(lines[:i]).strip()
            # Preserve case if it looks like PascalCase
            hashtags = [t[:30] for t in tags]
            return summary, hashtags

    # No hashtag line found
    return response.strip(), []


def generate_hashtags_tfidf(text: str, n: int, style: str = "lowercase") -> list[str]:
    """Generate hashtags from text using TF-IDF heuristic with bigram support.

    No NLTK dependency — uses a built-in stopword list and simple tokenization.
    Returns hashtags as lowercase #-prefixed strings, max 30 chars each.
    """
    if n <= 0 or not text:
        return []

    # unigrams
    unigrams = re.findall(r"[a-zA-Z]{3,}", text.lower())
    unigrams = [w for w in unigrams if w not in _STOPWORDS]

    # bigrams
    bigrams = []
    all_words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    for i in range(len(all_words) - 1):
        w1, w2 = all_words[i], all_words[i + 1]
        if w1 not in _STOPWORDS and w2 not in _STOPWORDS:
            bigrams.append(f"{w1}_{w2}")

    all_terms = unigrams + bigrams
    if not all_terms:
        return []

    # TF (term frequency)
    tf = Counter(all_terms)
    total = len(all_terms)

    # Heuristic: boost capitalized terms and multi-word terms
    capitalized = set(re.findall(r"\b[A-Z][a-z]{2,}", text))
    cap_lower = {w.lower() for w in capitalized}

    scored: list[tuple[str, float]] = []
    for term, count in tf.items():
        score = count / total
        
        # Boost unigrams that were capitalized
        if term in cap_lower:
            score *= 2.0
            
        # Boost bigrams (inherently more descriptive)
        if "_" in term:
            score *= 1.5
            
        scored.append((term, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Selection: ensure we don't pick a unigram that is already part of a top bigram
    top_terms = []
    seen_parts = set()
    for term, _ in scored:
        if len(top_terms) >= n:
            break
            
        if "_" in term:
            parts = term.split("_")
            top_terms.append(term)
            seen_parts.update(parts)
        else:
            if term not in seen_parts:
                top_terms.append(term)
                seen_parts.add(term)

    if style == "pascal":
        return [format_pascal_case(t)[:30] for t in top_terms]
    return [f"#{t}"[:30] for t in top_terms]
