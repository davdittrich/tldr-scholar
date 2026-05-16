"""tldr-scholar: Standalone academic text summarizer.

Public API:
    summarize(text=...) or summarize(request=SummaryRequest(...))
    summarize_file(path=...)
    summarize_url(url=...)
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from loguru import logger

from tldr_scholar.backends import run_with_fallback
from tldr_scholar.hashtags import (
    build_hashtag_instruction,
    generate_hashtags_tfidf,
    parse_hashtags_from_response,
)
from tldr_scholar.ingest import ingest
from tldr_scholar.models import (
    AudienceEnum,
    SummaryMetadata,
    SummaryRequest,
    SummaryResult,
    ToneEnum,
)

__all__ = [
    "summarize", "summarize_file", "summarize_url",
    "SummaryRequest", "SummaryResult", "SummaryMetadata",
    "AudienceEnum", "ToneEnum",
]


def _strip_url_credentials(url: str) -> str:
    """Remove userinfo (user:pass@) from URL for safe metadata storage."""
    parsed = urlparse(url)
    if parsed.username or parsed.password:
        hostname = parsed.hostname or ""
        netloc = f"{hostname}:{parsed.port}" if parsed.port else hostname
        return parsed._replace(netloc=netloc).geturl()
    return url


def summarize(
    text: str | None = None,
    *,
    request: SummaryRequest | None = None,
    max_chars: int = 500,
    focus: str = "main findings and novel insights",
    hashtags: int = 0,
    hashtag_style: str = "lowercase",
    audience: AudienceEnum | str = AudienceEnum.EXPERT,
    tone: ToneEnum | str = ToneEnum.PROFESSIONAL,
    backend: str = "auto",
    backend_config: dict[str, Any] | None = None,
    mode: str = "scientific",
    sentence_count: int = 5,
    persona: str | None = None,
) -> SummaryResult:
    """Summarize text. Accepts either text= or request=, not both.

    Returns SummaryResult with .text, .hashtags, and .metadata.
    """
    if text is not None and request is not None:
        raise ValueError("Pass either text= or request=, not both")
    if text is None and request is None:
        raise ValueError("text or request required")

    if request is not None:
        req = request
    else:
        req = SummaryRequest(
            text=text,
            max_chars=max_chars,
            focus=focus,
            hashtags=hashtags,
            hashtag_style=hashtag_style,  # type: ignore[arg-type]
            audience=audience,  # type: ignore[arg-type]
            tone=tone,          # type: ignore[arg-type]
            persona=persona,
            backend=backend,
            backend_config=backend_config or {},
        )

    hashtag_instruction = build_hashtag_instruction(req.hashtags, style=req.hashtag_style)
    response, backend_used, usage = run_with_fallback(
        text=req.text,
        max_chars=req.max_chars,
        focus=req.focus,
        hashtag_instruction=hashtag_instruction,
        backend=req.backend,
        audience=req.audience,
        tone=req.tone,
        config=req.backend_config if req.backend_config else None,
        mode=mode,
        sentence_count=sentence_count,
        persona=req.persona,
    )

    if not response:
        return SummaryResult(
            text="",
            hashtags=[],
            metadata=SummaryMetadata(
                backend_used=backend_used,
                max_chars=req.max_chars,
                focus=req.focus,
                audience=req.audience,
                tone=req.tone,
                persona=req.persona,
                hashtag_style=req.hashtag_style,
            ),
        )

    # Hashtag dispatch: extractive uses TF-IDF, LLM backends parse from response
    if backend_used == "extractive" and req.hashtags > 0:
        summary_text = response
        hashtag_list = generate_hashtags_tfidf(req.text, req.hashtags, style=req.hashtag_style)
    elif req.hashtags > 0:
        summary_text, hashtag_list = parse_hashtags_from_response(response)
        if len(hashtag_list) < req.hashtags:
            logger.debug(f"LLM returned {len(hashtag_list)} hashtags, requested {req.hashtags}")
    else:
        summary_text = response
        hashtag_list = []

    metadata = SummaryMetadata(
        backend_used=backend_used,
        max_chars=req.max_chars,
        focus=req.focus,
        char_count=len(summary_text.strip()),
        audience=req.audience,
        tone=req.tone,
        persona=req.persona,
        hashtag_style=req.hashtag_style,
    )
    if usage is not None:
        metadata.tokens_used = usage.tokens_used
        metadata.cost_usd = usage.cost_usd
        metadata.cost_currency = usage.cost_currency
        metadata.tokens_estimated = usage.tokens_estimated
        metadata.cost_estimated = usage.cost_estimated

    return SummaryResult(
        text=summary_text.strip(),
        hashtags=hashtag_list,
        metadata=metadata,
    )


def summarize_file(path: str, **kwargs) -> SummaryResult:
    """Summarize a local file (PDF, Markdown, or plain text)."""
    text, input_type = ingest(path)
    result = summarize(text=text, **kwargs)
    result.metadata.source = path
    result.metadata.input_type = input_type
    return result


def summarize_url(url: str, **kwargs) -> SummaryResult:
    """Summarize a web page or PDF URL."""
    backend_config = kwargs.get("backend_config", {})
    text, input_type = ingest(url, backend_config=backend_config)
    result = summarize(text=text, **kwargs)
    result.metadata.source = _strip_url_credentials(url)
    result.metadata.input_type = input_type
    return result
