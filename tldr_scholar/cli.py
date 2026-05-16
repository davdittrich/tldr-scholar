"""CLI entry point for tldr-scholar."""
from __future__ import annotations

import json as json_mod
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import typer
from loguru import logger

from tldr_scholar.types import AudienceEnum, ToneEnum
from tldr_scholar import summarize_file, summarize_url
from tldr_scholar.config import load_config
from tldr_scholar.ingest import (
    EmptyTextError,
    PasswordProtectedError,
    UnsupportedInputError,
)
from tldr_scholar.prompts import SENTENCE_COUNTS

app = typer.Typer(help="Academic text summarizer — PDF, URL, Markdown, text.")

_LENGTH_PRESETS = {"short": 200, "medium": 500, "long": 1000}


@app.command()
def main(
    source: str = typer.Argument(help="File path or URL to summarize"),
    length: str = typer.Option("medium", "--length", help="short|medium|long"),
    max_chars: Optional[int] = typer.Option(None, "--max-chars", help="Override length preset"),
    focus: str = typer.Option("main findings and novel insights", "--focus"),
    hashtags: int = typer.Option(0, "--hashtags", help="Number of hashtags to generate"),
    hashtag_style: str = typer.Option("lowercase", "--hashtag-style", help="lowercase|pascal"),
    audience: AudienceEnum = typer.Option(AudienceEnum.EXPERT, "--audience", help="Target audience"),
    tone: ToneEnum = typer.Option(ToneEnum.PROFESSIONAL, "--tone", help="Desired tone"),
    persona: Optional[str] = typer.Option(None, "--persona", help="Persona name (e.g. stitched)"),
    output_format: str = typer.Option("text", "--format", help="text|json|markdown"),
    backend: str = typer.Option("auto", "--backend",
                                help="gemini|lemonade|ollama|extractive|auto"),
    mode: str = typer.Option("scientific", "--mode", help="scientific|general"),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path"),
    gemini_timeout: Optional[int] = typer.Option(
        None, "--gemini-timeout",
        help="Override Gemini request timeout in seconds (default: 90)",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Summarize a PDF, URL, Markdown file, or text file."""
    # Validate backend
    valid_backends = {"auto", "gemini", "lemonade", "ollama", "extractive"}
    if backend not in valid_backends:
        typer.echo(f"Invalid backend '{backend}'. Choose from: {', '.join(sorted(valid_backends))}", err=True)
        raise typer.Exit(code=2)

    # Validate mode
    if mode not in ("scientific", "general"):
        typer.echo(f"Invalid mode '{mode}'. Choose: scientific, general", err=True)
        raise typer.Exit(code=2)

    # Validate format
    if output_format not in ("text", "json", "markdown"):
        typer.echo(f"Invalid format '{output_format}'. Choose: text, json, markdown", err=True)
        raise typer.Exit(code=2)

    # Validate hashtag style
    if hashtag_style not in ("lowercase", "pascal"):
        typer.echo(f"Invalid hashtag style '{hashtag_style}'. Choose: lowercase, pascal", err=True)
        raise typer.Exit(code=2)

    # Logging
    log_level = "DEBUG" if verbose else ("WARNING" if quiet else "INFO")
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Resolve max_chars
    if max_chars is not None:
        effective_max_chars = max_chars
    elif length in _LENGTH_PRESETS:
        effective_max_chars = _LENGTH_PRESETS[length]
    else:
        typer.echo(f"Invalid length '{length}'. Choose: short, medium, long", err=True)
        raise typer.Exit(code=2)

    # Compute sentence count from length preset
    length_key = "medium"
    if max_chars is not None:
        # Map max_chars to nearest length preset for sentence count
        if effective_max_chars <= 250:
            length_key = "short"
        elif effective_max_chars >= 750:
            length_key = "long"
    else:
        length_key = length
    sentence_count = SENTENCE_COUNTS.get(length_key, 5)

    # Load config — dict-of-dicts keyed by backend name so auto mode
    # doesn't merge conflicting hosts/models across backends
    config_path = config or os.environ.get("TLDR_SCHOLAR_CONFIG")
    backend_config: dict = {}
    if config_path:
        cfg = load_config(Path(config_path))
        backend_config = {
            "gemini": cfg.gemini.model_dump(),
            "lemonade": cfg.lemonade.model_dump(),
            "ollama": cfg.ollama.model_dump(),
            "oa": cfg.oa.model_dump(),
        }
    if gemini_timeout is not None:
        backend_config.setdefault("gemini", {})["timeout"] = gemini_timeout

    # Run summarization
    try:
        parsed = urlparse(source)
        kwargs = {
            "max_chars": effective_max_chars,
            "focus": focus,
            "hashtags": hashtags,
            "hashtag_style": hashtag_style,
            "audience": audience,
            "tone": tone,
            "backend": backend,
            "backend_config": backend_config,
            "mode": mode,
            "sentence_count": sentence_count,
            "persona": persona,
        }
        if parsed.scheme in ("http", "https"):
            result = summarize_url(source, **kwargs)
        else:
            result = summarize_file(source, **kwargs)
    except UnsupportedInputError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)
    except PasswordProtectedError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except EmptyTextError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    # Handle empty result
    if not result.text and backend != "auto":
        typer.echo(f"Error: backend '{backend}' returned empty response", err=True)
        raise typer.Exit(code=1)
    elif not result.text:
        typer.echo("Error: all backends failed to produce a summary", err=True)
        raise typer.Exit(code=1)

    # Output
    if output_format == "json":
        typer.echo(json_mod.dumps(result.model_dump(mode="json"), indent=2, default=str))
    elif output_format == "markdown":
        typer.echo(f"## Summary\n\n{result.text}")
        if result.hashtags:
            typer.echo(f"\n## Hashtags\n\n{' '.join(result.hashtags)}")
        if result.metadata.tokens_used is not None and not quiet:
            tok_prefix = "~" if result.metadata.tokens_estimated else ""
            typer.echo("\n## Usage\n")
            typer.echo(f"- {tok_prefix}Tokens: {result.metadata.tokens_used}")
            if result.metadata.cost_usd is not None:
                currency = result.metadata.cost_currency or "USD"
                cost_prefix = "~" if result.metadata.cost_estimated else ""
                typer.echo(f"- Cost: {cost_prefix}{currency} {result.metadata.cost_usd:.6f}")
    else:  # text
        typer.echo(result.text)
        if result.hashtags:
            typer.echo(" ".join(result.hashtags))
        if result.metadata.tokens_used is not None and not quiet:
            tok_prefix = "~" if result.metadata.tokens_estimated else ""
            parts = [f"{tok_prefix}Tokens: {result.metadata.tokens_used}"]
            if result.metadata.cost_usd is not None:
                currency = result.metadata.cost_currency or "USD"
                cost_prefix = "~" if result.metadata.cost_estimated else ""
                parts.append(f"Cost: {cost_prefix}{currency} {result.metadata.cost_usd:.6f}")
            typer.echo(" | ".join(parts))
