# Contributing to tldr-scholar

Thank you for your interest in improving `tldr-scholar`! We welcome contributions to backends, prompt logic, ingestion strategies, and documentation.

## Development Setup

1.  Clone the repository.
2.  Install dependencies in editable mode:
    ```bash
    pip install -e ".[dev]"
    ```
3.  Run the test suite:
    ```bash
    pytest
    ```

## Quality Gates

We enforce high standards for code quality and correctness:

-   **TDD (Test-Driven Development)**: Always write a failing test before implementing a fix or feature.
-   **Coverage**: Maintain 100% test coverage for new code.
-   **Type Safety**: Use Pydantic models for data structures and avoid `Any` where possible.
-   **Style**: Follow standard Python (PEP 8) conventions.

## Adding a New Backend

1.  Create a new class in `tldr_scholar/backends/` that inherits from `BackendBase`.
2.  Implement the `summarize` method.
3.  Register the backend in `tldr_scholar/backends/__init__.py`.
4.  Add unit tests in `tests/test_backends.py`.

## Adding a New Persona

You don't need to modify the code to add a persona! Just create a YAML file in `~/.config/tldr-scholar/personas/` with the following schema:

```yaml
name: my-persona
role: "A description of the persona's role"
tone: "Adjectives describing the tone"
structure_pattern: "stitched" or "bullet_points"
hashtag_style: "lowercase" or "pascal"
```

## Pull Request Process

1.  Create a feature branch.
2.  Ensure all tests pass and coverage is maintained.
3.  Update the documentation and `SERVICE-INVENTORY.md` if applicable.
4.  Submit a PR with a clear description of the "Why" and "What".
