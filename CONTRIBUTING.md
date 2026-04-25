# Contributing to dspy-langchain-bridge

Thank you for your interest in contributing!

---

## Dev Environment Setup

```bash
git clone https://github.com/ANIMESHIOLOGY/dspy-langchain-bridge
cd dspy-langchain-bridge
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
# Unit tests only (no API calls)
pytest tests/unit/

# With coverage
pytest tests/unit/ --cov=dspy_lc_bridge

# Integration tests (requires API keys)
pytest tests/integration/ -m slow
```

## Code Style

- **Formatter**: `black` (line length 100)
- **Linter**: `ruff`
- **Types**: `mypy --strict`

Run all checks:

```bash
black .
ruff check .
mypy dspy_lc_bridge/
```

## Branch Naming

- `feature/<bridge-name>` — new bridge component
- `fix/<short-description>` — bug fix
- `docs/<topic>` — documentation only

## PR Checklist

- [ ] Unit tests written and passing
- [ ] Docstrings added for all public symbols (Google style)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] `mypy` passes with no new errors
- [ ] `ruff` and `black` pass

## Docstring Standard

Use **Google-style** docstrings:

```python
def signature_to_prompt(sig: dspy.Signature) -> PromptTemplate:
    """Convert a DSPy Signature to a LangChain PromptTemplate.

    Args:
        sig: A DSPy Signature with input and output fields.

    Returns:
        A LangChain PromptTemplate whose input_variables match the
        Signature's input fields.

    Example:
        >>> sig = dspy.Signature("question -> answer")
        >>> prompt = signature_to_prompt(sig)
        >>> prompt.input_variables
        ['question']
    """
```

## Building Docs Locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

---

Questions? Open a [Discussion](https://github.com/ANIMESHIOLOGY/dspy-langchain-bridge/discussions).
